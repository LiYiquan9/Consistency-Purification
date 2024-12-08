import flax
import jax
import jax.numpy as jnp
import numpy as np
import logging
import functools
import haiku as hk

from consistency_models_cifar10.jcm.models import utils as mutils
from consistency_models_cifar10.jcm import sde_lib
from consistency_models_cifar10.jcm import sampling
from consistency_models_cifar10.jcm import checkpoints
from consistency_models_cifar10.jcm.models import ncsnpp
from consistency_models_cifar10.jcm import losses
from consistency_models_cifar10.configs.cifar10_purification_edm_multistep import get_config
import math

import torch


class EDM_multistep_cifar10():
    def __init__(self, sigma = 0.25, checkpoint_dir= "path_to_checkpoint",checkpoint_steps=40):
        self.config = get_config()
        self.rng = hk.PRNGSequence(self.config.seed + 1)
     
        self.score_model, init_model_state, initial_params = mutils.init_model(next(self.rng), self.config)
        optimizer, optimize_fn = losses.get_optimizer(self.config)
        if self.config.training.loss.lower().endswith(
            ("ema", "adaptive", "progressive_distillation")
        ):
            self.state = mutils.StateWithTarget(
                step=0,
                lr=self.config.optim.lr,
                ema_rate=self.config.model.ema_rate,
                params=initial_params,
                target_params=initial_params,
                params_ema=initial_params,
                model_state=init_model_state,
                opt_state=optimizer.init(initial_params),
                rng_state=self.rng.internal_state,
            )
        else:
            self.state = mutils.State(
                step=0,
                lr=self.config.optim.lr,
                ema_rate=self.config.model.ema_rate,
                params=initial_params,
                params_ema=initial_params,
                model_state=init_model_state,
                opt_state=optimizer.init(initial_params),
                rng_state=self.rng.internal_state,
            )

        self.sde = sde_lib.get_sde(self.config)
        self.sampling_shape = (
            self.config.eval.batch_size // jax.local_device_count(),
            self.config.data.image_size,
            self.config.data.image_size,
            self.config.data.num_channels,
        )
        
        if self.config.eval.enable_loss:
            # Create the one-step evaluation function when loss computation is enabled
            train_loss_fn, eval_loss_fn, self.state = losses.get_loss_fn(
                self.config, self.sde, self.score_model, self.state, next(self.rng)
            )

        self.checkpoint_dir = checkpoint_dir
        self.state = checkpoints.restore_checkpoint(self.checkpoint_dir, self.state, step=checkpoint_steps)
        self.pstate = flax.jax_utils.replicate(self.state)
        
        self.denoiser_fn = mutils.get_denoiser_fn(
            self.sde,
            self.score_model,
            self.state.params_ema,
            self.state.model_state,
            train=False,
            return_state=False,
        )
        
        self.sigma = sigma
        scaled_sigma = self.sigma * 2
        self.num_steps = self.config.sampling.n_steps
        sigma_max = self.sde.t_max
        sigma_min = self.sde.t_min
        rho = self.sde.rho
        
        step_indices = torch.arange(self.num_steps, dtype=torch.float32)
        
        self.t_steps = (sigma_max ** (1 / rho) + step_indices / (self.num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        self.t_steps = torch.cat([self.t_steps, torch.zeros_like(self.t_steps[:1])]) # t_N = 0
        for i in range(len(self.t_steps)-1):
            if self.t_steps[i]>=scaled_sigma and self.t_steps[i+1]<scaled_sigma:
                if self.t_steps[i]-scaled_sigma > scaled_sigma-self.t_steps[i+1]:
                    t = i+1
                    break
                else:
                    t = i
                    break
            t = len(self.t_steps)-1
        self.t_pos = t
        
    def image_editing_sample(self, img=None, bs_id=0, tag=None):
        """
        img: range [-1,1]
        """
        
        timesteps = self.t_steps
        
        img = jnp.transpose(img.cpu().numpy(), (0, 2, 3, 1))

        x = img
        for i in range(self.t_pos, self.num_steps-1):
            
            t = timesteps[i]
            t_jax = jnp.array(t.numpy())
            
            vec_t = jnp.ones((img.shape[0],)) * t_jax
           
            denoiser = self.denoiser_fn(x, vec_t)
            d = 1 / t_jax * x - 1 / t_jax * denoiser
            next_t = timesteps[i + 1]
            next_t_jax = jnp.array(next_t.numpy())
            samples = x + (next_t_jax - t_jax) * d

            vec_next_t = jnp.ones((img.shape[0],)) * next_t_jax
            denoiser = self.denoiser_fn(samples, vec_next_t)
            next_d = 1 / next_t_jax * samples - 1 / next_t_jax * denoiser
            samples = x + (next_t_jax - t_jax) / 2 * (d + next_d)

            x = samples   
    
        t = jnp.array(timesteps[self.num_steps - 1].numpy())
        
        vec_t = jnp.ones((img.shape[0],)) * t
        denoiser = self.denoiser_fn(x, vec_t)
        d = 1 / t * x - 1 / t * denoiser
        next_t =  jnp.array(timesteps[self.num_steps].numpy())
        
        samples = x + (next_t - t) * d

        samples = samples.reshape(
            (
                -1,
                self.config.data.image_size,
                self.config.data.image_size,
                self.config.data.num_channels,
            )
        )
        x0 = samples 
        x0 = torch.from_numpy(np.array(x0)).permute(0, 3, 1, 2).cuda()

        
        return x0
        
