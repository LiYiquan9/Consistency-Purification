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
from consistency_models_cifar10.configs.cifar10_purification_edm_onestep import get_config
import math

import torch

from PIL import Image

class EDM_onestep_cifar10():
    def __init__(self, sigma = 0.25,checkpoint_dir="path_to_checkpoint",checkpoint_steps=40):
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
        
        self.state = checkpoints.restore_checkpoint(checkpoint_dir, self.state, step=checkpoint_steps)
        self.pstate = flax.jax_utils.replicate(self.state)
        
        self.model_fn = mutils.get_denoiser_fn(
            self.sde,
            self.score_model,
            self.state.params_ema,
            self.state.model_state,
            train=False,
            return_state=False,
        )
        
        self.sigma = sigma
        scaled_sigma = self.sigma * 2
        num_steps = self.config.sampling.n_steps
        sigma_max = self.sde.t_max
        sigma_min = self.sde.t_min
        rho = self.sde.rho
        
        step_indices = torch.arange(num_steps, dtype=torch.float32)
        
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
        for i in range(len(t_steps)-1):
            if t_steps[i]>=scaled_sigma and t_steps[i+1]<scaled_sigma:
                if t_steps[i]-scaled_sigma > scaled_sigma-t_steps[i+1]:
                    t = i+1
                    break
                else:
                    t = i
                    break
            t = len(t_steps)-1
    
        self.noise_level = t_steps[t].item()
        
        
    def image_editing_sample(self, img=None, bs_id=0, tag=None):
        """
        img: range [-1,1]
        """

        img = jnp.transpose(img.cpu().numpy(), (0, 2, 3, 1))
            
        samples = self.model_fn(img, jnp.ones((img.shape[0],)) * self.noise_level)
        
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
        