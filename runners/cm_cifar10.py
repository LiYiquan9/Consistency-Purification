import flax
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk

from consistency_models_cifar10.jcm.models import utils as mutils
from consistency_models_cifar10.jcm import sde_lib
from consistency_models_cifar10.jcm import checkpoints
from consistency_models_cifar10.jcm import losses
from consistency_models_cifar10.configs.cifar10_purification_cm import get_config
import torch


from consistency_models_cifar10.configs.cifar10_k_ve import get_config as get_ref_config

class CM_cifar10():
    def __init__(self, sigma=0.25,checkpoint_dir="path_to_checkpoint",checkpoint_steps=80):
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
             
        self.checkpoint_dir = checkpoint_dir 
        self.state = checkpoints.restore_checkpoint(self.checkpoint_dir, self.state, step=checkpoint_steps)
        self.pstate = flax.jax_utils.replicate(self.state)
        
        self.model_fn = mutils.get_distiller_fn(
            self.sde,
            self.score_model,
            self.state.params_ema,
            self.state.model_state,
            train=False,
            return_state=False,
        )
        
        indices_arrange = np.arange(0, self.config.sampling.n_steps)
        t_arrange = (self.sde.t_max ** (1 / self.sde.rho) + indices_arrange / (self.config.sampling.n_steps - 1) * (self.sde.t_min ** (1 / self.sde.rho) - self.sde.t_max ** (1 / self.sde.rho)))**self.sde.rho
        closest_index = np.argmin(np.abs(t_arrange - sigma*2))
        self.noise_level = t_arrange[closest_index]

        
    def image_editing_sample(self, img=None, bs_id=0, tag=None):
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