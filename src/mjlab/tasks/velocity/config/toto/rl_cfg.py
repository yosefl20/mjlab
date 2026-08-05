"""RL configuration for Unitree G1 velocity task."""

from mjlab.tasks.velocity.config.toto.agent_profile_utils import (
  apply_toto_agent_profile,
)
from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def solumade_toto_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Toto velocity task."""
  cfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="toto_velocity",
    save_interval=200,
    num_steps_per_env=24,
    max_iterations=10_000,
  )
  apply_toto_agent_profile(cfg)
  return cfg
