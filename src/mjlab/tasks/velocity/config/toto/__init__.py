from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  soulmade_toto_flat_env_cfg,
  soulmade_toto_rough_env_cfg,
)
from .rl_cfg import solumade_toto_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Soulmade-Toto",
  env_cfg=soulmade_toto_rough_env_cfg(),
  play_env_cfg=soulmade_toto_rough_env_cfg(play=True),
  rl_cfg=solumade_toto_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Soulmade-Toto",
  env_cfg=soulmade_toto_flat_env_cfg(),
  play_env_cfg=soulmade_toto_flat_env_cfg(play=True),
  rl_cfg=solumade_toto_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
