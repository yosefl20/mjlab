"""YAML profile overrides for Soulmade Toto velocity configs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

_PROFILE_DIR = Path(__file__).with_name("profiles")
_DEFAULT_PROFILE = "stable"


def _as_range(value: list[float] | tuple[float, float]) -> tuple[float, float]:
  return (float(value[0]), float(value[1]))


def _apply_noise(cfg: ManagerBasedRlEnvCfg, group: str, term: str, value) -> None:
  if group not in cfg.observations or term not in cfg.observations[group].terms:
    return
  noise = cfg.observations[group].terms[term].noise
  assert noise is not None
  noise.n_min, noise.n_max = _as_range(value)


def _apply_reward_params(cfg: ManagerBasedRlEnvCfg, name: str, values: dict) -> None:
  if name not in cfg.rewards:
    return
  params = cfg.rewards[name].params
  for key, value in values.items():
    params[key] = value


def _load_profile(profile: str | None = None) -> dict[str, Any]:
  profile = profile or os.environ.get("MJLAB_TOTO_PROFILE", _DEFAULT_PROFILE)
  if profile.lower() in ("", "none", "off"):
    return {}

  path = _PROFILE_DIR / f"{profile}.yaml"
  if not path.exists():
    available = sorted(p.stem for p in _PROFILE_DIR.glob("*.yaml"))
    raise FileNotFoundError(
      f"Toto profile '{profile}' not found at {path}. Available profiles: {available}"
    )

  with path.open() as f:
    data = yaml.safe_load(f) or {}
  if not isinstance(data, dict):
    raise ValueError(f"Toto profile '{path}' must contain a YAML mapping.")
  return data


def apply_toto_profile(
  cfg: ManagerBasedRlEnvCfg,
  profile: str | None = None,
) -> None:
  """Apply a Toto YAML tuning profile to an already-created env config."""
  data = _load_profile(profile)

  observations = data.get("observations", {})
  if "base_ang_vel_noise" in observations:
    _apply_noise(cfg, "actor", "base_ang_vel", observations["base_ang_vel_noise"])
  if "projected_gravity_noise" in observations:
    _apply_noise(
      cfg,
      "actor",
      "projected_gravity",
      observations["projected_gravity_noise"],
    )
  if "joint_pos_noise" in observations:
    _apply_noise(cfg, "actor", "joint_pos", observations["joint_pos_noise"])
  if "joint_vel_noise" in observations:
    _apply_noise(cfg, "actor", "joint_vel", observations["joint_vel_noise"])
  if "height_scan_noise" in observations:
    _apply_noise(cfg, "actor", "height_scan", observations["height_scan_noise"])
  if "critic_base_lin_vel_noise" in observations:
    _apply_noise(
      cfg,
      "critic",
      "base_lin_vel",
      observations["critic_base_lin_vel_noise"],
    )
  # if "phase_period" in observations:
  #   period = float(observations["phase_period"])
  #   cfg.observations["actor"].terms["phase"].params["period"] = period
  #   if "foot_gait" in cfg.rewards:
  #     cfg.rewards["foot_gait"].params["period"] = period

  events = data.get("events", {})
  if "reset_base" in events:
    reset_base = events["reset_base"]
    if "pose_range" in reset_base:
      cfg.events["reset_base"].params["pose_range"] = {
        axis: _as_range(value) for axis, value in reset_base["pose_range"].items()
      }
    if "velocity_range" in reset_base:
      cfg.events["reset_base"].params["velocity_range"] = {
        axis: _as_range(value) for axis, value in reset_base["velocity_range"].items()
      }
  if "reset_robot_joints" in events:
    reset_joints = events["reset_robot_joints"]
    if "position_range" in reset_joints:
      cfg.events["reset_robot_joints"].params["position_range"] = _as_range(
        reset_joints["position_range"]
      )
    if "velocity_range" in reset_joints:
      cfg.events["reset_robot_joints"].params["velocity_range"] = _as_range(
        reset_joints["velocity_range"]
      )
  if "encoder_bias" in events:
    cfg.events["encoder_bias"].params["bias_range"] = _as_range(
      events["encoder_bias"]
    )
  if "foot_friction" in events:
    cfg.events["foot_friction"].params["ranges"] = _as_range(
      events["foot_friction"]
    )
  if "base_com" in events:
    base_com = events["base_com"]
    cfg.events["base_com"].params["ranges"] = {
      0: _as_range(base_com["x"]),
      1: _as_range(base_com["y"]),
      2: _as_range(base_com["z"]),
    }
  if "push_robot" in events and "push_robot" in cfg.events:
    push_robot = events["push_robot"]
    if "interval_range_s" in push_robot:
      cfg.events["push_robot"].interval_range_s = _as_range(
        push_robot["interval_range_s"]
      )
    velocity_range = push_robot.get("velocity_range", push_robot)
    cfg.events["push_robot"].params["velocity_range"] = {
      axis: _as_range(value)
      for axis, value in velocity_range.items()
      if axis != "interval_range_s"
    }

  commands = data.get("commands", {})
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  if "resampling_time_range" in commands:
    twist_cmd.resampling_time_range = _as_range(commands["resampling_time_range"])
  for name in (
    "rel_standing_envs",
    "rel_heading_envs",
    "rel_forward_envs",
    "heading_control_stiffness",
  ):
    if name in commands:
      setattr(twist_cmd, name, float(commands[name]))
  for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
    if name in commands:
      setattr(twist_cmd.ranges, name, _as_range(commands[name]))
  if "velocity_stages" in commands and "command_vel" in cfg.curriculum:
    stages = []
    for stage in commands["velocity_stages"]:
      stages.append(
        {
          key: (int(value) if key == "step" else _as_range(value))
          for key, value in stage.items()
        }
      )
    cfg.curriculum["command_vel"].params["velocity_stages"] = stages

  rewards = data.get("rewards", {})
  reward_weights = rewards.get("weights", rewards)
  for name, weight in reward_weights.items():
    if name == "params":
      continue
    if name in cfg.rewards:
      cfg.rewards[name].weight = float(weight)

  reward_params = rewards.get("params", {})
  for name, values in reward_params.items():
    _apply_reward_params(cfg, name, values)
