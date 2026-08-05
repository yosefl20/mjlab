"""YAML profile overrides for Soulmade Toto RSL-RL configs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from mjlab.rl import RslRlOnPolicyRunnerCfg

_PROFILE_DIR = Path(__file__).with_name("agent_profiles")
_DEFAULT_PROFILE = "stable"


def _as_tuple(value: list[Any] | tuple[Any, ...]) -> tuple[Any, ...]:
  return tuple(value)


def _load_profile(profile: str | None = None) -> dict[str, Any]:
  profile = profile or os.environ.get("MJLAB_TOTO_AGENT_PROFILE", _DEFAULT_PROFILE)
  if profile.lower() in ("", "none", "off"):
    return {}

  path = _PROFILE_DIR / f"{profile}.yaml"
  if not path.exists():
    available = sorted(p.stem for p in _PROFILE_DIR.glob("*.yaml"))
    raise FileNotFoundError(
      f"Toto agent profile '{profile}' not found at {path}. "
      f"Available profiles: {available}"
    )

  with path.open() as f:
    data = yaml.safe_load(f) or {}
  if not isinstance(data, dict):
    raise ValueError(f"Toto agent profile '{path}' must contain a YAML mapping.")
  return data


def apply_toto_agent_profile(
  cfg: RslRlOnPolicyRunnerCfg,
  profile: str | None = None,
) -> None:
  """Apply a Toto YAML tuning profile to an already-created RSL-RL config."""
  data = _load_profile(profile)

  runner = data.get("runner", {})
  tuple_fields = {"wandb_tags"}
  for key, value in runner.items():
    if hasattr(cfg, key):
      setattr(cfg, key, _as_tuple(value) if key in tuple_fields else value)

  actor = data.get("actor", {})
  for key, value in actor.items():
    if hasattr(cfg.actor, key):
      setattr(cfg.actor, key, _as_tuple(value) if key == "hidden_dims" else value)

  critic = data.get("critic", {})
  for key, value in critic.items():
    if hasattr(cfg.critic, key):
      setattr(cfg.critic, key, _as_tuple(value) if key == "hidden_dims" else value)

  algorithm = data.get("algorithm", {})
  for key, value in algorithm.items():
    if hasattr(cfg.algorithm, key):
      setattr(cfg.algorithm, key, value)
