from mjlab.asset_zoo.robots import (
  TOTO_ACTION_SCALE,
  get_toto_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.tasks.velocity.config.toto.profile_utils import apply_toto_profile


def soulmade_toto_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Toto rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70

  cfg.scene.entities = {"robot": get_toto_robot_cfg()}

  # Set raycast sensor frame to Toto pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "base_link"

  site_names = ("left_foot_site", "right_foot_site")
  geom_names = (r"left_foot_collision_.*", r"right_foot_collision_.*")



  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_pitch_link|right_ankle_pitch_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = TOTO_ACTION_SCALE

  cfg.viewer.body_name = "base_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15
  # cfg.curriculum.pop("command_vel", None)

  twist_cmd.ranges.lin_vel_x = (-0.5, 0.5)  # Forward/backward: ±0.5 m/s (was ±1.0)
  twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)  # Left/right: ±0.3 m/s (was ±1.0)
  twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)  # Rotation: ±0.3 rad/s (was ±0.5)

  # cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names

  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # - Wrists are loose (0.3) since they don't affect balance much.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.02}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_yaw.*": 0.05,
    r".*hip_roll.*": 0.1,
    r".*hip_pitch.*": 0.2,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.3,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_yaw.*": 0.02,
    r".*hip_roll.*": 0.02,
    r".*hip_pitch.*": 0.02,
    r".*knee.*": 0.02,
    r".*ankle_pitch.*": 0.02,
  }
  cfg.rewards["pose"].weight = 1.0

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)
  # cfg.rewards["upright"].params["std"] = 0.2
  
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # cfg.observations["actor"].terms["phase"].params["period"] = 0.52
  # cfg.rewards["foot_gait"].weight = 1.0
  # cfg.rewards["foot_gait"].params["period"] = 0.52
  # cfg.rewards["foot_gait"].params["threshold"] = 0.7

  cfg.rewards["air_time"].weight = 1.0
  cfg.rewards["air_time"].params["threshold_min"] = 0.15
  cfg.rewards["air_time"].params["threshold_max"] = 0.30

  cfg.rewards["foot_clearance"].weight = -1.0
  cfg.rewards["foot_clearance"].params["target_height"] = 0.04

  cfg.rewards["foot_swing_height"].params["target_height"] = 0.04
  cfg.rewards["foot_swing_height"].weight = -0.1

  cfg.rewards["angular_momentum"].weight = -0.01
  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["action_rate_l2"].weight = -0.2

  apply_toto_profile(cfg)

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def soulmade_toto_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity configuration."""
  cfg =soulmade_toto_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.terminations.pop("out_of_terrain_bounds", None)

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  return cfg
