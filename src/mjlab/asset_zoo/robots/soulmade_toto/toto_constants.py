"""soulmade Toto constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia,
)
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

Toto_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "soulmade_toto" / "xmls" / "toto.xml"
)
assert Toto_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(Toto_XML))


##
# Actuator config.
##
# Toto uses actuator type 02 for all joints.
ROTOR_INERTIA_02 = 6.99e-5
GEAR_RATIO_02 = 7.75
LOW_SPEED_EQUIVALENT_INERTIA_02 = 4.2e-3
POSITION_LIMIT_02 = (-12.57, 12.57)
VELOCITY_LIMIT_02 = 44.0
RATED_EFFORT_02 = 6.0
EFFORT_LIMIT_02 = 17.0
TORQUE_CONSTANT_02 = 1.22  # N.m / Arms
RATED_PHASE_CURRENT_02 = 7.0  # Apk
MAX_PHASE_CURRENT_02 = 23.0  # Apk
NOMINAL_VOLTAGE_02 = 48.0  # VDC
MAX_VOLTAGE_02 = 60.0  # VDC
ARMATURE_02 = reflected_inertia(ROTOR_INERTIA_02, GEAR_RATIO_02)
assert abs(ARMATURE_02 - LOW_SPEED_EQUIVALENT_INERTIA_02) < 1e-4

ACTUATOR_02 = ElectricActuator(
  reflected_inertia=ARMATURE_02,
  velocity_limit=VELOCITY_LIMIT_02,
  effort_limit=EFFORT_LIMIT_02,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 1.0

STIFFNESS_02 = ARMATURE_02 * NATURAL_FREQ**2
DAMPING_02 = 2.0 * DAMPING_RATIO * ARMATURE_02 * NATURAL_FREQ

# Toto Actuator Configuration
# All Toto joints use actuator type 02.

TOTO_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw_link_joint",),
  stiffness=STIFFNESS_02,
  damping=DAMPING_02,
  effort_limit=ACTUATOR_02.effort_limit,
  armature=ACTUATOR_02.reflected_inertia,
)

TOTO_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_link_joint",),
  stiffness=STIFFNESS_02,
  damping=DAMPING_02,
  effort_limit=ACTUATOR_02.effort_limit,
  armature=ACTUATOR_02.reflected_inertia,
)

TOTO_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_link_joint",),
  stiffness=STIFFNESS_02,
  damping=DAMPING_02,
  effort_limit=ACTUATOR_02.effort_limit,
  armature=ACTUATOR_02.reflected_inertia,
)

TOTO_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee_pitch_link_joint",),
  stiffness=STIFFNESS_02,
  damping=DAMPING_02,
  effort_limit=ACTUATOR_02.effort_limit,
  armature=ACTUATOR_02.reflected_inertia,
)

TOTO_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_link_joint",),
  stiffness=STIFFNESS_02,
  damping=DAMPING_02,
  effort_limit=ACTUATOR_02.effort_limit,
  armature=ACTUATOR_02.reflected_inertia,
)
##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.172),
  joint_pos={
    'left_hip_yaw_link_joint': 0.0,
    'left_hip_roll_link_joint': -0.0,
    'left_hip_pitch_link_joint': 0.0,
    'left_knee_pitch_link_joint': 0.5,
    'left_ankle_pitch_link_joint': -0.5,
    'right_hip_yaw_link_joint': 0.0,
    'right_hip_roll_link_joint': 0.0,
    'right_hip_pitch_link_joint': 0.0,
    'right_knee_pitch_link_joint': 0.5,
    'right_ankle_pitch_link_joint': -0.5,
  },
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.172),
  joint_pos={
    'left_hip_yaw_link_joint': 0.0,
    'left_hip_roll_link_joint': -0.0,
    'left_hip_pitch_link_joint': 0.0,
    'left_knee_pitch_link_joint': 0.5,
    'left_ankle_pitch_link_joint': -0.5,
    'right_hip_yaw_link_joint': 0.0,
    'right_hip_roll_link_joint': 0.0,
    'right_hip_pitch_link_joint': 0.0,
    'right_knee_pitch_link_joint': 0.5,
    'right_ankle_pitch_link_joint': -0.5,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##



# FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
#   geom_names_expr=(".*_collision",),
#   contype=0,
#   conaffinity=1,
#   condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
#   priority={r"^(left|right)_foot[1-7]_collision$": 1},
#   friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
# )


FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",), # 匹配所有 class="collision" 的 geom
  contype=1,
  conaffinity=1,
  condim={".*_foot_collision": 3, ".*": 1}, # 脚踝 3D 接触，身体其他部分 1D 接触（防止穿透即可）
  priority={".*_foot_collision": 1},
  friction={".*_foot_collision": (0.8,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={".*_foot_collision": 3, ".*": 1},
  priority={".*_foot_collision": 1},
  friction={".*_foot_collision": (0.8,)},
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(".*_foot_collision_*",), # 匹配我们在 XML 中起的名字
  contype=0,
  conaffinity=1,
  condim=3,      # 脚部需要 3D 接触（摩擦力）
  priority=1,
  friction=(0.8,), # 地面摩擦系数
)

##
# Final config.
##

TOTO_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    TOTO_ACTUATOR_HIP_YAW,
    TOTO_ACTUATOR_HIP_ROLL,
    TOTO_ACTUATOR_HIP_PITCH,
    TOTO_ACTUATOR_KNEE,
    TOTO_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_toto_robot_cfg() -> EntityCfg:

  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FEET_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=TOTO_ARTICULATION,
  )


TOTO_ACTION_SCALE: dict[str, float] = {}
for a in TOTO_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    TOTO_ACTION_SCALE[n] = 0.2 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_toto_robot_cfg())

  viewer.launch(robot.spec.compile())
