from mjlab.asset_zoo.robots.soulmade_toto.toto_constants import get_toto_robot_cfg

from mjlab.entity import Entity

# Load the robot
robot = Entity(get_toto_robot_cfg())
model = robot.spec.compile()

# Display robot information
print("✓ toto robot loaded successfully!")
print(f"  • Degrees of Freedom (DOF): {model.nv}")
print(f"  • Number of Actuators: {model.nu}")
print(f"  • Bodies: {model.nbody}")
print(f"  • Joints: {model.njnt}")