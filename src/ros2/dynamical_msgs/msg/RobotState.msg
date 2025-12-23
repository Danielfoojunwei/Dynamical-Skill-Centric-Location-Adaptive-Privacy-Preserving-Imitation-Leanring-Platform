# Complete robot state message
# Published at 1kHz from state estimator

std_msgs/Header header

# Joint state
float64[] joint_positions      # radians
float64[] joint_velocities     # rad/s
float64[] joint_torques        # Nm
float64[] joint_temperatures   # Celsius

# End-effector state
geometry_msgs/Pose ee_pose
geometry_msgs/Twist ee_velocity

# Base state (for mobile robots)
geometry_msgs/Pose base_pose
geometry_msgs/Twist base_velocity

# Gripper state
float64 gripper_position       # 0.0 (closed) to 1.0 (open)
float64 gripper_force          # N

# Status flags
bool state_valid
bool in_motion
bool at_home_position
