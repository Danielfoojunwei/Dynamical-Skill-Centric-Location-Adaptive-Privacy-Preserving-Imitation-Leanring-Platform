cmake_minimum_required(VERSION 3.8)
project(dynamical_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

# Generate messages, services, and actions
rosidl_generate_interfaces(${PROJECT_NAME}
  # Messages
  "msg/RobotState.msg"
  "msg/JointState.msg"
  "msg/SafetyStatus.msg"
  "msg/SafetyViolation.msg"
  "msg/PerceptionFeatures.msg"
  "msg/Detection.msg"
  "msg/SkillStatus.msg"
  "msg/SystemStatus.msg"
  "msg/TelemetryData.msg"

  # Services
  "srv/ExecuteSkill.srv"
  "srv/StopSkill.srv"
  "srv/GetRobotState.srv"
  "srv/SetControlMode.srv"
  "srv/TriggerEstop.srv"
  "srv/ResetEstop.srv"
  "srv/LoadSkill.srv"
  "srv/GetSystemStatus.srv"

  # Actions
  "action/ExecuteSkillAction.action"
  "action/MoveToPosition.action"

  DEPENDENCIES
    std_msgs
    geometry_msgs
    sensor_msgs
    trajectory_msgs
    builtin_interfaces
)

ament_package()
