cmake_minimum_required(VERSION 3.8)
project(dynamical_runtime)

# C++17 for modern features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(dynamical_msgs REQUIRED)
find_package(realtime_tools REQUIRED)

# Include directories
include_directories(include)

# =============================================================================
# Safety Node (1kHz, real-time)
# =============================================================================
add_library(safety_node SHARED
  src/safety_node.cpp
)
target_include_directories(safety_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(safety_node
  rclcpp
  rclcpp_components
  rclcpp_lifecycle
  std_msgs
  sensor_msgs
  geometry_msgs
  dynamical_msgs
  realtime_tools
)
rclcpp_components_register_node(safety_node
  PLUGIN "dynamical::SafetyNode"
  EXECUTABLE safety_node_exec
)

# =============================================================================
# State Estimator Node (1kHz)
# =============================================================================
add_library(state_estimator_node SHARED
  src/state_estimator_node.cpp
)
target_include_directories(state_estimator_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(state_estimator_node
  rclcpp
  rclcpp_components
  std_msgs
  sensor_msgs
  geometry_msgs
  dynamical_msgs
)
rclcpp_components_register_node(state_estimator_node
  PLUGIN "dynamical::StateEstimatorNode"
  EXECUTABLE state_estimator_node_exec
)

# =============================================================================
# Actuator Node (1kHz)
# =============================================================================
add_library(actuator_node SHARED
  src/actuator_node.cpp
)
target_include_directories(actuator_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(actuator_node
  rclcpp
  rclcpp_components
  rclcpp_lifecycle
  std_msgs
  sensor_msgs
  trajectory_msgs
  dynamical_msgs
  realtime_tools
)
rclcpp_components_register_node(actuator_node
  PLUGIN "dynamical::ActuatorNode"
  EXECUTABLE actuator_node_exec
)

# =============================================================================
# Skill Executor Node
# =============================================================================
add_library(skill_executor_node SHARED
  src/skill_executor_node.cpp
)
target_include_directories(skill_executor_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(skill_executor_node
  rclcpp
  rclcpp_components
  rclcpp_action
  std_msgs
  dynamical_msgs
)
rclcpp_components_register_node(skill_executor_node
  PLUGIN "dynamical::SkillExecutorNode"
  EXECUTABLE skill_executor_node_exec
)

# =============================================================================
# Install
# =============================================================================
install(TARGETS
  safety_node
  state_estimator_node
  actuator_node
  skill_executor_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

# =============================================================================
# Testing
# =============================================================================
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_safety_node test/test_safety_node.cpp)
  target_link_libraries(test_safety_node safety_node)
endif()

ament_package()
