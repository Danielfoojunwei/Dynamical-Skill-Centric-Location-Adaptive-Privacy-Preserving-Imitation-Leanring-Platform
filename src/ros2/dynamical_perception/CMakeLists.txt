cmake_minimum_required(VERSION 3.8)
project(dynamical_perception)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(dynamical_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)

# Include directories
include_directories(include)

# Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python nodes
install(PROGRAMS
  scripts/perception_pipeline_node.py
  scripts/cascade_manager_node.py
  scripts/detection_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files and config
install(DIRECTORY launch config models
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
