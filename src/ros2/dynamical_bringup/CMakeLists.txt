cmake_minimum_required(VERSION 3.8)
project(dynamical_bringup)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Python scripts
ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/api_bridge_node.py
  scripts/telemetry_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch and config
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
