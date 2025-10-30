cmake_minimum_required(VERSION 3.8)
project(safe_ac_exp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ros2_unitree_legged_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

set(EXTRA_LIBS -pthread libunitree_legged_sdk_${ARCH}.so lcm)
set(CMAKE_CXX_FLAGS "-O3")

# Link to boost
set(BOOST_ROOT "${CMAKE_SOURCE_DIR}/../unitree_ros2_to_real/unitree_legged_sdk/boost_1_71_0")
set(Boost_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/../unitree_ros2_to_real/unitree_legged_sdk/boost_1_71_0")
include_directories(${Boost_INCLUDE_DIRS}) 

# Include directories
include_directories(
    include
    include/${PROJECT_NAME}
    ${CMAKE_SOURCE_DIR}/../unitree_ros2_to_real/unitree_legged_sdk-master/include
    ${sensor_msgs_INCLUDE_DIRS}
)

# Link to unitree legged sdk library
link_directories(${CMAKE_SOURCE_DIR}/../unitree_ros2_to_real/unitree_legged_sdk-master/lib)

set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/unitree_legged_sdk-master/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_executable(communication_node src/safe_ac_exp/communication.cpp)
ament_target_dependencies(communication_node rclcpp sensor_msgs ros2_unitree_legged_msgs)

# Install targets
install(TARGETS
  communication_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# Install Python modules
ament_python_install_package(${PROJECT_NAME}
  PACKAGE_DIR src/${PROJECT_NAME}
)

# Install Python scripts
install(PROGRAMS
  src/safe_ac_exp/robot_control_ac.py
  src/safe_ac_exp/omni_control.py
  DESTINATION lib/${PROJECT_NAME}
)
ament_package()
