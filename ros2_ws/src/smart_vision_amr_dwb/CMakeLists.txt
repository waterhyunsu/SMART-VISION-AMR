cmake_minimum_required(VERSION 3.8)
project(smart_vision_amr_dwb)

find_package(ament_cmake REQUIRED)
find_package(dwb_core REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_util REQUIRED)
find_package(nav_2d_msgs REQUIRED)
find_package(nav_2d_utils REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)

add_library(route_tracking_dwb SHARED src/route_tracking_dwb.cpp)
target_compile_features(route_tracking_dwb PUBLIC cxx_std_17)
target_compile_options(route_tracking_dwb PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(route_tracking_dwb PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(route_tracking_dwb
  dwb_core nav2_core nav2_util nav_2d_msgs nav_2d_utils geometry_msgs std_msgs pluginlib rclcpp rclcpp_lifecycle)
pluginlib_export_plugin_description_file(nav2_core route_tracking_plugin.xml)

install(TARGETS route_tracking_dwb
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
ament_export_include_directories(include)
ament_export_libraries(route_tracking_dwb)
ament_export_dependencies(dwb_core nav2_core nav2_util nav_2d_msgs nav_2d_utils geometry_msgs std_msgs pluginlib rclcpp rclcpp_lifecycle)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_route_horizon test/test_route_horizon.cpp)
  target_include_directories(test_route_horizon PRIVATE include)
  ament_target_dependencies(test_route_horizon nav_2d_msgs)
endif()

ament_package()
