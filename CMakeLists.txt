cmake_minimum_required(VERSION 3.1.0)

project(RealsenseExamplesPointcloud)

find_package(realsense2 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)


# if(BUILD_GRAPHICAL_EXAMPLES)
  add_executable(rs-pointcloud visualizer.cpp)
  # set_property(TARGET rs-pointcloud PROPERTY CXX_STANDARD 11)
  target_link_libraries(rs-pointcloud ${realsense2_LIBRARY} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})
  # include_directories(../)
  # set_target_properties (rs-pointcloud PROPERTIES FOLDER Examples)
  # install(TARGETS rs-pointcloud RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
# endif()