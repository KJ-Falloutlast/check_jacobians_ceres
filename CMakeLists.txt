cmake_minimum_required(VERSION 3.5)
project(gravity_align)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Debug")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  rosbag
  std_msgs
  tf
  tf2 
  tf2_ros 
  message_generation
)
find_package(Ceres REQUIRED)


#找到tbb, glog
find_package(TBB REQUIRED)
find_package(Glog REQUIRED)
include_directories(${Glog_INCLUDE_DIRS})

# 查找YAML-CPP库
find_package(yaml-cpp REQUIRED)
find_package(Eigen3)
find_package(Sophus)
find_package( PCL 1.8 REQUIRED)#解决版本问题

# include directories
include_directories(
	include
	${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
	${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  # ${SUITESPARSE_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
)
set(Library_Set
    ${OpenCV_LIBRARIES}
    ${PCL_LIBRARIES}
    ${catkin_LIBRARIES}
    ${Boost_LIBRARIES}
    ${Reflexxes_LIBRARIES}
    ${CERES_LIBRARIES}
    fmt
)

add_executable(test_ceres_line src/test_ceres_line.cpp)
target_link_libraries(test_ceres_line  
              ${Library_Set}
                  )


add_executable(test_ceres_ba src/test_ceres_ba.cpp)
target_link_libraries(test_ceres_ba  
                  ${Library_Set}
                  )

add_executable(test_ceres_error src/test_ceres_error.cpp)
target_link_libraries(test_ceres_error  
                  ${Library_Set}
                  )

# add_executable(test_analytic_factor src/test_analytic_factor.cpp
#                                     src/poseParameterization.cpp)
# target_link_libraries(test_analytic_factor  
#                   ${Library_Set}
#                   )

add_executable(test_analytic_factor_clic src/test_analytic_factor_clic.cpp
                  src/poseParameterization.cpp)
target_link_libraries(test_analytic_factor_clic  
${Library_Set}
)