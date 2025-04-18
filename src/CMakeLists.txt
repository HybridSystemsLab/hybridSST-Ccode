find_package(ompl REQUIRED)

add_library(HySST HySST.cpp)
add_library(HybridStateSpace HybridStateSpace.cpp)
add_library(HybridTimeStateSpace HybridTimeStateSpace.cpp)
target_link_libraries(HySST HybridStateSpace HybridTimeStateSpace ${OMPL_LIBRARIES})

set(CMAKE_TOOLCHAIN_FILE ${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)

find_package(Boost REQUIRED
             COMPONENTS serialization filesystem system program_options)
include_directories(${Boost_INCLUDE_DIRS})
link_libraries(${Boost_LIBRARIES})


# add_library(RapidQuadcopterTrajectories RapidTrajectoryGenerator.cpp SingleAxisTrajectory.cpp)
add_library(Quartic quartic.cpp)

include_directories(${HySST_SOURCE_DIR}) # Include the directory where HySST.h is located

# Include necessary directories
include_directories(${OMPL_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(CMAKE_BUILD_TYPE Debug)