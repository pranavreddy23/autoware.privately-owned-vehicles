# Optional heuristic Occupancy BEV module — keep this tree self-contained so it
# can be copied into autowarefoundation/vision_pilot with a small glue patch.
cmake_minimum_required(VERSION 3.22.1)

find_package(OpenCV REQUIRED)

add_library(vp_occupancy
        src/occupancy_view.cpp
        src/occupancy_bridge.cpp
)

target_include_directories(vp_occupancy
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(vp_occupancy
        PUBLIC
        ${OpenCV_LIBRARIES}
        models
        common
)
