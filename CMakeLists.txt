# ---------------------
#    BeBot project
# ---------------------

cmake_minimum_required(VERSION 3.10.0)

project(
    bebot
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "BeBOT Library."
)
set (CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Main bebot target
add_subdirectory(src)

# Demo applications target
add_executable(main apps/main.cpp)
target_include_directories(main PRIVATE 
    src
    include/bebot
)
target_link_libraries(main PRIVATE bebot)

# Tests
add_subdirectory(tests)