# ---------------------
#    Main bebot target
# ---------------------

add_library(bebot STATIC
    bebot.cpp
    common.cpp
    polynomials/algorithms/algorithm.cpp
    polynomials/primitives/bernstein.cpp
    polynomials/primitives/rational_bernstein.cpp
    polynomials/primitives/bernstein_helpers.cpp
    motion_planning/mpc.cpp
)
set_target_properties(bebot PROPERTIES CXX_STANDARD 17)

target_include_directories(bebot PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

find_package (Eigen3 3.4 REQUIRED NO_MODULE)
target_link_libraries(bebot Eigen3::Eigen)