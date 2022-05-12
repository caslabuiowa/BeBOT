add_library(bebot
    bebot.cpp 
    polynomials/algorithms/algorithm.cpp
    polynomials/primitives/bernstein.cpp
    polynomials/primitives/rational_bernstein.cpp
    motion_planning/mpc.cpp
)
target_include_directories(bebot PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(bebot PROPERTIES CXX_STANDARD 17)
find_package (Eigen3 3.4 REQUIRED NO_MODULE)
target_link_libraries(bebot Eigen3::Eigen)