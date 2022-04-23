add_library(bebot bebot.cpp ../include/bebot/bebot.hpp)
target_include_directories(bebot PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_subdirectory(motion_planning)
add_subdirectory(polynomials)

find_package (Eigen3 3.4 REQUIRED NO_MODULE)
target_link_libraries(bebot Eigen3::Eigen)