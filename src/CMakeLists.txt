add_library(bebot bebot.cpp ../include/bebot/bebot.hpp)
target_include_directories(bebot PUBLIC ../include)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries(bebot Eigen3::Eigen)