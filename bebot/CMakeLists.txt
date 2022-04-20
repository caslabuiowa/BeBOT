## [main]

# See: https://cliutils.gitlab.io/modern-cmake/chapters/basics/example.html

# Almost all CMake files should start with this
# You should always specify a range with the newest
# and oldest tested versions of CMake. This will ensure
# you pick up the best policies.
cmake_minimum_required(VERSION 3.0.0)

# This is your project statement. You should always list languages;
# Listing the version is nice here since it sets lots of useful variables
project(
    bebot
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "BeBOT Library.")

#add_subdirectory(src)
add_library(bebot src/bebot.cpp)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries(bebot Eigen3::Eigen)

add_executable(main apps/main.cpp)
target_include_directories(main PRIVATE src)
target_link_libraries(main PRIVATE bebot)
#add_subdirectory(apps) 

# # If you set any CMAKE_ variables, that can go here.
# # (But usually don't do this, except maybe for C++ standard)

# # Find packages go here.

# # You should usually split this into folders, but this is a simple example

# This is a "default" library, and will match the *** variable setting.
# Other common choices are STATIC, SHARED, and MODULE
# Including header files here helps IDEs but is not required.
# Output libname matches target name, with the usual extensions on your system
# add_library(bebot SHARED
#     src/bebot.cpp
#     include/bebot/bebot.hpp)

# # Link each target with other targets or add options, etc.

# #include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include") 

# add_executable(MyApp apps/main.cpp)

# target_include_directories(bebot PRIVATE include)
# target_include_directories(bebot PRIVATE src)

# # set(CPACK_PROJECT_NAME ${PROJECT_NAME})
# # set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
# # include(CPack)

# # # Adding something we can run - Output name matches target name
# # add_executable(MyExample simple_example.cpp)

# # # Make sure you link your targets with this command. It can also link libraries and
# # # even flags, so linking a target that does not exist will not give a configure-time error.
# # target_link_libraries(MyExample PRIVATE MyLibExample)