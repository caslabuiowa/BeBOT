# --------------
#    Tests
# --------------

# Define executable targets
add_executable(bernstein_tests
    bernstein_test.cpp
)
set_target_properties(bernstein_tests PROPERTIES CXX_STANDARD 17)


# For each test target, include library includes
target_include_directories(bernstein_tests PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
    ${gtest_SOURCE_DIR}/include
)

find_package (Eigen3 3.4 REQUIRED NO_MODULE)

# Gtest testing framework
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

target_link_libraries(bernstein_tests PUBLIC
    bebot
    Eigen3::Eigen 
    ${GTEST_LIBRARIES} 
    pthread
)

gtest_discover_tests(
    bernstein_tests
)