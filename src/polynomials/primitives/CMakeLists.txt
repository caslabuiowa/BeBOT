add_library(primitives 
bernstein.cpp
rational_bernstein.cpp)

target_include_directories(primitives PUBLIC ${CMAKE_SOURCE_DIR}/include)
