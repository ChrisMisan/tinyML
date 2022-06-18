cmake_minimum_required(VERSION 3.8)

project(tinyML_tests VERSION 1.0.0 LANGUAGES CXX)

find_c_and_cpp_files("${CMAKE_CURRENT_SOURCE_DIR}/include" tinyML_tests_headers)
find_c_and_cpp_files("${CMAKE_CURRENT_SOURCE_DIR}/src" tinyML_tests_sources)

add_executable(tinyML_tests
		src/main.cpp
		#${tinyML_tests_headers}
		#${tinyML_tests_sources}
)
target_link_libraries(tinyML_tests PRIVATE tinyML)
target_include_directories(tinyML_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_target_properties(tinyML_tests
        PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
