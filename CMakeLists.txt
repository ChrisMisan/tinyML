cmake_minimum_required(VERSION 3.8)

project(tinyML VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(common)
include(CPM)

CPMFindPackage(
		NAME nlohmann_json
		GITHUB_REPOSITORY nlohmann/json
		VERSION 3.10.5
)

find_c_and_cpp_files("${CMAKE_CURRENT_SOURCE_DIR}/include" tinyML_headers)
find_c_and_cpp_files("${CMAKE_CURRENT_SOURCE_DIR}/src" tinyML_sources)

add_library(tinyML STATIC
		${tinyML_headers}
		${tinyML_sources}
)
target_link_libraries(tinyML PRIVATE nlohmann_json::nlohmann_json)
target_include_directories(tinyML PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(tinyML PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

set_target_properties(tinyML
        PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

add_subdirectory(tests)