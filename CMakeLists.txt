cmake_minimum_required(VERSION 3.20)

project(tinyML)

add_executable(tinyML
		main.cpp
		mnist_reader_common.h
		mnist_reader_less.h
		common.hpp
		dataset.hpp
		dataset.cpp
		layer.hpp
		activation.hpp
		operations.cpp
        )
target_include_directories(tinyML PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(tinyML
        PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
		VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        )