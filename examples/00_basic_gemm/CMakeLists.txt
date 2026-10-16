cmake_minimum_required(VERSION 3.22)
project(topscctest)



set(CMAKE_BUILD_TYPE Debug)
file(GLOB_RECURSE _header_files "${CMAKE_CURRENT_SOURCE_DIR}/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
file(GLOB_RECURSE _src_files "${CMAKE_CURRENT_SOURCE_DIR}/*.cc" "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

set(TOPS_INTALL_DIR /opt/tops)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gcc-toolchain /usr/lib/gcc/x86_64-linux-gnu/11")
option(STATIC_LINK "Link topscc statically" OFF)
add_compile_options(-Werror -Wall)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Common)
if(STATIC_LINK)
	add_compile_options(-static)
endif()
unset(CAPS_LIBRARIES)

find_package(topscc REQUIRED PATHS ${TOPS_INTALL_DIR})

if(topscc_EXECUTABLE)
	message(STATUS "topscc executable: ${topscc_EXECUTABLE}")
endif()
set(CMAKE_C_COMPILER ${topscc_EXECUTABLE})
set(CMAKE_CXX_COMPILER ${topscc_EXECUTABLE})

find_package(topstx REQUIRED PATHS ${TOPS_INTALL_DIR})

if(STATIC_LINK)
	message(STATUS "no static libraries with topstx")
else()
	list(APPEND CAPS_LIBRARIES topstx)
endif()

find_package(topsruntime REQUIRED PATHS ${TOPS_INTALL_DIR})

if(STATIC_LINK)
	list(APPEND CAPS_LIBRARIES libtopsrt_static libtops_comgr_static librtcu_static)
else()
	list(APPEND CAPS_LIBRARIES libtopsrt libefrt libtops_comgr)
endif()

find_package(topscodec REQUIRED PATHS ${TOPS_INTALL_DIR})

if(STATIC_LINK)
	list(APPEND CAPS_LIBRARIES libtopscodec_static)
else()
	list(APPEND CAPS_LIBRARIES libtopscodec)
endif()

find_package(topspti REQUIRED PATHS ${TOPS_INTALL_DIR})

if(STATIC_LINK)
	message(STATUS "no static libraries with topspti")
else()
	list(APPEND CAPS_LIBRARIES topspti_rt topspti_prof)
endif()

unset(_target_files)
foreach(_src_file ${_src_files})
	get_filename_component(_src_file_name ${_src_file} NAME)
	string(REPLACE ".cc" "" _target_name ${_src_file_name})
	string(REPLACE ".cpp" "" _target_name ${_target_name})
	add_executable(${_target_name} ${_src_file})
	list(APPEND _target_files ${_target_name})
	target_link_libraries(${_target_name} ${CAPS_LIBRARIES})
endforeach()
