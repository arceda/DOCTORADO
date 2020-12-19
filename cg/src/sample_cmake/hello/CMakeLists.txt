cmake_minimum_required(VERSION 3.0)

#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}   -std=c++14" )

project(hello)


#add_executable(
#	sample
#	main.cpp
#)

add_executable(
	hello
	"${CMAKE_CURRENT_SOURCE_DIR}/main.c"
)





