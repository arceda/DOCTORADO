cmake_minimum_required(VERSION 3.0)
project(multiple_files)

add_executable(
	app
	"${CMAKE_CURRENT_SOURCE_DIR}/main.c"
)

target_include_directories(
	app
	PRIVATE 
		"${CMAKE_CURRENT_SOURCE_DIR}/include"
)




