cmake_minimum_required(VERSION 3.0)
project(librerias)

# se creara una libreia awesome
add_library(
	awesome
	"${CMAKE_CURRENT_SOURCE_DIR}/library/src/awesome.c"
)

add_executable(
	my_app
	"${CMAKE_CURRENT_SOURCE_DIR}/application/main.c"
)

target_include_directories(
	my_app
	PRIVATE 
		"${CMAKE_CURRENT_SOURCE_DIR}/library/include"
)

# agregamso la dependencia, primero se construyte la libreia y luego la applicaion
target_link_libraries(
	my_app
	PRIVATE
		awesome
)




