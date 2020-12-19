cmake_minimum_required(VERSION 3.0)

project(definitions)

add_executable(
	definitions
	"${CMAKE_CURRENT_SOURCE_DIR}/main.c"
)

# con esto podemos agregar definicions = enviar parametros al codigo
target_compile_definitions(definitions
	PRIVATE
		AWESOME
)


