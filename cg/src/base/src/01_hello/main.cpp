

#include <glad/glad.h>
#include <glfw/glfw3.h>

#include <iostream>

using namespace std;

// para que Opengl sepa cada vez que hacemos resize a la ventana
void framebuffer_size_callback(GLFWwindow* window, int width, int height){
    glViewport(0, 0, width, height);
}  

// to check user event in render loop
void processInput(GLFWwindow *window){
	cout<<"Entramos a un handler press key: "<<GLFW_PRESS<<endl;
    	if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS){ // si presiona esc
		cout<<"cerramos la ventana\n";
        	glfwSetWindowShouldClose(window, true);
	}
}

int main()
{
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	//glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);  // for mac os

	// create a window object **************************************************
	GLFWwindow* window = glfwCreateWindow(800, 600, "LearnOpenGL", NULL, NULL);
	if (window == NULL){	
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
	    	return -1;
	}
	glfwMakeContextCurrent(window);
	//**************************************************************************

	// initialize glad *********************************************************
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
    		std::cout << "Failed to initialize GLAD" << std::endl;
    		return -1;
	}
	//**************************************************************************

	// tell OpenGL the size of rendering window
	glViewport(0, 0, 800, 600);

	// para que openGL se actualialize cuando cambviamos de tamanio la ventana
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);	
	
	//RENDER LOOP  la app se mantiene graficando y handling usEr events hasta que terminemos
	while(!glfwWindowShouldClose(window)){ // si no se cerro la ventana
		processInput(window);

		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	        glClear(GL_COLOR_BUFFER_BIT);

		glfwSwapBuffers(window); // use in render
	    	glfwPollEvents();   //check if any event is triggered
	}
	
	// al cerrar la ventana, limpiamos todo
	glfwTerminate();

	return 0;
}


