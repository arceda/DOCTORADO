#include "stdio.h"

int main(){
	#ifdef AWESOME
		printf("WE ARE AWESOME");
	#else
		printf("we ARE NOT AWESOME");
	#endif

	return 0;

}
