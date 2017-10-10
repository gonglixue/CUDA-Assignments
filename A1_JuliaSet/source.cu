
#if 0
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void HelloWorld()
{
	printf("Hello World\n");
}

int main()
{
	HelloWorld << <1, 10 >> > ();
	return 0;
}

#endif