
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <stdio.h>
#include <stdlib.h>

static void HandleError(cudaError_t err, const char *file, int line) { if (err != cudaSuccess) { 
	printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);        
	exit(EXIT_FAILURE); } 
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
#define imin(a,b) (a<b?a:b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot_kernel(float* a, float* b, float* c)
{
	__shared__ float cache[threadsPerBlock];
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int cacheIndex = threadIdx.x;

	float temp = 0;
	while (tid < N)
	{
		temp += a[tid] * b[tid];
		//tid += threadsPerBlock*blocksPerGrid;
		tid += blockDim.x * gridDim.x;
	}

	cache[cacheIndex] = temp;
	__syncthreads();

	int i = blockDim.x / 2;
	while (i > 0)
	{
		if (cacheIndex < i)
			cache[cacheIndex] = cache[cacheIndex] + cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}

	// 经过上面的同步之后，chache[0]就是我们想要的改block的和
	// 把这个值存在global memory里，只需要一个线程来做这件事，这里选0线程
	if (cacheIndex == 0)
	{
		c[blockIdx.x] = cache[0];
	}


}

int main()
{
	float *h_a, *h_b, c, *h_partial_c;
	float *d_a, *d_b, *d_partial_c;

	h_a = (float*)malloc(N * sizeof(float));
	h_b = (float*)malloc(N * sizeof(float));
	h_partial_c = (float*)malloc(blocksPerGrid * sizeof(float));

	HANDLE_ERROR(cudaMalloc((void**)&d_a, N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&d_b, N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&d_partial_c, blocksPerGrid * sizeof(float)));

	// fill in data
	for (int i = 0; i < N; i++)
	{
		h_a[i] = i;
		h_b[i] = i * 2;
	}

	HANDLE_ERROR(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

	// launch kernel
	dot_kernel << <threadsPerBlock, blocksPerGrid >> > (d_a, d_b, d_partial_c);

	HANDLE_ERROR(cudaMemcpy(h_partial_c, d_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));

	c = 0;
	for (int i = 0; i < blocksPerGrid; i++)
	{
		c += h_partial_c[i];
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_partial_c);
	free(h_a);
	free(h_b);
	free(h_partial_c);

	printf("Square Sum:%f\n", c);
}