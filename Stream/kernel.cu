
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define N (1024*1024)
#define FULL_DATA_SIZE (N*20)  // �ֳ�20��chunk?

__global__ void kernel(int *a, int *b, int *c)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < N)
	{
		int idx1 = (idx + 1) % 256;
		int idx2 = (idx + 2) % 256;
		float as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
		float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
		c[idx] = (as + bs) / 2;
	}
}

int main()
{
	cudaDeviceProp prop;
	int whichDevice;
	cudaGetDevice(&whichDevice);
	cudaGetDeviceProperties(&prop, whichDevice);
	if (!prop.deviceOverlap) {
		printf("Device will not not handle overlaps, so no speed up from streams\n");
		return 0;
	}

	cudaEvent_t start, stop;
	float elapsedTime;
	// start the timers
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// create a stream
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// prepare data
	int *host_a, *host_b, *host_c;
	int *dev_a, *dev_b, *dev_c;

	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));

	cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);

	for (int i = 0; i < FULL_DATA_SIZE; i++)
	{
		host_a[i] = rand();
		host_b[i] = rand();
	}

	// loop over full data, in bite-sized chunks
	for (int i = 0; i < FULL_DATA_SIZE; i += N)
	{
		cudaMemcpyAsync(dev_a, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(dev_b, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);
		// ��Щ�����첽�ģ�a���ص�ʱ��δ��ִ���꣬����һ�ζ�������д���ʱ��aһ��ִ���ꡣ
		// Ҳ���ǣ����￪ʼд��b��ʱ��a�ѿ����ꡣ
		// kernelִ��ǰ��b�Ѿ������ꡣ
		kernel << <N / 256, 256, 0, stream >> > (dev_a, dev_b, dev_c);

		cudaMemcpyAsync(host_c + i, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost, stream);
	}
	// ��forѭ��ִ���꣬GPU���ܻ����������δ���
	cudaStreamSynchronize(stream);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime,
		start, stop);
	printf("Time taken: %3.1f ms\n", elapsedTime);
	// cleanup the streams and memory
	cudaFreeHost(host_a);
	cudaFreeHost(host_b);
	cudaFreeHost(host_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	cudaStreamDestroy(stream);
	return 0;
}