#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "../common/cpu_bitmap.h"
#include <time.h>

#define DIM 1000


struct cuComplex {
	float r;
	float i;
	cuComplex(float a, float b):r(a), i(b){}
	__host__ __device__ float magnitude2(void) { return r*r + i*i; }
	__host__ __device__ cuComplex operator* (const cuComplex& a) {
		return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
	}
	__host__ __device__ cuComplex operator+(const cuComplex& a) {
		return cuComplex(r + a.r, i + a.i);
	}
};
__host__ __device__ int julia(int x, int y, float* mag)
{
	const float Mag2Limit = 1.9f;

	const float scale = 1.5;
	float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
	float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

	cuComplex c(-0.8, 0.156);
	cuComplex a(jx, jy);

	int i = 0;
	for (i = 0; i < 200; i++) {
		a = a*a + c;
		*mag = a.magnitude2();
		if (*mag > Mag2Limit)
			return 0;
	}

	return 1;
}

void CPU_kernel(unsigned char* ptr)
{
	const float Mag2Limit = 1.9f;
	
	for (int y = 0; y < DIM; y++)
	{
		for (int x = 0; x < DIM; x++)
		{
			int offset = x + y*DIM;

			float mag;
			int juliaValue = julia(x, y, &mag);
			ptr[offset * 4 + 0] = 255 * juliaValue * (Mag2Limit/(mag+0.8));
			ptr[offset * 4 + 1] = 255 * (juliaValue * mag/Mag2Limit);
			ptr[offset * 4 + 2] = 255 * (juliaValue * (mag+0.3)/Mag2Limit);
			ptr[offset * 4 + 3] = 255;
		}
	}
}

__global__ void GPU_kernel(unsigned char *ptr)
{
	const float Mag2Limit = 1.9f;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = x + y * DIM;

	if (x >= DIM || y >= DIM)
		return;

	float mag;
	int juliaValue = julia(x, y, &mag);
	ptr[offset * 4 + 0] = 255 * juliaValue * (Mag2Limit / (mag + 0.8));
	ptr[offset * 4 + 1] = 255 * (juliaValue * mag / Mag2Limit);
	ptr[offset * 4 + 2] = 255 * (juliaValue * (mag + 0.3) / Mag2Limit);
	ptr[offset * 4 + 3] = 255;

}

int main(void)
{
	clock_t start, finish;

#if 1
#pragma region CPU
	CPUBitmap bitmap(DIM, DIM);
	unsigned char *ptr = bitmap.get_ptr();

	start = clock();
	DWORD start_d = GetTickCount();
	for (int i = 0; i < 100; i++)
	{
		CPU_kernel(ptr);
	}
	
	finish = clock();
	DWORD finish_d = GetTickCount();
	printf("CPU time: %f ms\n", 10.0 * (finish - start) / CLOCKS_PER_SEC);
	//printf("CPU time2: %f ms\n", (float)(finish_d - start_d));
	bitmap.display_and_exit();
#pragma endregion CPU

#else
#pragma region GPU
	float total_time = 0;

		CPUBitmap bitmap2(DIM, DIM);
		unsigned char *d_pixels;

		cudaMalloc((void**)&d_pixels, bitmap2.image_size());  // char size

		dim3 block_size(32, 32);
		dim3 grid_size((1024 + block_size.x - 1) / block_size.x, (1024 + block_size.x - 1) / block_size.x);

		for (int i = 0; i < 100; i++)
		{
			start = clock();
			GPU_kernel << <grid_size, block_size >> > (d_pixels);
			cudaDeviceSynchronize();
			finish = clock();
			cudaMemcpy(bitmap2.get_ptr(), d_pixels, bitmap2.image_size(), cudaMemcpyDeviceToHost);
			total_time += 1000.0*(finish - start) / CLOCKS_PER_SEC;
			printf("GPU time(including memcpy cost): %f ms\n", 1000.0*(finish - start) / CLOCKS_PER_SEC);
		}
		printf("Average GPU time (including memcpy cost): %f\n", total_time / 100);
		bitmap2.display_and_exit();

		cudaFree(d_pixels);
	

	
	
#pragma endregion

#endif
}