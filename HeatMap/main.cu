#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "../common/util.h"

#define DIM 1024
//__constant__ float SPEED = 0.25f;
//__constant__ float PI = 3.141592653f;
//__constant__ float MAX_TEMP = 1.0F;
//__constant__ float MIN_TEMP = 0.0001F;
#define SPEED 0.25f
#define PI 3.1415926f
#define MAX_TEMP 1.0F
#define MIN_TEMP 0.0001F

struct DataBlock {
	unsigned char *output_bitmap;
	float *dev_inSrc;
	float *dev_outSrc;
	float *dev_constSrc;
	cv::Mat *bitmap;
	cudaEvent_t start, stop;
	float totalTime;
	float frames;
};

__global__ void copy_const_kernel(float *iptr, const float *cptr)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y*blockDim.x * gridDim.x;

	if (cptr[offset] != 0)
		iptr[offset] = cptr[offset];
}

// a place for common kernels - starts here

__device__ unsigned char value(float n1, float n2, int hue) {
	if (hue > 360)      hue -= 360;
	else if (hue < 0)   hue += 360;

	if (hue < 60)
		return (unsigned char)(255 * (n1 + (n2 - n1)*hue / 60));
	if (hue < 180)
		return (unsigned char)(255 * n2);
	if (hue < 240)
		return (unsigned char)(255 * (n1 + (n2 - n1)*(240 - hue) / 60));
	return (unsigned char)(255 * n1);
}

__global__ void float_to_color(unsigned char *optr,
	const float *outSrc) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float l = outSrc[offset];
	float s = 1;
	int h = (180 + (int)(360.0f * outSrc[offset])) % 360;
	float m1, m2;

	if (l <= 0.5f)
		m2 = l * (1 + s);
	else
		m2 = l + s - l * s;
	m1 = 2 * l - m2;

	optr[offset * 4 + 0] = value(m1, m2, h + 120);
	optr[offset * 4 + 1] = value(m1, m2, h);
	optr[offset * 4 + 2] = value(m1, m2, h - 120);
	optr[offset * 4 + 3] = 255;
}

__global__ void float_to_color(uchar4 *optr,
	const float *outSrc) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float l = outSrc[offset];
	float s = 1;
	int h = (180 + (int)(360.0f * outSrc[offset])) % 360;
	float m1, m2;

	if (l <= 0.5f)
		m2 = l * (1 + s);
	else
		m2 = l + s - l * s;
	m1 = 2 * l - m2;

	optr[offset].x = value(m1, m2, h + 120);
	optr[offset].y = value(m1, m2, h);
	optr[offset].z = value(m1, m2, h - 120);
	optr[offset].w = 255;
}

__global__ void blend_kernel(float *outSrc, const float *inSrc)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y*blockDim.x * gridDim.x;

	int left = offset - 1;
	int right = offset + 1;
	if (x == 0) left++;
	if (x == DIM - 1)  right--;

	int top = offset - DIM;
	int bottom = offset + DIM;
	if (y == 0) top += DIM;
	if (y == DIM - 1) bottom -= DIM;

	outSrc[offset] = inSrc[offset] + SPEED * (inSrc[top] +
		inSrc[bottom] + inSrc[left] + inSrc[right] -
		inSrc[offset] * 4);
}

void anim_gpu(DataBlock *d, int ticks)
{
	cudaEventRecord(d->start, 0);
	dim3 grid_size(DIM / 16, DIM / 16);
	dim3 block_size(16, 16);

	cv::Mat *bitmap = d->bitmap;

	for (int i = 0; i < 90; i++) {
		copy_const_kernel << <grid_size, block_size >> > (d->dev_inSrc, d->dev_constSrc);

		blend_kernel << <grid_size, block_size >> > (d->dev_outSrc, d->dev_inSrc);

		//swap(d->dev_inSrc, d->dev_outSrc);  // 为什么要交换，直接把out赋给in可以吗?貌似是可以的
		d->dev_inSrc = d->dev_outSrc;
	}
	
	float_to_color << <grid_size, block_size >> > (d->output_bitmap, d->dev_inSrc);


	cudaMemcpy(bitmap->data, d->output_bitmap, DIM * DIM * 4 * sizeof(uchar), cudaMemcpyDeviceToHost);

	cudaEventRecord(d->stop, 0);
	cudaEventSynchronize(d->stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, d->start, d->stop);

	d->totalTime += elapsedTime;
	++d->frames;

	printf("average time per frame: %3.1f ms\n", d->totalTime / d->frames);
}

void anim_exit(DataBlock *d)
{
	cudaFree(d->dev_inSrc);
	cudaFree(d->dev_constSrc);
	cudaFree(d->dev_outSrc);

	cudaEventDestroy(d->start);
	cudaEventDestroy(d->stop);
}

int main(void)
{
	DataBlock data;
	cv::Mat bitmap(DIM, DIM, CV_8UC4);
	data.bitmap = &bitmap;
	data.totalTime = 0;
	data.frames = 0;
	cudaEventCreate(&data.start);
	cudaEventCreate(&data.stop);

	int bitmap_size = DIM*DIM * 4;
	cudaMalloc((void**)&data.output_bitmap, bitmap_size);  // 4 channels, output_bitmap是在显存上，存储的是0~255颜色值
	cudaMalloc((void**)&data.dev_inSrc, bitmap_size );  // inSrc和constSrc相当于是1通道的float，所以不需要*sizeof(float)
	cudaMalloc((void**)&data.dev_outSrc, bitmap_size);  
	cudaMalloc((void**)&data.dev_constSrc, bitmap_size );

	float *temp = (float*)malloc(bitmap_size * sizeof(float));  //用于初始化constSrc
	for (int i = 0; i < DIM*DIM; i++)
	{
		temp[i] = 0;
		int x = i%DIM;
		int y = i / DIM;
		if ((x > 300)&(x < 600) & (y > 310) && (y < 601))
			temp[i] = MAX_TEMP;
	}
	temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2;
	temp[DIM * 700 + 100] = MIN_TEMP;
	temp[DIM * 300 + 300] = MIN_TEMP;
	temp[DIM * 200 + 700] = MIN_TEMP;
	for (int y = 800; y<900; y++) {
		for (int x = 400; x<500; x++) {
			temp[x + y*DIM] = MIN_TEMP;
		}
	}

	cudaMemcpy(data.dev_constSrc, temp, bitmap_size, cudaMemcpyHostToDevice);

	for (int y = 800; y < DIM; y++)
	{
		for (int x = 0; x < 200; x++) {
			temp[x + y*DIM] = MAX_TEMP;
		}
	}
	cudaMemcpy(data.dev_inSrc, temp, bitmap_size, cudaMemcpyHostToDevice);

	free(temp);

	while (true)
	{
		anim_gpu(&data, 0);
		cv::imshow("anim", *data.bitmap);
		int input = cv::waitKey(30);
		if (input == 27)
			break;
	}

	anim_exit(&data);
}