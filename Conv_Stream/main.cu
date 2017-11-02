#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <opencv2\opencv.hpp>
#include <opencv2/core/core.hpp>
#include <stdio.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <string>

#define GAUSSIAN_KERNEL_SIZE 9
#define GAUSSIAN_KERNEL_RADIUS 4
#define CHUNCKS 8

static void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

// chunck_height包括overlap的部分
__global__ void GaussianWithSharedMem(unsigned char* in_data, unsigned char* out_data, int chunck_width, int chunck_height)
{
	const int block_size = 32;
	__shared__ uchar block_cache[block_size * 2][block_size * 2];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	// if idx > width
	int x, y;
	x = idx - block_size / 2 + threadIdx.x;
	y = idy - block_size / 2 + threadIdx.y;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			int temp_x = x + i;
			int temp_y = y + j + GAUSSIAN_KERNEL_RADIUS;
			int value;
			if (temp_x < 0 || temp_x>(chunck_width - 1))
				value = 0;
			else if (temp_y < 0 || temp_y >(chunck_height - 1))
				value = 0;
			else
				value = in_data[(temp_y) * chunck_width + temp_x];

			block_cache[2 * threadIdx.y + j][2 * threadIdx.x + i] = value;
		}
	}
	__syncthreads();

	if (idx < chunck_width && (idy+GAUSSIAN_KERNEL_RADIUS) < (chunck_height-GAUSSIAN_KERNEL_RADIUS))  // 不含overlap部分
	{
		// convolution
		float kernelx[9][9] = {


			0.0084,    0.0096,    0.0107,    0.0113,    0.0115,    0.0113,0.0107,0.0096,0.0084,
			0.0096,    0.0111,    0.0123,    0.0130 ,   0.0133 ,   0.0130,0.0123,0.0111,0.0096,
			0.0107,    0.0123,    0.0136 ,   0.0144  ,  0.0147  ,  0.0144,0.0136,0.0123,0.0107,
			0.0113,    0.0130,    0.0144  ,  0.0153   , 0.0156   , 0.0153,0.0144,0.0130,0.0113,
			0.0115,    0.0133,    0.0147   , 0.0156    ,0.0159    ,0.0156,0.0147,0.0133,0.0115,
			0.0113,    0.0130,    0.0144    ,0.0153    ,0.0156    ,0.0153,0.0144,0.0130,0.0113,
			0.0107,    0.0123,    0.0136    ,0.0144    ,0.0147    ,0.0144,0.0136,0.0123,0.0107,
			0.0096,    0.0111,    0.0123    ,0.0130    ,0.0133    ,0.0130,0.0123,0.0111,0.0096,
			0.0084,    0.0096,    0.0107    ,0.0113    ,0.0115    ,0.0113,0.0107,0.0096,0.0084
		};
		int kernel_radius = 4;

		float result = 0;
		int centerx_in_cache = threadIdx.y + block_size / 2;
		int centery_in_cache = threadIdx.x + block_size / 2;

		for (int i = -1 * GAUSSIAN_KERNEL_RADIUS; i <= GAUSSIAN_KERNEL_RADIUS; i++)  // row
		{
			for (int j = -1 * GAUSSIAN_KERNEL_RADIUS; j <= GAUSSIAN_KERNEL_RADIUS; j++)
			{
				int cachex = centerx_in_cache + i;
				int cachey = centery_in_cache + j;
				result += kernelx[i + GAUSSIAN_KERNEL_RADIUS][j + GAUSSIAN_KERNEL_RADIUS] * block_cache[cachex][cachey];
			}
		}

		result = abs(result);
		int pixel_id = idy * chunck_width + idx;
		out_data[pixel_id] = (int)result;
		//out_data[pixel_id] = 0;
	}
}

__global__ void GaussianWithGlobalMem(unsigned char* in_data, unsigned char* out_data, int chunck_width, int chunck_height)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int pixel_id = idy * chunck_width + idx;

	if (idx >= chunck_width || idy >= (chunck_height - 2 * GAUSSIAN_KERNEL_RADIUS))
		return;

	float kernelx[9][9] = {


		0.0084,    0.0096,    0.0107,    0.0113,    0.0115,    0.0113,0.0107,0.0096,0.0084,
		0.0096,    0.0111,    0.0123,    0.0130 ,   0.0133 ,   0.0130,0.0123,0.0111,0.0096,
		0.0107,    0.0123,    0.0136 ,   0.0144  ,  0.0147  ,  0.0144,0.0136,0.0123,0.0107,
		0.0113,    0.0130,    0.0144  ,  0.0153   , 0.0156   , 0.0153,0.0144,0.0130,0.0113,
		0.0115,    0.0133,    0.0147   , 0.0156    ,0.0159    ,0.0156,0.0147,0.0133,0.0115,
		0.0113,    0.0130,    0.0144    ,0.0153    ,0.0156    ,0.0153,0.0144,0.0130,0.0113,
		0.0107,    0.0123,    0.0136    ,0.0144    ,0.0147    ,0.0144,0.0136,0.0123,0.0107,
		0.0096,    0.0111,    0.0123    ,0.0130    ,0.0133    ,0.0130,0.0123,0.0111,0.0096,
		0.0084,    0.0096,    0.0107    ,0.0113    ,0.0115    ,0.0113,0.0107,0.0096,0.0084
	};

	float result;
	for (int i = -1 * GAUSSIAN_KERNEL_RADIUS; i <= GAUSSIAN_KERNEL_RADIUS; i++)
	{
		for (int j = -1 * GAUSSIAN_KERNEL_RADIUS; j <= GAUSSIAN_KERNEL_RADIUS; j++)
		{
			int tempx = idx + j;
			int tempy = idy + i + GAUSSIAN_KERNEL_RADIUS;
			if (tempx >= 0 && tempx < chunck_width && tempy >= 0 && tempy < chunck_height)
				result += kernelx[i + GAUSSIAN_KERNEL_RADIUS][j + GAUSSIAN_KERNEL_RADIUS]
				* in_data[tempy * chunck_width + tempx];
		}
	}

	result = abs(result);
	out_data[pixel_id] = (int)result;

}
int main(int argc, char** argv)
{
	// load image;
	cv::Mat origin_image = cv::imread(argv[1], 0);
	cv::imshow("origin", origin_image);
	int width, height;
	width = origin_image.cols;
	height = origin_image.rows;


	int chunck_height = height / CHUNCKS;
	int chunck_data_size = width*(chunck_height + GAUSSIAN_KERNEL_RADIUS * 2);
	int chunck_valid_size = width * chunck_height;

	// start the timers
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// host data: origin_image.data
	uchar* h_image_data,*h_image_out;
	uchar* d_image_data0, *d_image_data1;
	uchar* d_image_out0, *d_image_out1;

	cudaHostAlloc((void**)&h_image_data, width*(height+2*GAUSSIAN_KERNEL_RADIUS), cudaHostAllocDefault);  // paged-locked
	cudaHostAlloc((void**)&h_image_out, width*height, cudaHostAllocDefault);


	cudaMalloc((void**)&d_image_data0, chunck_data_size);  // 输入包含上下overlap部分
	cudaMalloc((void**)&d_image_data1, chunck_data_size);
	cudaMalloc((void**)&d_image_out0, chunck_valid_size);
	cudaMalloc((void**)&d_image_out1, chunck_valid_size);

	memset(h_image_data, 255, width*(height + 2 * GAUSSIAN_KERNEL_RADIUS));
	memcpy(h_image_data+width*GAUSSIAN_KERNEL_RADIUS, origin_image.data, width*height);

	// initialize the treams
	cudaStream_t stream0, stream1;
	float elapsedTime;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);

	dim3 block_size = dim3(32, 32, 1);
	dim3 grid_size = dim3((width + block_size.x - 1) / block_size.x, 
		(chunck_height+block_size.y - 1) / block_size.y, 
		1);

	// 实际线程只覆盖有效像素区域
	for(int n=0;n<100;n++)
	for (int i = 0; i < CHUNCKS/2; i++)
	{
		int data_offset0 = width*(chunck_height * 2 * i) + width*GAUSSIAN_KERNEL_RADIUS;
		int data_offset1 = width*(chunck_height * (2 * i + 1)) + width*GAUSSIAN_KERNEL_RADIUS;
		
		cudaMemcpyAsync(d_image_data0, h_image_data + data_offset0 - width*GAUSSIAN_KERNEL_RADIUS, chunck_data_size,
			cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(d_image_data1, h_image_data + data_offset1 - width*GAUSSIAN_KERNEL_RADIUS, chunck_data_size,
			cudaMemcpyHostToDevice, stream1);

		GaussianWithSharedMem << <grid_size, block_size, 0, stream0 >> > (d_image_data0, d_image_out0, width, chunck_height+2*GAUSSIAN_KERNEL_RADIUS);
		GaussianWithSharedMem << <grid_size, block_size, 0, stream1 >> > (d_image_data1, d_image_out1, width, chunck_height + 2 * GAUSSIAN_KERNEL_RADIUS);
		//GaussianWithGlobalMem << <grid_size, block_size, 0, stream0 >> > (d_image_data0, d_image_out0, width, chunck_height + 2 * GAUSSIAN_KERNEL_RADIUS);
		//GaussianWithGlobalMem << <grid_size, block_size, 0, stream1 >> > (d_image_data1, d_image_out1, width, chunck_height + 2 * GAUSSIAN_KERNEL_RADIUS);

		cudaMemcpyAsync(h_image_out + data_offset0 - width*GAUSSIAN_KERNEL_RADIUS, d_image_out0, chunck_valid_size,
			cudaMemcpyDeviceToHost, stream0);
		cudaMemcpyAsync(h_image_out + data_offset1 - width*GAUSSIAN_KERNEL_RADIUS, d_image_out1, chunck_valid_size,
			cudaMemcpyDeviceToHost, stream1);

	}

	cudaStreamSynchronize(stream0);
	cudaStreamSynchronize(stream1);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("convolution with stream: %3.1f ms\n", elapsedTime);

	cv::Mat result_image(height, width, CV_8UC1);
	result_image.data = h_image_out;
	cv::imshow("result", result_image);

	//uchar* d_image_out_full, *d_image_in_full;
	//cudaMalloc((void**)&d_image_out_full, width*height);
	//cudaMalloc((void**)&d_image_in_full, width*(height + 2 * GAUSSIAN_KERNEL_RADIUS));
	//cudaMemcpy(d_image_in_full, h_image_data, width*(height + 2 * GAUSSIAN_KERNEL_RADIUS), cudaMemcpyHostToDevice);

	//grid_size = dim3((width + block_size.x - 1) / block_size.x,
	//	(height + block_size.y - 1) / block_size.y,
	//	1);
	//GaussianWithGlobalMem << <grid_size, block_size >> > (d_image_in_full, d_image_out_full, width, height + 2 * GAUSSIAN_KERNEL_RADIUS);
	//cv::Mat result_full_image(height, width, CV_8UC1);
	//cudaMemcpy(result_full_image.data, d_image_out_full, width*(height + 2 * GAUSSIAN_KERNEL_RADIUS), cudaMemcpyDeviceToHost);
	//cv::imshow("result full", result_full_image);

	cv::waitKey();
	cudaFreeHost(h_image_data);
	cudaFreeHost(h_image_out);
	cudaFree(d_image_data1);
	cudaFree(d_image_data0);
	cudaFree(d_image_out1);
	cudaFree(d_image_out0);
	cudaStreamDestroy(stream0);
	cudaStreamDestroy(stream1);

	return 0;
}