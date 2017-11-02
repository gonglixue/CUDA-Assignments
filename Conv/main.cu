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

#define TEST_TIMES 100

__global__ void Simple_SobelX_Kernel(unsigned char *ptr, unsigned short* out, int width, int height, int depth)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	
	int pixel_id = idy * width + idx;
	if (idx >= width || idy >= height)
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
	int kernel_radius = 4;

	float resultx = 0;
	for (int i = -1*kernel_radius; i <= kernel_radius; i++)  // row
	{
		for (int j = -1*kernel_radius; j <= kernel_radius; j++)  // col
		{
			int tempx = idx + j;
			int tempy = idy + i;
			if(tempx>=0 && tempx<width && tempy>=0 && tempy<height)
				resultx += kernelx[i + kernel_radius][j + kernel_radius] * ptr[tempy*width + tempx];
		}
	}

	resultx = abs(resultx);
	int temp = resultx;
	//temp = temp > 65535 ? 65535 : temp;
	out[pixel_id ] = temp;
	//out[pixel_id * depth + channel] = resultx + resulty;
	//printf("x:%d, y:%d  z;%d origin:%d result:%d width:%d height:%d\n", idx, idy, channel, p4, temp, width, height);
	
}

__global__ void Advanced_Sobel_Kernel(unsigned char *ptr, unsigned short* out, int width, int height)
{
	const int block_size = 32;
	__shared__ uchar block_cache[block_size *2][block_size *2];  // 32*32 is block size, each thread in a block load 4 element
	int idx = blockIdx.x*blockDim.x + threadIdx.x;  // image coordinate
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	
	int pixel_id = idy * width + idx;

	
	int x, y;
	// upper left
	x = idx - block_size / 2 + threadIdx.x; // minus half of block edge size;
	y = idy - block_size / 2 + threadIdx.y;
	for (int i = 0; i < 2; i++)  // col
	{
		for (int j = 0; j < 2; j++)  // row
		{
			// sample in image;
			int temp_x = x + i;  // image coordinate
			int temp_y = y + j;
			int value;
			if (temp_x <0 || temp_x>(width - 1))
				value = 0;
			else if (temp_y<0 || temp_y>(height - 1))
				value = 0;
			else
				value = ptr[temp_y*width + temp_x];
			block_cache[2*threadIdx.y + j][2*threadIdx.x + i] = value;
			//printf("what are you doing?\n");
		}
	}
	__syncthreads();
	//printf("after sync threads\n");
	
	if (idx < width && idy < height)
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

		float resultx = 0;
		int centerx_in_cache = threadIdx.y + block_size / 2;
		int centery_in_cache = threadIdx.x + block_size / 2;

		for (int i = -1*kernel_radius; i <= kernel_radius; i++)  // row
		{
			for (int j = -1*kernel_radius; j <= kernel_radius; j++)  // col
			{
				int cachex = centerx_in_cache + i;
				int cachey = centery_in_cache + j;
				resultx += kernelx[i + kernel_radius][j + kernel_radius] * block_cache[cachex][cachey];
			}
		}

		resultx = abs(resultx);
		int temp = resultx;
		//temp = temp > 65535 ? 65535 : temp;
		//out[pixel_id] = temp;
		
		out[pixel_id] = temp;

		//printf("x:%d y:%d origin:%d result:%d\n", idx, idy, p4, temp);
		//printf("after sync in kernel\n");
	}
	
	
}

__global__ void Sobel_Cache(unsigned char *ptr, unsigned short* out, int width, int height)
{
	const int block_size = 32;
	__shared__ uchar block_cache[block_size + 2][block_size + 2];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int pixel_id = idy * width + idx;

	// load data into shared memory
	if (idx < width || idy < height ) {
		int value = ptr[pixel_id];
		block_cache[threadIdx.y + 1][threadIdx.x + 1] = value; // read itself into cache
	
		// read left-up corner element
		int tempx, tempy;
		if (threadIdx.x == 0 && threadIdx.y == 0) {
			tempx = idx - 1; 
			tempy = idy - 1;
			if (tempx>=0 && tempy>=0)
				block_cache[0][0] = ptr[tempy*width + tempx];
			else
				block_cache[0][0] = 0;
		}
		// read right-up corner element
		else if (threadIdx.x == block_size - 1 && threadIdx.y == 0) {
			tempx = idx + 1;
			tempy = idy - 1;
			if (tempx < width && tempy >=0)
				block_cache[0][block_size+1] = ptr[tempy*width + tempx];
			else
				block_cache[0][block_size+1] = 0;
		}
		// read left-bottom corner element
		else if (threadIdx.x == 0 && threadIdx.y == block_size - 1) {
			tempx = idx - 1;
			tempy = idy + 1;
			if (tempx >=0 && tempy<height)
				block_cache[block_size+1][0] = ptr[tempy*width + tempx];
			else
				block_cache[block_size+1][0] = 0;
		}
		// read right-bottom corner element
		else if (threadIdx.x == block_size - 1 && threadIdx.y == block_size - 1) {
			tempx = idx + 1;
			tempy = idy + 1;
			if (tempx < width && tempy < height)
				block_cache[block_size+1][block_size+1] = ptr[tempy*width + tempx];
			else
				block_cache[block_size+1][block_size+1] = 0;
		}

		// read up side
		if (threadIdx.y == 0) {
			tempx = idx;
			tempy = idy - 1;
			if (tempy >= 0)
				block_cache[0][threadIdx.x + 1] = ptr[tempy*width + tempx];
			else
				block_cache[0][threadIdx.x + 1] = 0;
		}
		// read bottom side
		else if (threadIdx.y == block_size - 1) {
			tempx = idx;
			tempy = idy + 1;
			if (tempy < height)
				block_cache[block_size+1][threadIdx.x + 1] = ptr[tempy*width + tempx];
			else
				block_cache[block_size+1][threadIdx.x + 1] = 0;
		}
		// left
		if (threadIdx.x == 0) {
			tempx = idx - 1;
			tempy = idy;
			if (tempx >= 0)
				block_cache[threadIdx.y + 1][0] = ptr[tempy*width + tempx];
			else
				block_cache[threadIdx.y + 1][0] = 0;
		}
		else if (threadIdx.x == block_size - 1)
		{
			tempx = idx + 1;
			tempy = idy;
			if (tempx < width)
				block_cache[threadIdx.y + 1][block_size+1] = ptr[tempy*width + tempx];
			else
				block_cache[threadIdx.y + 1][block_size+1] = 0;
		}


	}
	
	__syncthreads();

	if (idx < width && idy < height)
	{
		float kernelx[5][5] = {


			0.0369,    0.0392,    0.0400    ,0.0392 ,   0.0369,
			0.0392 ,   0.0416 ,   0.0424   , 0.0416  ,  0.0392,
			0.0400  ,  0.0424  ,  0.0433  ,  0.0424   , 0.0400,
			0.0392   , 0.0416   , 0.0424 ,   0.0416    ,0.0392,
			0.0369    ,0.0392    ,0.0400,    0.0392    ,0.0369
		};
		float resultx = 0;
		//int resulty = 0;
		int centerx_in_cache = threadIdx.y + 1;
		int centery_in_cache = threadIdx.x + 1;

		for (int i = -2; i <= 2; i++)  // row
		{
			for (int j = -2; j <= 2; j++)  // col
			{
				int cachex = centerx_in_cache + i;
				int cachey = centery_in_cache + j;
				resultx += kernelx[i + 1][j + 1] * block_cache[cachex][cachey];
			}
		}

		resultx = abs(resultx);

		int temp = resultx;
		//temp = temp > 65535 ? 65535 : temp;
		//out[pixel_id] = temp;

		out[pixel_id] = temp;
	}
}

int main(int argc, char**argv)
{
	cv::Mat test = cv::imread(argv[1], 0);
	cv::imshow("origin", test);
	int width, height, channels;
	width = test.cols;
	height = test.rows;
	channels = test.channels();
	printf("channel: %d\n", channels);

#pragma region CPU_OpenCV
	cv::Mat StandardCVResult;
	clock_t start = clock();
	for (int i = 0; i < TEST_TIMES; i++)
		//cv::Sobel(test, StandardCVResult, CV_8U, 1, 1, 3);
		cv::GaussianBlur(test, StandardCVResult, cv::Size(9,9), 5, 5);
	clock_t finish = clock();

	printf("Standard OpenCV Sobel Timing: %f ms\n", 1000 * (double)(finish - start) / CLOCKS_PER_SEC);
	cv::imshow("standard cv", StandardCVResult);
#pragma endregion

	// allocate host memory
	uchar* h_image_raw_data = new uchar[width*height * channels];
	memcpy(h_image_raw_data, test.data, width*height * channels);

	// allocate device data
	uchar* d_image_raw_data;
	unsigned short* d_out_data;
	cudaMalloc((void**)&d_image_raw_data, width*height * channels);
	cudaMalloc((void**)&d_out_data, width*height*channels*sizeof(unsigned short));
	cudaMemcpy(d_image_raw_data, test.data, width*height * channels, cudaMemcpyHostToDevice);


	dim3 block_size, grid_size;

#if 1
#pragma region GPU_Simple
	block_size = dim3(32, 32, channels);
	grid_size = dim3((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y, 1);

	cudaEvent_t start_cuda, finish_cuda;
	float time;
	cudaEventCreate(&start_cuda, 0);
	cudaEventCreate(&finish_cuda, 0);

	cudaEventRecord(start_cuda, 0);
	//start = clock();
	for (int i = 0; i < TEST_TIMES; i++)
	{
		Simple_SobelX_Kernel << <grid_size, block_size >> > (d_image_raw_data, d_out_data, width, height, channels);
		cudaDeviceSynchronize();
	}
	//finish = clock();
	cudaEventRecord(finish_cuda, 0);
	cudaEventSynchronize(finish_cuda);
	cudaEventElapsedTime(&time, start_cuda, finish_cuda);
	cudaEventDestroy(start_cuda);
	cudaEventDestroy(finish_cuda);

	printf("Simple GPU Timing: %f ms\n", time);
	//printf("Simple GPU Timing2: %f ms\n", 1000.0 * (finish - start) / CLOCKS_PER_SEC);
	cv::Mat SimpleResult(height, width, CV_16UC1); 
	cudaMemcpy(SimpleResult.data, d_out_data, width*height * channels * sizeof(unsigned short), cudaMemcpyDeviceToHost);
	cv::Mat SimpleResult2(height, width, CV_8UC1);
	SimpleResult.convertTo(SimpleResult2, CV_8UC1, 1);
	cv::imshow("simple kernel", SimpleResult2);

#pragma endregion
#endif

#if 1
#pragma region Advanced_GPU
	block_size = dim3(32, 32, 1);
	grid_size = dim3((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y, 1);
	
	//cudaMemset(d_out_data, 0, width*height * sizeof(unsigned short));
	time = 0;
	cudaEventCreate(&start_cuda, 0);
	cudaEventCreate(&finish_cuda, 0);

	cudaEventRecord(start_cuda, 0);
	for (int i = 0; i < TEST_TIMES; i++) {
		Advanced_Sobel_Kernel << <grid_size, block_size >> > (d_image_raw_data, d_out_data, width, height);
		cudaDeviceSynchronize();
	}
	cudaEventRecord(finish_cuda, 0);
	cudaEventSynchronize(finish_cuda);
	cudaEventElapsedTime(&time, start_cuda, finish_cuda);
	cudaEventDestroy(start_cuda);
	cudaEventDestroy(finish_cuda);
	//printf("device syn\n");
	printf("Advanced GPU:%f ms\n", time);

	cv::Mat AdvancedResult(height, width, CV_16UC1);
	cudaMemcpy(AdvancedResult.data, d_out_data, width*height * sizeof(unsigned short), cudaMemcpyDeviceToHost);
	cv::Mat AdvancedResult2(height, width, CV_8UC1);
	AdvancedResult.convertTo(AdvancedResult2, CV_8UC1, 1);
	cv::imshow("advanced kernel", AdvancedResult2);

#pragma endregion
#endif

#if 0
#pragma region Less_Cache
	block_size = dim3(32, 32, 1);
	grid_size = dim3((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y, 1);
	
	time = 0;
	cudaEventCreate(&start_cuda, 0);
	cudaEventCreate(&finish_cuda, 0);

	cudaEventRecord(start_cuda, 0);
	for (int i = 0; i < TEST_TIMES; i++) {
		Sobel_Cache << <grid_size, block_size >> > (d_image_raw_data, d_out_data, width, height);
		cudaDeviceSynchronize();
	}
	cudaEventRecord(finish_cuda, 0);
	cudaEventSynchronize(finish_cuda);
	cudaEventElapsedTime(&time, start_cuda, finish_cuda);
	cudaEventDestroy(start_cuda);
	cudaEventDestroy(finish_cuda);
	//printf("device syn\n");
	printf("Less Cache:%f ms\n", time);

	cv::Mat LessCacheResult(height, width, CV_16UC1);
	cudaMemcpy(LessCacheResult.data, d_out_data, width*height * sizeof(unsigned short), cudaMemcpyDeviceToHost);
	cv::Mat LessCacheResult2(height, width, CV_8UC1);
	LessCacheResult.convertTo(LessCacheResult2, CV_8UC1, 1);
	cv::imshow("less cache", LessCacheResult2);
#pragma endregion
#endif
	cudaEventDestroy(start_cuda);
	cudaEventDestroy(finish_cuda);
	cv::waitKey();

	cudaFree(d_image_raw_data);
	free(h_image_raw_data);
	cudaFree(d_out_data);
	return 0;
}


