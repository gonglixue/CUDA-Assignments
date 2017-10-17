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

__global__ void Simple_SobelX_Kernel(unsigned char *ptr, unsigned short* out, int width, int height, int depth)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	
	int pixel_id = idy * width + idx;
	if (idx > width || idy > height)
		return;
	if (idx == 0 || idx == width - 1)
	{
		out[pixel_id] = ptr[pixel_id];
		return;
	};
	if (idy == 0 || idy == height - 1)
	{
		out[pixel_id] = ptr[pixel_id];
		return;
	}

	

	int p0 = ptr[(pixel_id-width-1)];
	int p1 = ptr[(pixel_id-width)];
	int p2 = ptr[(pixel_id-width+1)];

	int p3 = ptr[(pixel_id-1)];
	int p4 = ptr[pixel_id];
	int p5 = ptr[(pixel_id + 1)];

	int p6 = ptr[(pixel_id + width - 1)];
	int p7 = ptr[(pixel_id + width)];
	int p8 = ptr[(pixel_id + width + 1)];

	int resultx = -1 * p0 - 2 * p3 - 1 * p6
		+ 1 * p2 + 2 * p5 + 1 * p8;
	resultx = abs(resultx);

	int resulty = -1 * p0 - 2 * p1 - 1 * p2
		+ p6 + 2 * p7 + p8;
	resulty = abs(resulty);

	//int temp = sqrtf(resultx*resultx + resulty*resulty);
	int temp = resultx + resulty;
	temp = temp > 65535 ? 65535 : temp;
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
			block_cache[2*threadIdx.x + j][2*threadIdx.y + i] = value;
			//printf("what are you doing?\n");
		}
	}
	__syncthreads();
	//printf("after sync threads\n");

	/*
	if (idx > width || idy > height)
		return;
	if (idx == 0 || idx == width - 1)
	{
		out[pixel_id] = ptr[pixel_id];
		return;
	};
	if (idy == 0 || idy == height - 1)
	{
		out[pixel_id] = ptr[pixel_id];
		return;
	}

	int p0 = ptr[(pixel_id - width - 1)];
	int p1 = ptr[(pixel_id - width)];
	int p2 = ptr[(pixel_id - width + 1)];

	int p3 = ptr[(pixel_id - 1)];
	int p4 = ptr[pixel_id];
	int p5 = ptr[(pixel_id + 1)];

	int p6 = ptr[(pixel_id + width - 1)];
	int p7 = ptr[(pixel_id + width)];
	int p8 = ptr[(pixel_id + width + 1)];

	int resultx = -1 * p0 - 2 * p3 - 1 * p6
		+ 1 * p2 + 2 * p5 + 1 * p8;
	resultx = abs(resultx);

	int resulty = -1 * p0 - 2 * p1 - 1 * p2
		+ p6 + 2 * p7 + p8;
	resulty = abs(resulty);


	int temp = resultx + resulty;
	temp = temp > 65535 ? 65535 : temp;
	out[pixel_id] = temp;
	*/
	
	
	
	if (idx < width && idy < height)
	{
		
		// convolution
		int centerx_in_cache = threadIdx.x + block_size/2;
		int centery_in_cache = threadIdx.y + block_size/2;

		int p0 = block_cache[centerx_in_cache - 1][centery_in_cache - 1];
		int p1 = block_cache[centerx_in_cache][centery_in_cache - 1];
		int p2 = block_cache[centerx_in_cache + 1][centery_in_cache - 1];

		int p3 = block_cache[centerx_in_cache - 1][centery_in_cache];
		int p4 = block_cache[centerx_in_cache][centery_in_cache];
		int p5 = block_cache[centerx_in_cache + 1][centery_in_cache];

		int p6 = block_cache[centerx_in_cache - 1][centery_in_cache + 1];
		int p7 = block_cache[centerx_in_cache][centery_in_cache + 1];
		int p8 = block_cache[centerx_in_cache + 1][centery_in_cache + 1];

		int resultx = -1 * p0 - 2 * p3 - 1 * p6
			+ 1 * p2 + 2 * p5 + 1 * p8;
		resultx = abs(resultx);

		int resulty = -1 * p0 - 2 * p1 - 1 * p2
			+ p6 + 2 * p7 + p8;
		resulty = abs(resulty);

		//int temp = sqrtf(resultx*resultx + resulty*resulty);
		int temp = resultx + resulty;
		temp = temp > 65535 ? 65535 : temp;
		//out[pixel_id] = temp;
		
		out[pixel_id] = temp;

		//printf("x:%d y:%d origin:%d result:%d\n", idx, idy, p4, temp);
		//printf("after sync in kernel\n");
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
	for(int i=0;i<100;i++)
		cv::Sobel(test, StandardCVResult, CV_8U, 1, 1, 3);
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
	block_size = dim3(16, 16, channels);
	grid_size = dim3((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y, 1);

	cudaEvent_t start_cuda, finish_cuda;
	float time;
	cudaEventCreate(&start_cuda, 0);
	cudaEventCreate(&finish_cuda, 0);

	cudaEventRecord(start_cuda, 0);
	//start = clock();
	for (int i = 0; i < 100; i++)
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
	SimpleResult.convertTo(SimpleResult2, CV_8UC1, 255.0 / 1000);
	cv::imshow("simple kernel", SimpleResult2);

#pragma endregion
#endif

#if 1
#pragma region Advanced_GPU
	block_size = dim3(32, 32, 1);
	grid_size = dim3((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y, 1);
	printf("cpu test: block_size %d grid_size %d\n", block_size.x, grid_size.x);


	//cudaMemset(d_out_data, 0, width*height * sizeof(unsigned short));
	cudaEvent_t start_cuda2, finish_cuda2;
	time = 0;
	cudaEventCreate(&start_cuda2, 0);
	cudaEventCreate(&finish_cuda2, 0);

	cudaEventRecord(start_cuda2, 0);
	for (int i = 0; i < 100; i++) {
		Advanced_Sobel_Kernel << <grid_size, block_size >> > (d_image_raw_data, d_out_data, width, height);
		cudaDeviceSynchronize();
	}
	cudaEventRecord(finish_cuda2, 0);
	cudaEventSynchronize(finish_cuda2);
	cudaEventElapsedTime(&time, start_cuda2, finish_cuda2);
	cudaEventDestroy(start_cuda2);
	cudaEventDestroy(finish_cuda2);
	//printf("device syn\n");
	printf("Advanced GPU:%f ms\n", time);

	cv::Mat AdvancedResult(height, width, CV_16UC1);
	cudaMemcpy(AdvancedResult.data, d_out_data, width*height * sizeof(unsigned short), cudaMemcpyDeviceToHost);
	cv::Mat AdvancedResult2(height, width, CV_8UC1);
	AdvancedResult.convertTo(AdvancedResult2, CV_8UC1, 255.0 / 1000);
	cv::imshow("advanced kernel", AdvancedResult2);

	cv::FileStorage fsFeature("./ad_r.xml", cv::FileStorage::WRITE);
	fsFeature << "dataMat2" << AdvancedResult;
	fsFeature.release();

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


