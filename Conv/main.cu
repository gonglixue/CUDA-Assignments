#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <opencv2\opencv.hpp>
#include <stdio.h>
#include <time.h>


__global__ void Simple_SobelX_Kernel(unsigned char *ptr, unsigned short* out, int width, int height, int depth)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int channel = threadIdx.z;

	if (idx > width || idy > height)
		return;
	if (idx == 0 || idx == width - 1)
		return;
	if (idy == 0 || idy == height - 1)
		return;

	int pixel_id = idy * width + idx;

	int p0 = ptr[(pixel_id-width-1)*depth + channel];
	int p1 = ptr[(pixel_id-width)*depth + channel];
	int p2 = ptr[(pixel_id-width+1)*depth + channel];

	int p3 = ptr[(pixel_id-1)*depth + channel];
	int p4 = ptr[pixel_id*depth + channel];
	int p5 = ptr[(pixel_id + 1) * depth + channel];

	int p6 = ptr[(pixel_id + width - 1) * depth + channel];
	int p7 = ptr[(pixel_id + width) * depth + channel];
	int p8 = ptr[(pixel_id + width + 1) * depth + channel];

	int resultx = -1 * p0 - 2 * p3 - 1 * p6
		+ 1 * p2 + 2 * p5 + 1 * p8;
	resultx = abs(resultx);

	int resulty = -1 * p0 - 2 * p1 - 1 * p2
		+ p6 + 2 * p7 + p8;
	resulty = abs(resulty);

	//int temp = sqrtf(resultx*resultx + resulty*resulty);
	int temp = resultx + resulty;
	temp = temp > 65535 ? 65535 : temp;
	out[pixel_id * depth + channel] = temp;
	//out[pixel_id * depth + channel] = resultx + resulty;
	//printf("x:%d, y:%d  z;%d origin:%d result:%d width:%d height:%d\n", idx, idy, channel, p4, temp, width, height);
	
}

__global__ void Advanced_Sobel_Kernel(unsigned char* in, unsigned short* out, int width, int height)
{
	__shared__ uchar block_cache[32*2][32*2];  // 32*32 is block size, each thread in a block load 4 element
	int idx = blockIdx.x*blockDim.x + threadIdx.x;  // image coordinate
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int pixel_id = idy * width + idx;

	if (pixel_id > width*height)
		return;

	int x, y;
	// upper left
	x = 2 * idx - 16; // minus half of block edge size;
	y = 2 * idy - 16;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 2; j++)
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
				value = in[pixel_id];
			block_cache[threadIdx.x + i][threadIdx.x + j] = value;
		}
	}
	__syncthreads();

	// convolution
	int centerx_in_cache = 2 * threadIdx.x + 16;
	int centery_in_cache = 2 * threadIdx.y + 16;
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

#pragma region GPU_Simple
	dim3 block_size = dim3(16, 16, channels);
	dim3 grid_size = dim3((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y, 1);

	cudaEvent_t start_cuda, finish_cuda;
	float time;
	cudaEventCreate(&start_cuda, 0);
	cudaEventCreate(&finish_cuda, 0);

	cudaEventRecord(start_cuda, 0);
	start = clock();
	for (int i = 0; i < 100; i++)
	{
		Simple_SobelX_Kernel << <grid_size, block_size >> > (d_image_raw_data, d_out_data, width, height, channels);
		cudaDeviceSynchronize();
	}
	finish = clock();
	cudaEventRecord(finish_cuda, 0);
	cudaEventSynchronize(finish_cuda);
	cudaEventElapsedTime(&time, start_cuda, finish_cuda);
	cudaEventDestroy(start_cuda);
	cudaEventDestroy(finish_cuda);

	printf("Simple GPU Timing: %f ms\n", time);
	printf("Simple GPU Timing2: %f ms\n", 1000.0 * (finish - start) / CLOCKS_PER_SEC);
	cv::Mat SimpleResult(height, width, CV_16UC1); 
	cudaMemcpy(SimpleResult.data, d_out_data, width*height * channels * sizeof(unsigned short), cudaMemcpyDeviceToHost);
	cv::Mat SimpleResult2(height, width, CV_8UC1);
	SimpleResult.convertTo(SimpleResult2, CV_8UC1, 255.0 / 1000);
	cv::imshow("simple kernel", SimpleResult2);

#pragma endregion
	
	cv::waitKey();

	cudaFree(d_image_raw_data);
	free(h_image_raw_data);
	cudaFree(d_out_data);
	return 0;
}


