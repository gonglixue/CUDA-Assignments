#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <opencv2\opencv.hpp>
#include <stdio.h>

__global__ void Simple_SobelX_Kernel(unsigned char *ptr, unsigned char* out, int width, int height, int depth)
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
	//if (resultx < 0)
	//	resultx = 0;
	resultx = abs(resultx);
	if (resultx > 255)
		resultx = 255;


	int resulty = -1 * p0 - 2 * p1 - 1 * p2
		+ p6 + 2 * p7 + p8;
	//if (resulty < 0)
	//	resulty = 0;
	resulty = abs(resulty);
	if (resulty > 255)
		resulty = 255;

	//int temp = sqrtf(resultx*resultx + resulty*resulty);
	int temp = resultx + resulty;
	if (temp > 255)
		temp = 255;
	out[pixel_id * depth + channel] = temp;
	//out[pixel_id * depth + channel] = resultx + resulty;
	//printf("x:%d, y:%d  z;%d origin:%d result:%d width:%d height:%d\n", idx, idy, channel, p4, result, width, height);
	
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

	cv::Mat StandardCVResult;
	cv::Sobel(test, StandardCVResult, CV_8U, 1, 1, 3);
	cv::imshow("standard cv", StandardCVResult);

	// allocate host memory
	uchar* h_image_raw_data = new uchar[width*height * channels];
	memcpy(h_image_raw_data, test.data, width*height * channels);

	// allocate device data
	uchar* d_image_raw_data;
	uchar* d_out_data;
	cudaMalloc((void**)&d_image_raw_data, width*height * channels);
	cudaMalloc((void**)&d_out_data, width*height*channels);
	cudaMemcpy(d_image_raw_data, test.data, width*height * channels, cudaMemcpyHostToDevice);

	dim3 block_size = dim3(16, 16, channels);
	dim3 grid_size = dim3((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y, 1);
	Simple_SobelX_Kernel << <grid_size, block_size >> > (d_image_raw_data, d_out_data,  width, height, channels);
	cudaDeviceSynchronize();

	cudaMemcpy(h_image_raw_data, d_out_data, width*height * channels, cudaMemcpyDeviceToHost);
	cv::Mat SimpleResult(height, width, CV_8UC1);
	SimpleResult.data = h_image_raw_data;
	cv::imshow("simple kernel", SimpleResult);


	
	cv::waitKey();

	cudaFree(d_image_raw_data);
	free(h_image_raw_data);
	return 0;
}


