#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <iostream>

using namespace std;

__global__ void kernel_square(uchar4 *img, int size, uchar4 color1, uchar4 color2)
{
	int2 pos = {
		blockDim.x * blockIdx.x + threadIdx.x,
		blockDim.y * blockIdx.y + threadIdx.y
	};

	if (pos.x > size || pos.y > size)
	{
		return;
	}

	double2 mid = {
		abs(((double)size / (double)2) - pos.x),
		abs(((double)size / (double)2) - pos.y)
	};

	double midd = mid.x > mid.y ? mid.x : mid.y;

	midd = (double)(2 * midd) / (double)(size);

	uchar4 bgr;

	bgr.x = color1.x * midd + color2.x * (1 - midd);
	bgr.y = color1.y * midd + color2.y * (1 - midd);
	bgr.z = color1.z * midd + color2.z * (1 - midd);

	img[pos.y * size + pos.x] = bgr;

	return;
}

__global__ void kernel_grid(uchar4 *img, int size, int step, uchar4 grid_color)
{
	int2 pos = {
		blockDim.x * blockIdx.x + threadIdx.x,
		blockDim.y * blockIdx.y + threadIdx.y
	};

	if (pos.x > size || pos.y > size)
	{
		return;
	}

	if (pos.x % step == 0 || pos.y % step == 0)
	{
		img[pos.y * size + pos.x] = grid_color;
	}
	return;
}

__global__ void kernel_circ(uchar4 *img, int size, int2 pos, double radius, uchar4 color)
{
	int2 act_pos = {
		blockDim.x * blockIdx.x + threadIdx.x,
		blockDim.y * blockIdx.y + threadIdx.y
	};

	if (pos.x > size || pos.y > size)
	{
		return;
	}

	double length = sqrt(
		(double)(abs(pos.x - act_pos.x) * abs(pos.x - act_pos.x)) + 
		(double)(abs(pos.y - act_pos.y) * abs(pos.y - act_pos.y))
	);

	if (length <= radius)
	{
		img[act_pos.y * size + act_pos.x] = color;
	}
	return;
}

__host__ void run_kernel(uchar4 *img,
						 int size,
						 uchar4 color1,
						 uchar4 color2,
						 int grid_step,
						 uchar4 grid_color,
						 int2 circ_pos,
						 double radius,
						 uchar4 circ_color)
{
	cudaError_t cudaErr;
	uchar4 *cudaPic;
	int2 block = {16, 16};

	cudaErr = cudaMalloc(&cudaPic, size * size * sizeof(uchar4));
	if (cudaErr != cudaSuccess)
	{
		cout << "CUDA Error, line " << __LINE__ << " - " << cudaGetErrorString(cudaErr) << endl;
	}

	cudaErr = cudaMemcpy(cudaPic, img, size * size * sizeof(uchar4), cudaMemcpyHostToDevice);
	if (cudaErr != cudaSuccess)
	{
		cout << "CUDA Error, line " << __LINE__ << " - " << cudaGetErrorString(cudaErr) << endl;
	}

	dim3 grid(
		(size + block.x - 1) / block.x,
		(size + block.y - 1) / block.y
	);
	dim3 blocks(block.x, block.y);

	kernel_square<<< grid, blocks >>>(cudaPic, size, color1, color2);
	kernel_grid<<< grid, blocks >>>(cudaPic, size, grid_step, grid_color);
	kernel_circ<<< grid, blocks >>>(cudaPic, size, circ_pos, radius, circ_color);

	if ((cudaErr = cudaGetLastError()) != cudaSuccess)
	{
		cout << "CUDA Error, line " << __LINE__ << " - " << cudaGetErrorString(cudaErr) << endl;
	}

	cudaErr = cudaMemcpy(img, cudaPic, size * size * sizeof(uchar4), cudaMemcpyDeviceToHost);
	if (cudaErr != cudaSuccess)
	{
		cout << "CUDA Error, line " << __LINE__ << " - " << cudaGetErrorString(cudaErr) << endl;
	}

	cudaFree(cudaPic);

	return;
}