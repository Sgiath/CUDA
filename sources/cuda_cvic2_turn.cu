#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void kernel_turn_right(uchar4 *orig_pic, uchar4 *output_pic, int2 img_dim, int2 new_img_dim)
{
	int2 pos = {
		blockDim.x * blockIdx.x + threadIdx.x,
		blockDim.y * blockIdx.y + threadIdx.y
	};

	if (pos.x > img_dim.x || pos.y > img_dim.y)
	{
		return;
	}

	output_pic[pos.x * new_img_dim.x + (new_img_dim.x - pos.y)] = orig_pic[pos.y * img_dim.x + pos.x];

	return;
}

__global__ void kernel_turn_opposit(uchar4 *orig_pic, uchar4 *output_pic, int2 img_dim)
{
	int2 pos = {
		blockDim.x * blockIdx.x + threadIdx.x,
		blockDim.y * blockIdx.y + threadIdx.y
	};

	if (pos.x > img_dim.x || pos.y > img_dim.y)
	{
		return;
	}

	output_pic[(img_dim.y - pos.y) * img_dim.x + (img_dim.x - pos.x)] = orig_pic[pos.y * img_dim.x + pos.x];

	return;
}

__global__ void kernel_turn_left(uchar4 *orig_pic, uchar4 *output_pic, int2 img_dim, int2 new_img_dim)
{
	int2 pos = {
		blockDim.x * blockIdx.x + threadIdx.x,
		blockDim.y * blockIdx.y + threadIdx.y
	};

	if (pos.x > img_dim.x || pos.y > img_dim.y)
	{
		return;
	}

	output_pic[(new_img_dim.y - pos.x) * new_img_dim.x + pos.y] = orig_pic[pos.y * img_dim.x + pos.x];

	return;
}

__host__ int run_kernel_turn(uchar4 *orig, uchar4 *fin, int2 img_dim, int2 new_img_dim, int turn)
{
	if (turn != 1 && turn != 2 && turn != 3) return -1;

	cudaError_t cerr;
	uchar4 *cudaOrigPic;
	uchar4 *cudaTurnPic;
	int2 block = {16, 16};

	cerr = cudaMalloc(&cudaOrigPic, img_dim.x * img_dim.y * sizeof(uchar4));
	if (cerr != cudaSuccess)
		printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(cerr));

	cerr = cudaMalloc(&cudaTurnPic, new_img_dim.x * new_img_dim.y * sizeof(uchar4));
	if (cerr != cudaSuccess)
		printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(cerr));

	cerr = cudaMemcpy(cudaOrigPic, orig, img_dim.x * img_dim.y * sizeof(uchar4), cudaMemcpyHostToDevice);
	if (cerr != cudaSuccess)
		printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(cerr));

	dim3 blocks(
		(img_dim.x + block.x - 1) / block.x,
		(img_dim.y + block.y - 1) / block.y
	);
	dim3 threads(block.x, block.y);

	if (turn == 1)
	{
		kernel_turn_right<<< blocks, threads >>>(cudaOrigPic, cudaTurnPic, img_dim, new_img_dim);
	}
	else if (turn == 2)
	{
		kernel_turn_opposit<<< blocks, threads >>>(cudaOrigPic, cudaTurnPic, img_dim);
	}
	else
	{
		kernel_turn_left<<< blocks, threads >>>(cudaOrigPic, cudaTurnPic, img_dim, new_img_dim);
	}

	if ((cerr = cudaGetLastError()) != cudaSuccess)
		printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(cerr));

	cerr = cudaMemcpy(fin, cudaTurnPic, new_img_dim.x * new_img_dim.y * sizeof(uchar4), cudaMemcpyDeviceToHost);
	if (cerr != cudaSuccess)
		printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(cerr));

	cudaFree(cudaOrigPic);
	cudaFree(cudaTurnPic);

	return 0;
}