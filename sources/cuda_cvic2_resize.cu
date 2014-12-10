#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

__global__ void kernel_resize(uchar4 *orig_pic, uchar4 *output_pic, int2 img_dim, int2 new_img_dim, int index)
{
	int2 pos = {
		blockDim.x * blockIdx.x + threadIdx.x,
		blockDim.y * blockIdx.y + threadIdx.y
	};

	if (pos.x > img_dim.x || pos.y > img_dim.y)
	{
		return;
	}

	uchar4 bgr = orig_pic[pos.y * img_dim.x + pos.x];
	bgr.w = 0;

	output_pic[(pos.y * index) * new_img_dim.x + (pos.x * index)] = bgr;

	bgr.w = 1;

	for (int x = pos.x * index; x < pos.x * index + index; x++)
	{
		for (int y = pos.y * index; y < pos.y * index + index; y++)
		{
			if (x >= new_img_dim.x || y >= new_img_dim.y)
			{
				continue;
			}
			output_pic[y * new_img_dim.x + x] = bgr;
		}
	}

	return;
}

__global__ void kernel_interpolar(uchar4 *orig_pic, uchar4 *output_pic, int2 img_dim)
{
	int2 pos = {
		blockDim.x * blockIdx.x + threadIdx.x,
		blockDim.y * blockIdx.y + threadIdx.y
	};

	if (pos.x > img_dim.x || pos.y > img_dim.y)
	{
		return;
	}

	if (orig_pic[pos.y * img_dim.x + pos.x].w == 0)
	{
		return;
	}

	int3 sum = {0, 0, 0};
	int count = 0;

	for (int x = pos.x - 1; x <= pos.x + 1; x++)
	{
		for (int y = pos.y - 1; y <= pos.y + 1; y++)
		{
			if (x > img_dim.x || y > img_dim.y || x < 0 || y < 0)
			{
				continue;
			}
			if (orig_pic[y * img_dim.x + x].w == 1)
			{
				continue;
			}
			if (x == pos.x && y == pos.y)
			{
				continue;
			}
			sum.x = orig_pic[y * img_dim.x + x].x;
			sum.y = orig_pic[y * img_dim.x + x].y;
			sum.z = orig_pic[y * img_dim.x + x].z;
			count++;
		}
	}

	if (count == 0)
	{
		return;
	}

	uchar4 bgr = {
		(int)round((double)sum.x / (double)count),
		(int)round((double)sum.y / (double)count),
		(int)round((double)sum.z / (double)count),
		0
	};

	output_pic[pos.y * img_dim.x + pos.x] = bgr;

	return;
}

__host__ void run_kernel_resize(uchar4 *orig, uchar4 *fin, int2 img_dim, int2 new_img_dim, int index)
{
	cudaError_t cerr;
	uchar4 *cudaOrigPic;
	uchar4 *cudaResizePic;
	uchar4 *cudaInterpolarPic;
	int2 block = {16, 16};

	cerr = cudaMalloc(&cudaOrigPic, img_dim.x * img_dim.y * sizeof(uchar4));
	if (cerr != cudaSuccess)
		printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(cerr));

	cerr = cudaMalloc(&cudaResizePic, new_img_dim.x * new_img_dim.y * sizeof(uchar4));
	if (cerr != cudaSuccess)
		printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(cerr));

	cerr = cudaMalloc(&cudaInterpolarPic, new_img_dim.x * new_img_dim.y * sizeof(uchar4));
	if (cerr != cudaSuccess)
		printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(cerr));

	cerr = cudaMemcpy(cudaOrigPic, orig, img_dim.x * img_dim.y * sizeof(uchar4), cudaMemcpyHostToDevice);
	if (cerr != cudaSuccess)
		printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(cerr));

	cerr = cudaMemcpy(cudaResizePic, fin, new_img_dim.x * new_img_dim.y * sizeof(uchar4), cudaMemcpyHostToDevice);
	if (cerr != cudaSuccess)
		printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(cerr));

	dim3 blocks(
		(img_dim.x + block.x - 1) / block.x,
		(img_dim.y + block.y - 1) / block.y
		);
	dim3 threads(block.x, block.y);

	kernel_resize<<< blocks, threads >>>(cudaOrigPic, cudaResizePic, img_dim, new_img_dim, index);

	blocks.x = (new_img_dim.x + block.x - 1) / block.x;
	blocks.y = (new_img_dim.y + block.y - 1) / block.y;

	for (int i = 0; i < index; i++)
	{
		kernel_interpolar<<< blocks, threads >>>(cudaResizePic, cudaInterpolarPic, new_img_dim);
	}

	if ((cerr = cudaGetLastError()) != cudaSuccess)
		printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(cerr));

	cerr = cudaMemcpy(fin, cudaResizePic, new_img_dim.x * new_img_dim.y * sizeof(uchar4), cudaMemcpyDeviceToHost);
	if (cerr != cudaSuccess)
		printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(cerr));

	// Free memory
	cudaFree(cudaOrigPic);
	cudaFree(cudaResizePic);
	cudaFree(cudaInterpolarPic);

	return;
}