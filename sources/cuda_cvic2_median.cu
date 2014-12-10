#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

__global__ void kernel_median(uchar4 *orig_pic, uchar4 *output_pic, int2 img_dim, int neigh)
{
	int2 pos = {
		blockDim.x * blockIdx.x + threadIdx.x,
		blockDim.y * blockIdx.y + threadIdx.y
	};

	if (pos.x > img_dim.x || pos.y > img_dim.y)
	{
		return;
	}

	int4 sum = {0, 0, 0};
	int count = 0;


	for (int x = pos.x - neigh; x < pos.x + neigh; x++)
	{
		for (int y = pos.y - neigh; y < pos.y + neigh; y++)
		{
			if (x >= 0 && x < img_dim.x && y >= 0 && y < img_dim.y)
			{
				sum.x += orig_pic[y * img_dim.x + x].x;
				sum.y += orig_pic[y * img_dim.x + x].y;
				sum.z += orig_pic[y * img_dim.x + x].z;
				count++;
			}
		}
	}

	//actual point
	sum.x += orig_pic[pos.y * img_dim.x + pos.x].x;
	sum.y += orig_pic[pos.y * img_dim.x + pos.x].y;
	sum.z += orig_pic[pos.y * img_dim.x + pos.x].z;
	count++;

	uchar4 bgr = {
		(int)round((double)sum.x / (double)count),
		(int)round((double)sum.y / (double)count),
		(int)round((double)sum.z / (double)count)
	};

	output_pic[pos.y * img_dim.x + pos.x] = bgr;

	return;
}

__host__ void run_kernel_median(uchar4 *orig, uchar4 *fin, int2 img_dim, int neigh)
{
	cudaError_t cerr;
	uchar4 *cudaOrigPic;
	uchar4 *cudaMedPic;
	int2 block = {16, 16};

	cerr = cudaMalloc(&cudaOrigPic, img_dim.x * img_dim.y * sizeof(uchar4));
	if (cerr != cudaSuccess)
		printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(cerr));

	cerr = cudaMalloc(&cudaMedPic, img_dim.x * img_dim.y * sizeof(uchar4));
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

	kernel_median<<< blocks, threads >>>(cudaOrigPic, cudaMedPic, img_dim, neigh);

	if ((cerr = cudaGetLastError()) != cudaSuccess)
		printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(cerr));

	cerr = cudaMemcpy(fin, cudaMedPic, img_dim.x * img_dim.y * sizeof(uchar4), cudaMemcpyDeviceToHost);
	if (cerr != cudaSuccess)
		printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(cerr));

	cudaFree(cudaOrigPic);
	cudaFree(cudaMedPic);

	return;
}