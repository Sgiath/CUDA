#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void kernel_invert(uchar4 *orig_pic, uchar4 *output_pic, int2 img_dim)
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
	bgr.x = 255 - bgr.x;
	bgr.y = 255 - bgr.y;
	bgr.z = 255 - bgr.z;

	output_pic[pos.y * img_dim.x + pos.x] = bgr;

	return;
}

__host__ void run_kernel_invert(uchar4 *orig, uchar4 *fin, int2 img_dim)
{
	cudaError_t cerr;
	uchar4 *cudaOrigPic;
	uchar4 *cudaInvertPic;
	int2 block = {16, 16};

	cerr = cudaMalloc(&cudaOrigPic, img_dim.x * img_dim.y * sizeof(uchar4));
	if (cerr != cudaSuccess)
		printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(cerr));

	cerr = cudaMalloc(&cudaInvertPic, img_dim.x * img_dim.y * sizeof(uchar4));
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

	kernel_invert<<< blocks, threads >>>(cudaOrigPic, cudaInvertPic, img_dim);

	if ((cerr = cudaGetLastError()) != cudaSuccess)
		printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(cerr));

	cerr = cudaMemcpy(fin, cudaInvertPic, img_dim.x * img_dim.y * sizeof(uchar4), cudaMemcpyDeviceToHost);
	if (cerr != cudaSuccess)
		printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(cerr));

	// Free memory
	cudaFree(cudaOrigPic);
	cudaFree(cudaInvertPic);

	return;
}