#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void kernel_cut(uchar4 *orig_pic, uchar4 *output_pic, int2 img_dim, int2 new_img_dim, int4 cut)
{
	int2 pos = {
		blockDim.x * blockIdx.x + threadIdx.x,
		blockDim.y * blockIdx.y + threadIdx.y
	};

	if (pos.x < cut.x || pos.x >= img_dim.x - cut.y || pos.y < cut.z || pos.y >= img_dim.y - cut.w)
	{
		return;
	}
	else
	{
		output_pic[(pos.y - cut.z) * new_img_dim.x + (pos.x - cut.x)] = orig_pic[pos.y * img_dim.x + pos.x];
	}
	
	return;
}

__host__ void run_kernel_cut(uchar4 *orig, uchar4 *fin, int2 img_dim, int2 new_img_dim, int4 cut)
{
	cudaError_t cerr;
	uchar4 *cudaOrigPic;
	uchar4 *cudaCutPic;
	int2 block = {16, 16};

	cerr = cudaMalloc(&cudaOrigPic, img_dim.x * img_dim.y * sizeof(uchar4));
	if (cerr != cudaSuccess)
		printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(cerr));

	cerr = cudaMalloc(&cudaCutPic, new_img_dim.x * new_img_dim.y * sizeof(uchar4));
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

	kernel_cut<<< blocks, threads >>>(cudaOrigPic, cudaCutPic, img_dim, new_img_dim, cut);

	if ((cerr = cudaGetLastError()) != cudaSuccess)
		printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(cerr));

	cerr = cudaMemcpy(fin, cudaCutPic, new_img_dim.x * new_img_dim.y * sizeof(uchar4), cudaMemcpyDeviceToHost);
	if (cerr != cudaSuccess)
		printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(cerr));

	// Free memory
	cudaFree(cudaOrigPic);
	cudaFree(cudaCutPic);

	return;
}