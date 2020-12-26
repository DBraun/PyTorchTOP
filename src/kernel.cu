/* Shared Use License: This file is owned by Derivative Inc. (Derivative)
* and can only be used, and/or modified for use, in conjunction with
* Derivative's TouchDesigner software, and only if you are a licensee who has
* accepted Derivative's TouchDesigner license or assignment agreement
* (which also govern the use of this file). You may share or redistribute
* a modified version of this file provided the following conditions are met:
*
* 1. The shared file or redistribution must retain the information set out
* above and this list of conditions.
* 2. Derivative's name (Derivative Inc.) or its trademarks may not be used
* to endorse or promote products derived from this file without specific
* prior written permission from Derivative.
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

surface<void, cudaSurfaceType2D> inputSurface;
surface<void, cudaSurfaceType2D> outputSurface;

__global__ void
copyTextureRGBA8(int width, int height)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	uchar4 color;
	surf2Dread(&color, inputSurface, x * 4, y);
	surf2Dwrite(color, outputSurface, x * 4, y);
}

__global__ void
makeOutputRed(int width, int height)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	uchar4 color = make_uchar4(255, 0, 0, 255);
	surf2Dwrite(color, outputSurface, x * 4, y);
}

int
divUp(int a, int b)
{
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t doCUDAOperation(int width, int height, cudaArray *input, cudaArray *output)
{
    cudaError_t cudaStatus;

	cudaBindSurfaceToArray(outputSurface, output);
	
	dim3 blockSize(16, 16, 1);

	dim3 gridSize(divUp(width, blockSize.x), divUp(height, blockSize.y), 1);

	if (input)
	{
		cudaBindSurfaceToArray(inputSurface, input);
		copyTextureRGBA8<<<gridSize, blockSize>>>(width, height);
	}
	else
	{
		makeOutputRed << <gridSize, blockSize >> > (width, height);
	}


#ifdef _DEBUG
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }
#else
	cudaStatus = cudaSuccess;
#endif

    return cudaStatus;
}
