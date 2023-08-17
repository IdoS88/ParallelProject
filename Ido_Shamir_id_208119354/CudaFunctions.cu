#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "math.h"
#include <assert.h>
#include "main.h"
#include "sm_60_atomic_functions.h"
#include <stdio.h>
// ======================================= Service methods =======================================

//int offset = (blockIdx.y * blockDim.y + threadIdx.y) * (N - W + 1) + blockIdx.x * blockDim.x + threadIdx.x;
//int offset = blockIdx.y * blockDim.y * N + blockIdx.x * blockDim.x + threadIdx.y * N + threadIdx.x;


__host__ void allocateMatOnGPU(Matrix image, int** deviceImage)
{
	int colorsInImage = image.size * image.size;
	cudaError_t error = cudaSuccess;

	// Allocates and copies the object to GPU
	error = cudaMalloc(deviceImage, colorsInImage * sizeof(int));
	if (error != cudaSuccess)
	{
		printf("Cannot allocate GPU memory for image: %s (%d)\n", cudaGetErrorString(error), error);
		exit(0);
	}
	error = cudaMemcpy(*deviceImage, image.data, colorsInImage * sizeof(int), cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
	{
		printf("Cannot copy image to GPU: %s (%d)\n", cudaGetErrorString(error), error);
		exit(0);
	}
}
__host__ void freeMatFromGPU(int** deviceMat)
{
	cudaError_t error = cudaSuccess;

	// Free the picture from GPU memory
	error = cudaFree(*deviceMat);
	if (error != cudaSuccess)
	{
		printf("Cannot free Mat from GPU: %s (%d)\n", cudaGetErrorString(error), error);
		exit(0);
	}
}
__device__ int getPositionsPerDimension(int pictureDim, int objectDim) // called from both CPU and GPU
{
	return (pictureDim - objectDim) + 1;
}

__device__ double difference(int p, int o)
{
	// printf("p %d o %d 0:%f 1: %f 2:%d\n", p, o, (double)(p - o) / p, ceil((double)(p - o) / p), abs(ceil((double)(p - o) / p)));
	//printf("p-o %lf  calculate:%lf final %lf\n", (double)(p - o), (double)(p - o) / p, fabs((double)(p - o) / p));
	return fabs((double)(p - o) / p);
}
// __global__ void findMatchingSubmatrix(int* deviceSubmatrix, int* deviceBigMatrix, int subMatrixDim, int bigMatrixDim, double* matching, int* positionFlags, double matchingV)
// {
// 	int tid = threadIdx.x + blockIdx.x * blockDim.x;

// 	int checkDim = pow(getPositionsPerDimension(bigMatrixDim, subMatrixDim), 2); //max size of matrix dim being checked
// 	if (tid < checkDim * checkDim) // make sure we access the required memory of elements
// 	{
// 		int i = tid / checkDim; // picture matrix row
// 		int j = tid % checkDim; // picture matrix col
// 		double sum = 0.0;
// 		for (int subMatrixOffset = 0; subMatrixOffset < subMatrixDim * subMatrixDim; subMatrixOffset++)
// 		{
// 			int r = subMatrixOffset / subMatrixDim; // object matrix row
// 			int c = subMatrixOffset % subMatrixDim; // object matrix col
// 			int idx = (i + r) * bigMatrixDim + j + c; // offset of the picture matrix
// 			assert(idx < bigMatrixDim * bigMatrixDim);
// 			sum += difference(deviceBigMatrix[idx], deviceSubmatrix[subMatrixOffset]);
// 		}
// 		matching[i * checkDim + j] = sum / (subMatrixDim * subMatrixDim);
// 		positionFlags[i * checkDim + j] = (matching[i * checkDim + j] < matchingV) ? 1 : 0;
// 	}
// }
__global__ void sliding_window_kernel(int* M, int M_size, int* N, int N_size, double* matching, int* positionFlags, double matchingV) {
	// // Compute the thread index
	// int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	// int sizeResults = getPositionsPerDimension(M_size, N_size);
	// printf("\n\n\nthread id %d", thread_id);
	// // Compute the row and column indices of the window
	// int window_row = thread_id / sizeResults;
	// int window_col = thread_id % sizeResults;
	// printf("\n\n\nwindow row %d window col %d", window_row, window_col);

	// // Check if the window is within the bounds of the larger matrix
	// if ((window_row + N_size) < M_size && (window_col + N_size) < M_size) {
	// 	printf("\n\n\nthread id passed %d", thread_id);
	// 	// Check if the current window matches the smaller matrix
	// 	int matchingCalc = 0;
	// 	for (int i = 0; i < N_size; i++) {
	// 		for (int j = 0; j < N_size; j++) {
	// 			int M_index = (window_row + i) * M_size + (window_col + j);
	// 			int N_index = i * N_size + j;
	// 			printf("\n\n\nM index %d row %d col %d N Index %d row %d col %d picture size %d object size %d", M_index, M_index / M_size, M_index % M_size, N_index, N_index / N_size, N_index % N_size, M_size, N_size);
	// 			assert(M_index < M_size * M_size);
	// 			assert(N_index < N_size * N_size);
	// 			matchingCalc += difference(M[M_index], N[N_index]);
	// 		}
	// 	}
	// 	// Store the result of the match in the output array
	// 	int index = (window_row)*M_size + (window_col);
	// 	assert(index < sizeResults * sizeResults);
	// 	matching[index] = matchingCalc / pow(sizeResults, 2);
	// 	if (matching[index] < matchingV)
	// 		positionFlags[index] = 1;
	// }
	// Compute the thread index


	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	int sizeResults = getPositionsPerDimension(M_size, N_size);
	// for (int i = 0;i < sizeResults * sizeResults;i++)
	// 	assert(positionFlags[i] == 0);
	// Compute the row and column indices of the window
	int window_row = thread_id / sizeResults;
	int window_col = thread_id % sizeResults;

	// Check if the window is within the bounds of the larger matrix
	if ((window_row + N_size) <= M_size && (window_col + N_size) <= M_size) {

		// Check if the current window matches the smaller matrix
		double matchingCalc = 0.0;
		for (int i = 0; i < N_size; i++) {
			for (int j = 0; j < N_size; j++) {
				int M_index = (window_row + i) * M_size + (window_col + j);
				int N_index = i * N_size + j;
				assert(M_index < M_size * M_size);
				assert(N_index < N_size * N_size);
				matchingCalc += difference(M[M_index], N[N_index]);
			}
		}
		// Store the result of the match in the output array
		int index = window_row * sizeResults + window_col;
		assert(index < sizeResults * sizeResults);
		assert(positionFlags[index] == 0);
		matching[index] = matchingCalc / (N_size * N_size);
		if (matching[index] < matchingV) {
			positionFlags[index] = 1;
		}
	}


}


// ======================================= Entry Point =======================================
__host__ int* searchOnGPU(int pictureDim, int* devicePictureMatrix, Matrix object, double matchingV)
{

	int positionsPerDim = pictureDim - object.size + 1, positionsCount = pow(positionsPerDim, 2);
	//	printf("\npositionsCount %d\n", positionsCount);
	int blocksPerGrid = (positionsCount + BLOCK_SIZE - 1) / BLOCK_SIZE;

	int* hostPositionFlagsArray, * devicePositionFlagsArray, * deviceObjectMatrix;
	double* deviceMatchingsArray;
	cudaError_t error = cudaSuccess;

	// Allocates memory for the position flags array
	hostPositionFlagsArray = (int*)malloc(positionsCount * sizeof(int));
	if (hostPositionFlagsArray == NULL)
	{
		printf("Cannot allocate memory for position flags array\n");
		return NULL;
	}

	// Allocates and initializes required variables on the GPU
	allocateMatOnGPU(object, &deviceObjectMatrix);

	error = cudaMalloc((void**)&devicePositionFlagsArray, positionsCount * sizeof(int));
	if (error != cudaSuccess)
	{
		printf("Cannot allocate GPU memory for position flags array: %s (%d)\n", cudaGetErrorString(error), error);
		return NULL;
	}

	error = cudaMalloc((void**)&deviceMatchingsArray, positionsCount * sizeof(double));
	// printf("\n\npositionsCount %d\n\n", positionsCount);
	if (error != cudaSuccess)
	{
		printf("Cannot allocate GPU memory for matchings array: %s (%d)\n", cudaGetErrorString(error), error);
		return NULL;
	}

	error = cudaMemset(devicePositionFlagsArray, 0, positionsCount * sizeof(int));
	if (error != cudaSuccess)
	{
		printf("Cannot initialize position flags array on GPU: %s (%d)\n", cudaGetErrorString(error), error);
		return NULL;
	}

	// if (blocksPerGrid > max_blocks_per_grid) {
	// 	// Too many blocks!
	// 	printf("\nToo many blocks!\n");
	// 	exit(EXIT_FAILURE);
	// }
	// if (threadsPerBlock > max_threads_per_block) {
	// 	// Too many threads per block!
	// 	printf("\nToo many threads per block!\n");
	// 	exit(EXIT_FAILURE);
	// }
	printf("\n\n\n m size  %d n size %d line %d\n\n", pictureDim, object.size, __LINE__);
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("CUDA error: %s line %d\n", cudaGetErrorString(error), __LINE__);
	}
	// Searches the object in the picture using CUDA - each block searches 256 positions in the picture
	sliding_window_kernel << <blocksPerGrid, BLOCK_SIZE >> > (devicePictureMatrix, pictureDim, deviceObjectMatrix, object.size, deviceMatchingsArray, devicePositionFlagsArray, matchingV);

	// findMatchingSubmatrix << <blocksPerGrid, threadsPerBlock >> > (deviceObjectMatrix, devicePictureMatrix, object.size, pictureDim, deviceMatchingsArray, devicePositionFlagsArray, matchingV);
	//  Check for errors after the kernel call
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		return NULL;
	}

	// Copies the position flags array from GPU to host
	error = cudaMemcpy(hostPositionFlagsArray, devicePositionFlagsArray, positionsCount * sizeof(int), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
	{
		printf("Cannot copy position flags from GPU to host: %s (%d)\n", cudaGetErrorString(error), error);
		return NULL;
	}

	// Frees allocated variables from the GPU
	error = cudaFree(deviceMatchingsArray);
	if (error != cudaSuccess)
	{
		printf("Cannot free matchings array from GPU: %s (%d)\n", cudaGetErrorString(error), error);
		return NULL;
	}

	error = cudaFree(devicePositionFlagsArray);
	if (error != cudaSuccess)
	{
		printf("Cannot free position flags array from GPU: %s (%d)\n", cudaGetErrorString(error), error);
		return NULL;
	}

	freeMatFromGPU(&deviceObjectMatrix);
	return hostPositionFlagsArray;
}