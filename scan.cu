//=============================================================================
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h> 
#include <thrust/device_vector.h> 
#include <iostream> 
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <cub.cuh> 
#include <cuda_profiler_api.h>
//=============================================================================
#define		N								4
#define		SIZE_IMG				N*N
//=============================================================================

void initialize(size_t m, size_t n, unsigned int *d1)
{
	for (size_t i = 0; i < m; ++i)
	{
		for (size_t j = 0; j < n; ++j)
		{
			d1[i * n + j] = (i * n + j);
		}
	}
}

void print(size_t m, size_t n, unsigned int* h_data)
{
	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
			std::cout << std::setw(8) << h_data[i * n + j] << " ";
		std::cout << "\n";
	}
	printf("\n");
}
int main()
{
	unsigned int *dVec_I1, *dVec_I2;
	unsigned int hVec_I1[SIZE_IMG], hVec_I3[SIZE_IMG];// , hVec_I3[SIZE_IMG], hVec_I4[SIZE_IMG]; //3 and 4 are expendables
	cudaMalloc(&dVec_I1, SIZE_IMG * sizeof(unsigned int));
	cudaMalloc(&dVec_I2, SIZE_IMG * sizeof(unsigned int));

	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;
	initialize(N, N, hVec_I1);
	cudaMemcpy(dVec_I1, hVec_I1, SIZE_IMG * sizeof(unsigned int), cudaMemcpyHostToDevice);

	summed_area_table_rowwise1:
		for (int i = 0; i < SIZE_IMG; i += N)
		{
			// Perform Inclusive sum for Vector 1
			cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, dVec_I1 + i, dVec_I1 + i, N);
			// Allocate temporary storage
			cudaMalloc(&d_temp_storage, temp_storage_bytes);
			cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, dVec_I1 + i, dVec_I1 + i, N);
			cudaMemcpy(hVec_I3, dVec_I1, SIZE_IMG * sizeof(unsigned int), cudaMemcpyDeviceToHost);// For verification
			cudaFree(d_temp_storage);
			d_temp_storage = NULL;
			temp_storage_bytes = 0;
		}
	print(N, N, hVec_I3);
	getchar();
	return 0;
}
