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
#include <cuda_profiler_api.h>
//=============================================================================
#define		N								4
#define		SIZE_IMG				N*N
//=============================================================================




// Initialize array with a synthetic input image
void initialize(size_t m, size_t n, int* d1)
{
	for (size_t i = 0; i < m; ++i)
	{
		for (size_t j = 0; j < n; ++j)
		{
			d1[i * n + j] = (i * n + j);
		}
	}
}

// print an M-by-N device_vector

void print(size_t m, size_t n, int* h_data)
{
	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
			std::cout << std::setw(8) << h_data[i * n + j] << " ";
		std::cout << "\n";
	}
	printf("\n");
}

// convert a linear index to a row index
struct row_index : public thrust::unary_function<size_t, size_t>
{
	size_t n;

	__host__ __device__
		row_index(size_t _n) : n(_n) {}

	__host__ __device__
		size_t operator()(size_t i)
	{
			return i / n;
		}
};

int main(void)
{
	// Array stored in row-major order [(0,0), (0,1), (0,2), ... ]
	int hVec_I1[SIZE_IMG];
	int *dVec_I1;
	cudaMalloc(&dVec_I1, sizeof(int)* SIZE_IMG);
	thrust::counting_iterator<int> indices(0);
	// Innitialize vector with synthetic image:
	initialize(N, N, hVec_I1);
	cudaMemcpy(dVec_I1, hVec_I1, SIZE_IMG*sizeof(int), cudaMemcpyHostToDevice);
	//print(N, N, hVec_I1);
	cudaProfilerStart();
	// Scan:
	thrust::device_ptr<int> dVec_M1 = thrust::device_pointer_cast(dVec_I1);
	thrust::inclusive_scan_by_key
		(thrust::make_transform_iterator(indices, row_index(N)),
		thrust::make_transform_iterator(indices, row_index(N)) + SIZE_IMG,
		dVec_M1,
		dVec_M1);
	cudaProfilerStop();
	cudaMemcpy(hVec_I1, dVec_I1, SIZE_IMG*sizeof(int), cudaMemcpyDeviceToHost);
	// Print result:
	print(N, N, hVec_I1);

	//getchar();
	return 0;
}
