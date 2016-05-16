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
#define		N								512
#define		NUM_BLOCKS			128	
#define		BLOCK_SIZE			16
#define		BLOCK_SQUARE		BLOCK_SIZE*BLOCK_SIZE
#define		SHIFT						4
#define		SIZE_IMG				N*N
#define		SIZE_IMG_BYTES	sizeof(int)*SIZE_IMG
		 //=============================================================================


void initialize(size_t m, size_t n, unsigned int *d1, unsigned int* d2)
{
	for (size_t i = 0; i < m; ++i)
	{
		for (size_t j = 0; j < n; ++j)
		{
			d1[i * n + j] = (i * n + j) % 200;
			//d1[i * n + j] = (i * n + j);
			d2[i * n + j] = d1[i * n + j] * d1[i * n + j];
		}
	}
}






// to be launched with one thread per row of output matrix
__global__ void transpose_parallel_per_row(unsigned int  *out, unsigned int *in)
{
	int i = threadIdx.x;

	for (int j = 0; j < N; j++)
		out[j + i*N] = in[i + j*N]; // out(j,i) = in(i,j)
}

/*
__global__ void tiled_kernel(KernelArray<int> in, KernelArray<int> out)
{
	const int x = threadIdx.x;
	const int y = blockIdx.x;

	const int pos = (y / NUM_BLOCKS * 4) * N + (y % (NUM_BLOCKS)) * 4;

	//out._array[y * BLOCK_SQUARE + x] = in._array[pos + (x % BLOCK_SIZE) + ((x / 16) * N)];
	out._array[y * BLOCK_SQUARE + x] = pos + (x % BLOCK_SIZE) + ((x / 16) * N);
}
*/
int main(void)
{
	unsigned int *dVec_I1, *dVec_I2;
	unsigned int* temp1, *temp2;
	unsigned int hVec_I1[SIZE_IMG], hVec_I2[SIZE_IMG], hVec_I3[SIZE_IMG], hVec_I4[SIZE_IMG]; //3 and 4 are expendables
	cudaMalloc(&dVec_I1, SIZE_IMG * sizeof(unsigned int));
	cudaMalloc(&dVec_I2, SIZE_IMG * sizeof(unsigned int));
	cudaMalloc(&temp1, SIZE_IMG * sizeof(unsigned int));
	cudaMalloc(&temp2, SIZE_IMG * sizeof(unsigned int));

	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;
	initialize(N, N, hVec_I1, hVec_I2);
	cudaMemcpy(dVec_I1, hVec_I1, SIZE_IMG * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(dVec_I2, hVec_I2, SIZE_IMG * sizeof(unsigned int), cudaMemcpyHostToDevice);

	transpose_parallel_per_row << < 1, N >> >(temp1, dVec_I1);
	transpose_parallel_per_row << < 1, N >> >(temp2, dVec_I2);
	cudaMemcpy(hVec_I3, temp1, SIZE_IMG * sizeof(unsigned int), cudaMemcpyDeviceToHost); //For verification
	cudaMemcpy(hVec_I4, temp2, SIZE_IMG * sizeof(unsigned int), cudaMemcpyDeviceToHost); //For verification

	summed_area_table_rowwise1:
		for (int i = 0; i < SIZE_IMG; i += N)
		{
			//Input and output data are in hVec_I1.
			// Perform Inclusive sum for Vector 1
			cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, temp1 + i, temp1 + i, N);
			// Allocate temporary storage
			cudaMalloc(&d_temp_storage, temp_storage_bytes);
			cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, temp1 + i, temp1 + i, N);
			cudaMemcpy(hVec_I3, temp1, SIZE_IMG * sizeof(unsigned int), cudaMemcpyDeviceToHost);// For verification
			cudaFree(d_temp_storage);
			d_temp_storage = NULL;
			temp_storage_bytes = 0;

			cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, temp2 + i, temp2 + i, N);
			// Allocate temporary storage
			cudaMalloc(&d_temp_storage, temp_storage_bytes);
			cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, temp2 + i, temp2 + i, N);
			cudaMemcpy(hVec_I4, temp2, SIZE_IMG * sizeof(unsigned int), cudaMemcpyDeviceToHost);// For verification
			cudaFree(d_temp_storage);
			d_temp_storage = NULL;
			temp_storage_bytes = 0;
		}
	transpose_parallel_per_row << < 1, N >> >(dVec_I1, temp1);   //Super reuse of device vectors here
	transpose_parallel_per_row << < 1, N >> >(dVec_I2, temp2);   //Super reuse of device vectors here

	cudaMemcpy(hVec_I3, dVec_I1, SIZE_IMG * sizeof(unsigned int), cudaMemcpyDeviceToHost);  //For verification
	cudaMemcpy(hVec_I4, dVec_I2, SIZE_IMG * sizeof(unsigned int), cudaMemcpyDeviceToHost);  //For verification

	summed_area_table_rowwise2:
		for (int i = 0; i < SIZE_IMG; i += N)
		{
			//Input and output data are in hVec_I1.
			// Perform Inclusive sum for Vector 1
			cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, dVec_I1 + i, dVec_I1 + i, N);
			// Allocate temporary storage
			cudaMalloc(&d_temp_storage, temp_storage_bytes);
			cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, dVec_I1 + i, dVec_I1 + i, N);
			cudaMemcpy(hVec_I3, dVec_I1, SIZE_IMG * sizeof(unsigned int), cudaMemcpyDeviceToHost);// For verification
			cudaFree(d_temp_storage);
			d_temp_storage = NULL;
			temp_storage_bytes = 0;

			cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, dVec_I2 + i, dVec_I2 + i, N);
			// Allocate temporary storage
			cudaMalloc(&d_temp_storage, temp_storage_bytes);
			cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, dVec_I2 + i, dVec_I2 + i, N);
			cudaMemcpy(hVec_I4, dVec_I2, SIZE_IMG * sizeof(unsigned int), cudaMemcpyDeviceToHost);// For verification
			cudaFree(d_temp_storage);
			d_temp_storage = NULL;
			temp_storage_bytes = 0;
		}
	// End of operations. Copying dVec_M1 and dVec_M2 into new host memory
	// Compute stats for each tile via integral image technique
	int s1 = 0, s2 = 0;
	size_t m1 = 0, n1 = 0;
	size_t m2 = 15, n2 = 15;
	while (m2 < N)
	{
		n1 = 0;
		n2 = 15;
		while (n2 < N)
		{
			//printf("(m1,n1) = (%d,%d)\n", m1, n1);
			//printf("(m2,n2) = (%d,%d)\n", m2, n2);
			int sum = 0;
			for (int i = m1; i <= m2; ++i)
			{
				for (int j = n1; j <= n2; ++j)
					sum += hVec_I1[i*N + j];
			}
			float mean = sum / 256.0f;
			float golden = 0;

			int cc_count = 0;
			for (int i = m1; i <= m2; ++i)
			{
				for (int j = n1; j <= n2; ++j)
					golden += (hVec_I1[i*N + j] - mean) * (hVec_I1[i*N + j] - mean);
			}
			golden = sqrt((golden) / (255.0f));
			//std::cout << "(m1,n1) -> " << m1*N + n1 << " => " << dVec_M1[m1*N + n1] << std::endl;
			//std::cout << "(m1,n2) -> " << m1*N + n2 << " => " << dVec_M1[m1*N + n2] << std::endl;
			//std::cout << "(m2,n1) -> " << m2*N + n1 << " => " << dVec_M1[m2*N + n1] << std::endl;
			//std::cout << "(m2,n2) -> " << m2*N + n2 << " => " << dVec_M1[m2*N + n2] << std::endl;
			//getchar();

			// Account for boundary conditions:
			int A1 = 0, B1 = 0, C1 = 0, D1 = 0;
			int A2 = 0, B2 = 0, C2 = 0, D2 = 0;
			if ((m1 == 0) && (n1 == 0))
			{
				A1 = 0;	A2 = 0;
				B1 = 0;	B2 = 0;
				C1 = 0;	C2 = 0;
				D1 = hVec_I3[m2*N + n2];
				D2 = hVec_I4[m2*N + n2];
			}
			else if ((m1 == 0) && (n1 > 0))
			{
				A1 = 0;	A2 = 0;
				B1 = 0;	B2 = 0;
				C1 = hVec_I3[m2*N + n1 - 1];
				C2 = hVec_I4[m2*N + n1 - 1];
				D1 = hVec_I3[m2*N + n2];
				D2 = hVec_I4[m2*N + n2];
			}
			else if ((m1 > 0) && (n1 == 0))
			{
				A1 = 0;	A2 = 0;
				B1 = hVec_I3[(m1 - 1)*N + n2];
				B2 = hVec_I4[(m1 - 1)*N + n2];
				C1 = 0;	C2 = 0;
				D1 = hVec_I3[m2*N + n2];
				D2 = hVec_I4[m2*N + n2];
			}
			else
			{
				A1 = hVec_I3[(m1 - 1)*N + n1 - 1];
				A2 = hVec_I4[(m1 - 1)*N + n1 - 1];
				B1 = hVec_I3[(m1 - 1)*N + n2];
				B2 = hVec_I4[(m1 - 1)*N + n2];
				C1 = hVec_I3[m2*N + n1 - 1];
				C2 = hVec_I4[m2*N + n1 - 1];
				D1 = hVec_I3[m2*N + n2];
				D2 = hVec_I4[m2*N + n2];
			}

			s1 = A1 - B1 - C1 + D1;
			s2 = A2 - B2 - C2 + D2;

			float TileSize = 256.0f;
			float coef = (1 / ((float)TileSize - 1));
			float sigma = sqrt(coef * (s2 - s1*s1 / (float)TileSize));

			double error = abs(sigma - golden);
			if (error > 1e-3)
			{
				printf("ERROR -- standard deviation is wrong\n");
				getchar();
			}

			n1 += 4;
			n2 = n1 + BLOCK_SIZE - 1;
		}
		m1 += 4;
		m2 = m1 + BLOCK_SIZE - 1;
	}
	printf("end of execution\n");
	getchar();
	return 0;
}
