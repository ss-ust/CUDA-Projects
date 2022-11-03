// Program to calculate vectors with CUDA and comparing speed difference between CPU and GPU.

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Function for GPU calculation
__global__ void vector_addition(int* a, int* b, int* c) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	c[tid] = a[tid] + b[tid];
}


// Function for filling the arrays with numbers between 0-32767
void fill_array(int* data) {
	for (int i = 0; i < 32768; i++) {
		data[i] = i;
	}
}

// CPU calculation and result print function
void calculate_print(int* a, int* b, int* c) {
	printf("#### CPU ####\n");
	for (int i = 0; i < 32768; i++) {
		c[i] = a[i] + b[i];
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}
	printf("------ CPU COMPLETE ------ \n Press 'ENTER' to continue...\n");
}

// GPU result print function
void print_array(int* a, int* b, int* c) {
	printf("\n#### GPU ####\n");
	for (int i = 0; i < 32768; i++) {
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}
	printf("------ GPU COMPLETE ------");
}

int main() {

	clock_t t_gpu, t_cpu;

	int a[32768], b[32768], c[32768];
	int h_a[32768], h_b[32768], h_c[32768];
	int* d_a, * d_b, * d_c;

	size_t size = sizeof(int) * 32768;

	fill_array(a);
	fill_array(b);

	fill_array(h_a);
	fill_array(h_b);

	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);
	cudaMalloc(&d_c, size);

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	t_cpu = clock();
	calculate_print(h_a, h_b, h_c);
	t_cpu = clock() - t_cpu;
	double time_taken_cpu = ((double)t_cpu) / CLOCKS_PER_SEC;

	getchar();

	t_gpu = clock();
	vector_addition << <32768, 1 >> > (d_a, d_b, d_c);

	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	print_array(a, b, c);
	t_gpu = clock() - t_gpu;
	double time_taken_gpu = ((double)t_gpu) / CLOCKS_PER_SEC;

	printf("\n ##### STATISTICS #####");
	printf("\n - CPU took %lf seconds to complete.\n", time_taken_cpu);
	printf(" - GPU took %lf seconds to complete.\n", time_taken_gpu);

	if (time_taken_cpu > time_taken_gpu) {
		double speed = time_taken_cpu / time_taken_gpu;
		printf(" - GPU is %lf times faster than the CPU.", speed);
	}
	else {
		double speed = time_taken_gpu / time_taken_cpu;
		printf(" - CPU is %lf times faster than the GPU.", speed);
	}

	return 0;
}