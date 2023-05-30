#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1000
#define P 15

void matMul(float** a, float** b, float** c) {
#pragma omp parallel for shared(a, b, c)
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
            c[i][j] = 0;
			for (int k = 0; k < N; k++) {
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}
}

int main() {
	double startTime;
	double endTime;
	float** a = (float **) malloc(N * sizeof(float *));
    float ** b = (float **) malloc(N * sizeof(float *));
    float ** c = (float **) malloc(N * sizeof(float *));

	for (int i = 0; i < N; i++) {
		b[i] = (float *) malloc(N * sizeof(float));
		a[i] = (float *) malloc(N * sizeof(float));
		c[i] = (float *) malloc(N * sizeof(float));
        for (int j = 0; j < N; j++) {
			a[i][j] = rand();
			b[i][j] = rand();
		}
	}

	for (int i = 1; i <= P; i++) {
		startTime = omp_get_wtime();
		omp_set_num_threads(i);
		matMul(a, b, c);
		endTime = omp_get_wtime();
		printf("Thread # is %d, time: %f\n", i, endTime - startTime);
	}
}
