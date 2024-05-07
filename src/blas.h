#ifndef BLAS_H
#define BLAS_H

#include "reform-c.h"

void flatten(float *x, int size, int layers, int batch, int forward);

void pm(int M, int N, float *A);

float *random_matrix(int rows, int cols);



#endif // BLAS_H