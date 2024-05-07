#ifndef CUDA_H
#define CUDA_H

#ifdef GPU

void check_error(cudaError_t status);

#endif // GPU
#endif // CUDA_H