#ifndef REFORM_C_API
#define REFORM_C_API

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

#if defined(WIN32) || defined(_WIN32)
#ifdef BUILDING_REFORM_SHARED
#define REFORM_C_EXPORT __declspec(dllexport)
#elif USING_REFORM_SHARED
#define REFORM_C_EXPORT __declspec(dllimport)
#else
#define REFORM_C_EXPORT
#endif
#else
#define REFORM_C_EXPORT
#endif

#ifdef GPU
#define BLOCK 512

#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>

#ifdef CUDNN
#include <cudnn.h>
#endif // CUDNN
#endif // GPU

#ifdef __cplusplus
extern "C" {
#endif

#define SECRET_NUM -1234
extern int gpu_index;

typedef struct {
    int classes;
    char **names;
} metadata;

metadata get_metadata(char *file);

typedef struct {
    int *leaf;
    int n;
    int *parent;
    int *child;
    int *group;
    char **name;

    int groups;
    int *group_size;
    int *group_offset;
} tree;

tree *read_tree(char *filename);

typedef enum {
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN, SELU
} ACTIVATION;

typedef enum {
    PNG, BMP, TGA, JPG
} IMTYPE;

typedef enum {
    MULT, ADD, SUB, DIV
} BINARY_ACTIVATION;

typedef enum {
    CONVOLUTIONAL,
    DECONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX,
    DETECTION,
    DROPOUT,
    CROP,
    ROUTE,
    COST,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    SHORTCUT,
    ACTIVE,
    RNN,
    GRU,
    LSTM,
    CRNN,
    BATCHNORM,
    NETWORK,
    XNOR,
    REGION,
    YOLO,
    ISEG,
    REORG,
    UPSAMPLE,
    LOGXENT,
    L2NORM,
    BLANK
} LAYER_TYPE;

typedef struct {
    SSE, MASKED, L1, SEG, SMOOTH, WGAN
} COST_TYPE;

typedef struct {
    int batch;
    float learning_rate;
    float momentum;
    float decay;
    int adam;
    float B1;
    float B2;
    float eps;
    int t;
} update_args;

struct network;
typedef struct network network;

struct layer;
typedef struct layer layer;

struct layer {
    LAYER_TYPE type;
    ACTIVATION activation;
    COST_TYPE cost_type;
    void (*forward) (struct layer, struct network);
    void (*backward) (struct layer, struct network);
    void (*update) (struct layer, update_args);
    void (*forward_gpu) (struct layer, struct network);
    void (*backward_gpu) (struct layer, struct network);
    void (*update_gpu) (struct layer, update_args);

    int batch_normalize;
    int shortcut;
    int batch;
    int forced;
    int flipped;
    int inputs;
    int outputs;
    int nweights;
    int nbiases;
    int extra;
    int truths;
    int h, w, c;
    int out_h, out_w, out_c;
    int n;
    int max_boxes;
    int groups;
    int size;
    int side;
    int stride;
    int reverse;
    int flatten;
    int spatial;
    int pad;
    int sqrt;
    int flip;
    int index;
    int binary;

#ifdef GPU
    int *indexes_gpu;

    float *z_gpu;
    float *r_gpu;
    float *h_gpu;

    float *temp_gpu;
    float *temp2_gpu;
    float *temp3_gpu;

    float *dh_gpu;

#ifdef CUDNN
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
#endif
#endif
};

void free_layer(layer);

typedef enum {
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
} learning_rate_policy;

typedef struct network {
    int n;
    int batch;
    size_t *seen;
    int *t;
};

#ifdef __cplusplus
}
#endif


#endif // REFORM_C_API