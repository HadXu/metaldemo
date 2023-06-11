//
// Created by lei on 2023/6/10.
//

#ifndef METALDEMO_LIBMETAL_H
#define METALDEMO_LIBMETAL_H

#ifdef __cplusplus
extern "C" {
#endif


struct tensor {
    void *data;
    size_t size;
};


struct metal_context *metal_init(void);

void metal_free(struct metal_context *ctx);

void vector_add(struct metal_context *ctx, const float *inputA, const float *inputB, float *output, uint32_t size);

#ifdef __cplusplus
}
#endif

#endif //METALDEMO_LIBMETAL_H
