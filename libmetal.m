#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "libmetal.h"


#define metal_printf(...) fprintf(stderr, __VA_ARGS__)


struct metal_buffer {
    const char *name;
    void *data;
    size_t size;
    id <MTLBuffer> metal;
};

struct metal_context {
    float *logits;
    id <MTLDevice> device;
    id <MTLCommandQueue> queue;
    id <MTLLibrary> library;
    int n_buffers;
    struct metal_buffer buffers[100];

#define METAL_DECL_KERNEL(name) \
    id<MTLFunction>             function_##name; \
    id<MTLComputePipelineState> pipeline_##name

    METAL_DECL_KERNEL(add);
#undef METAL_DECL_KERNEL
};

@interface GGMLMetalClass : NSObject
@end

@implementation GGMLMetalClass
@end

struct metal_context *metal_init(void) {
    struct metal_context *ctx = malloc(sizeof(struct metal_context));

    ctx->device = MTLCreateSystemDefaultDevice();
    ctx->queue = [ctx->device newCommandQueue];
    if (MPSSupportsMTLDevice(ctx->device)) {
        fprintf(stderr, "%s: using MPS\n", __func__);
    } else {
        fprintf(stderr, "%s: not using MPS\n", __func__);
    }

    NSError *error = nil;

    NSBundle *bundle = [NSBundle bundleForClass:[GGMLMetalClass class]];
    NSString *path = [bundle pathForResource:@"libmetal" ofType:@"metal"];
    fprintf(stderr, "%s: loading '%s'\n", __func__, [path UTF8String]);

    NSString *src = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
    if (error) {
        fprintf(stderr, "%s: error: %s\n", __func__, [[error description] UTF8String]);
        exit(1);
    }

    ctx->library = [ctx->device newLibraryWithSource:src options:nil error:&error];
    if (error) {
        fprintf(stderr, "%s: error: %s\n", __func__, [[error description] UTF8String]);
        exit(1);
    }

    {
#define METAL_ADD_KERNEL(name) \
    ctx->function_##name = [ctx->library newFunctionWithName:@"kernel_"#name]; \
    ctx->pipeline_##name = [ctx->device newComputePipelineStateWithFunction:ctx->function_##name error:nil]; \
    fprintf(stderr, "%s: loaded %-32s %16p\n", __func__, "kernel_"#name, (void *) ctx->pipeline_##name);
        METAL_ADD_KERNEL(add);
#undef METAL_ADD_KERNEL
    }

    return ctx;
}

void metal_free(struct metal_context *ctx) {
    fprintf(stderr, "%s: deallocating\n", __func__);
    free(ctx);
}


static id <MTLBuffer> metal_get_buffer(struct metal_context *ctx, struct tensor *t, size_t *offs) {
    for (int i = 0; i < ctx->n_buffers; ++i) {
        const int64_t ioffs = (int64_t) t->data - (int64_t) ctx->buffers[i].data;
        if (ioffs >= 0 && ioffs < (int64_t) ctx->buffers[i].size) {
            *offs = (size_t) ioffs;
            return ctx->buffers[i].metal;
        }
    }
    fprintf(stderr, "%s: error: buffer is nil\n", __func__);
    return nil;
}

void metal_set_tensor(struct metal_context *ctx, struct tensor *t) {
    metal_printf("%s: set input for tensor\n", __func__);

    size_t offs;
    id <MTLBuffer> id_dst = metal_get_buffer(ctx, t, &offs);
    memcpy((void *) ((uint8_t *) id_dst.contents + offs), t->data, t->size);
}


void metal_get_tensor(struct metal_context *ctx, struct tensor *t) {
    metal_printf("%s: extract results for tensor\n", __func__);
    size_t offs;
    id <MTLBuffer> id_src = metal_get_buffer(ctx, t, &offs);
    memcpy(t->data, (void *) ((uint8_t *) id_src.contents + offs), t->size);
}


void vector_add(struct metal_context *ctx, const float *inputA, const float *inputB, float *output, uint32_t size) {
    metal_printf("%s: evaluating graph\n", __func__);

    size_t offs_src0 = 0;
    size_t offs_src1 = 0;
    size_t offs_dst = 0;

    id <MTLCommandBuffer> command_buffer = [ctx->queue commandBuffer];
    id <MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

    id <MTLBuffer> id_src0 = [ctx->device newBufferWithBytes:inputA length:size * sizeof(float) options:0];
    id <MTLBuffer> id_src1 = [ctx->device newBufferWithBytes:inputB length:size * sizeof(float) options:0];
    id <MTLBuffer> id_dst = [ctx->device newBufferWithBytes:output length:size * sizeof(float) options:0];

    if (encoder == nil) {
        encoder = [command_buffer computeCommandEncoder];
    }

    [encoder setComputePipelineState:ctx->pipeline_add];
    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
    [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
    [encoder setBuffer:id_dst offset:offs_dst atIndex:2];

    [encoder dispatchThreadgroups:MTLSizeMake(size, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

    [encoder endEncoding];

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    memcpy(output, id_dst.contents, size * sizeof(float));
}













