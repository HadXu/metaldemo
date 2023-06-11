#include <iostream>
#include <random>

#include "libmetal.h"

int main() {
    struct metal_context *ctx = metal_init();


    float inputA[1024];
    float inputB[1024];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0, 1);

    for (size_t i = 0; i < 1024; ++i) {
        inputA[i] = dis(gen);
        inputB[i] = dis(gen);
    }


    uint32_t size = sizeof(inputA) / sizeof(float);
    float output[size];

    vector_add(ctx, inputA, inputB, output, size);

    std::cout << "Result: ";
    for (uint32_t i = 0; i < size; ++i) {
        std::cout << output[i] << " ";
    }

    std::cout << "\nHello, World!" << std::endl;
    return 0;
}
