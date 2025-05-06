#include "utils.hpp"

float random(float low, float high) {
    std::random_device rd;  
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(low, high);
    return dis(gen);
}

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

int chosenDigit(float outputs[10]) {
    int digit = -1;
    float maxx = -100;
    for (int i = 0; i < 10; ++i) {
        if (maxx < outputs[i]) {
            maxx = outputs[i];
            digit = i;
        }
    }
    return digit;
}
