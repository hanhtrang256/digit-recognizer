#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <iostream>
#include "matrix.hpp"
#include "../utils/utils.hpp"
using namespace std; 

struct NeuralNetwork {
    int numInput, numHidden, numOutput;
    Matrix IH, HO, BH, BO;
    Matrix outputHidden, outputOutput;
    float LEARNING_RATE;

    NeuralNetwork(int _numInput, int _numHidden, int _numOutput);
    NeuralNetwork(const NeuralNetwork &other);
    void SGD_feedForward(float inputs[], float outputs[]);
    void SGD_train(float inputs[], float targets[]);
    void MBGD_feedForward(const vector<vector<float>> &inputs, vector<vector<float>> &outputs);
    void MBGD_train(const vector<vector<float>> &inputs, const vector<vector<float>> &targets);
    void mutate(float rate);
};

#endif 