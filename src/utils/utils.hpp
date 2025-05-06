#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <random>
using namespace std;

const int BATCH_SIZE = 20;

float random(float low, float high);
float sigmoid(float x);
int chosenDigit(float outputs[10]);

#endif 