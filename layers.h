#ifndef LAYERS_H
#define LAYERS_H

#include "const.h"

void conv1(float input[28][28], float output[20][24][24]);
void bn1(float input[20][24][24], float output[20][24][24]);
void relu1(float input[20][24][24], float output[20][24][24]);
void pool1(float input[20][24][24], float output[20][12][12]);
// binconv2
void bn2(float input[20][12][12], float output[20][12][12]);
void conv2(int input[20][12][12], float output[50][8][8]);
void relu2(float input[50][8][8], float output[50][8][8]);
void pool2(float input[50][8][8], float output[50][4][4]);
// binip1
void bn3(float input[50][4][4], float output[50][4][4]);
void ip1(int input[800], float output[500]);
void relu3(float input[500], float output[500]);
void ip2(float input[500], float output[10]);

#endif









































