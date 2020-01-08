#ifndef UTIL_H
#define UTIL_H

int argmax(float input[10]);
void ternarize1(float input[20][12][12], int output[20][12][12]);
void ternarize2(float input[50][4][4], int output[800]);
void normalize(float input[28][28], float output[28][28]);

#endif
