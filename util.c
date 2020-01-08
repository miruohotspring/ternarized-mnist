#include "util.h"
#include <stdio.h>

int argmax(float input[10]) {
    float max = input[0];
    int arg = 0;
    for (int i = 1; i < 10; i++) {
        if (max < input[i]) {
            max = input[i];
            arg = i;
        }
    }
    return arg;
}

void ternarize1(float input[20][12][12], int output[20][12][12]) {
    for (int c = 0; c < 20; c++) {
        for (int w = 0; w < 12; w++) {
            for (int h = 0; h < 12; h++) {
                if (input[c][w][h] > 0.5) {
                    output[c][w][h] = 1;
                }
                else if (input[c][w][h] < -0.5) {
                    output[c][w][h] = -1;
                }
                else {
                    output[c][w][h] = 0;
                }
            }
        }
    }
}

void ternarize2(float input[50][4][4], int output[800]) {
    int i = 0;
    for (int c = 0; c < 50; c++) {
        for (int w = 0; w < 4; w++) {
            for (int h = 0; h < 4; h++) {
                if (input[c][w][h] > 0.5) {
                    output[i] = 1;
                }
                else if (input[c][w][h] < -0.5) {
                    output[i] = -1;
                }
                else {
                    output[i] = 0;
                }
                i++;
            }
        }
    }
    /*
    int count = 0;
    for(int i = 0; i < 800; i++) {
        count += output[i];
    }
    printf("%d\n", count);
    */
}


void normalize(float input[28][28], float output[28][28]) {
    for (int w = 0; w < 28; w++) {
        for (int h = 0; h < 28; h++) {
            output[w][h] = (input[w][h] - 0.1307) / 0.3081;
        }
    }
}
        
    



































