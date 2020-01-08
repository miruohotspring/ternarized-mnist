#include "layers.h"
#include "params.h"
#include "math.h"
#include <stdio.h>

void conv1(float input[28][28], float output[20][24][24]){
    for (int c = 0; c < 20; c++) {
        for (int h = 0; h < 24; h++) {
            for (int w = 0; w < 24; w++) {
                for (int hf = 0; hf < 5; hf++) {
                    for (int wf = 0; wf < 5; wf++) {
                        output[c][h][w] += input[h+hf][w+wf] * conv1_weight[c][0][hf][wf];
                    }
                }
                output[c][h][w] += conv1_bias[c];
                
            }
        }
    }
}

void bn1(float input[20][24][24], float output[20][24][24]){
    for (int c = 0; c < 20; c++) {
        for (int h = 0; h < 24; h++) {
            for (int w = 0; w < 24; w++) {
                output[c][h][w] = (input[c][h][w] - bnconv1_runningmean[c])\
                                    /(sqrt(bnconv1_runningvar[c]) + 0.0001);
            }
        }
    }
}

void relu1(float input[20][24][24], float output[20][24][24]) {
    for (int c = 0; c < 20; c++) {
        for (int h = 0; h < 24; h++) {
            for (int w = 0; w < 24; w++) {
                output[c][h][w] = fmaxf(0., input[c][h][w]);
            }
        }
    }
}

void pool1(float input[20][24][24], float output[20][12][12]) {
    for (int c = 0; c < 20; c++) {
        for (int h = 0; h < 12; h++) {
            for (int w = 0; w < 12; w++) {
                float max = input[c][h*2][w*2];
                if (input[c][h*2][w*2+1] > max) {
                    max = input[c][h*2][w*2+1];
                }
                if (input[c][h*2+1][w*2] > max) {
                    max = input[c][h*2+1][w*2];
                }
                if (input[c][h*2+1][w*2+1] > max) {
                    max = input[c][h*2+1][w*2+1];
                }
                output[c][h][w] = max;
            }
        }
    }
}

void bn2(float input[20][12][12], float output[20][12][12]){
    for (int c = 0; c < 20; c++) {
        for (int h = 0; h < 12; h++) {
            for (int w = 0; w < 12; w++) {
                output[c][h][w] = ((input[c][h][w] - binconv2_bn_runningmean[c])\
                                    / (sqrt(binconv2_bn_runningvar[c]) + 0.00001))\
                                    * binconv2_bn_weight[c] + binconv2_bn_bias[c];
            }
        }
    }
}

// 畳み込みフィルタbinconv2_weight[50][20][5][5]を使って
// input[20][12][12] -> output[50][8][8]
void conv2(int input[20][12][12], float output[50][8][8]){
    for (int cout = 0; cout < 50; cout++) {
        for (int h = 0; h < 8; h++) {
            for (int w = 0; w < 8; w++) {
                for (int cin = 0; cin < 20; cin++) {
                    // 25行に展開（した）
                    output[cout][h][w] += input[cin][h+0][w+0] * binconv2_conv_weight[cout][cin][0][0];
                    output[cout][h][w] += input[cin][h+0][w+1] * binconv2_conv_weight[cout][cin][0][1];
                    output[cout][h][w] += input[cin][h+0][w+2] * binconv2_conv_weight[cout][cin][0][2];
                    output[cout][h][w] += input[cin][h+0][w+3] * binconv2_conv_weight[cout][cin][0][3];
                    output[cout][h][w] += input[cin][h+0][w+4] * binconv2_conv_weight[cout][cin][0][4];
                    output[cout][h][w] += input[cin][h+1][w+0] * binconv2_conv_weight[cout][cin][1][0];
                    output[cout][h][w] += input[cin][h+1][w+1] * binconv2_conv_weight[cout][cin][1][1];
                    output[cout][h][w] += input[cin][h+1][w+2] * binconv2_conv_weight[cout][cin][1][2];
                    output[cout][h][w] += input[cin][h+1][w+3] * binconv2_conv_weight[cout][cin][1][3];
                    output[cout][h][w] += input[cin][h+1][w+4] * binconv2_conv_weight[cout][cin][1][4];
                    output[cout][h][w] += input[cin][h+2][w+0] * binconv2_conv_weight[cout][cin][2][0];
                    output[cout][h][w] += input[cin][h+2][w+1] * binconv2_conv_weight[cout][cin][2][1];
                    output[cout][h][w] += input[cin][h+2][w+2] * binconv2_conv_weight[cout][cin][2][2];
                    output[cout][h][w] += input[cin][h+2][w+3] * binconv2_conv_weight[cout][cin][2][3];
                    output[cout][h][w] += input[cin][h+2][w+4] * binconv2_conv_weight[cout][cin][2][4];
                    output[cout][h][w] += input[cin][h+3][w+0] * binconv2_conv_weight[cout][cin][3][0];
                    output[cout][h][w] += input[cin][h+3][w+1] * binconv2_conv_weight[cout][cin][3][1];
                    output[cout][h][w] += input[cin][h+3][w+2] * binconv2_conv_weight[cout][cin][3][2];
                    output[cout][h][w] += input[cin][h+3][w+3] * binconv2_conv_weight[cout][cin][3][3];
                    output[cout][h][w] += input[cin][h+3][w+4] * binconv2_conv_weight[cout][cin][3][4];
                    output[cout][h][w] += input[cin][h+4][w+0] * binconv2_conv_weight[cout][cin][4][0];
                    output[cout][h][w] += input[cin][h+4][w+1] * binconv2_conv_weight[cout][cin][4][1];
                    output[cout][h][w] += input[cin][h+4][w+2] * binconv2_conv_weight[cout][cin][4][2];
                    output[cout][h][w] += input[cin][h+4][w+3] * binconv2_conv_weight[cout][cin][4][3];
                    output[cout][h][w] += input[cin][h+4][w+4] * binconv2_conv_weight[cout][cin][4][4];
                }
                // alphaとbiasの処理
                output[cout][h][w] = output[cout][h][w] \
                            * binconv2_alpha[cout] + binconv2_conv_bias[cout];
            }
        }
    }
}

void relu2(float input[50][8][8], float output[50][8][8]) {
    for (int c = 0; c < 50; c++) {
        for (int h = 0; h < 8; h++) {
            for (int w = 0; w < 8; w++) {
                output[c][h][w] = fmaxf(0., input[c][h][w]);
            }
        }
    }
}

void pool2(float input[50][8][8], float output[50][4][4]) {
    for (int c = 0; c < 50; c++) {
        for (int h = 0; h < 4; h++) {
            for (int w = 0; w < 4; w++) {
                float max = input[c][h*2][w*2];
                if (input[c][h*2][w*2+1] > max) {
                    max = input[c][h*2][w*2+1];
                }
                if (input[c][h*2+1][w*2] > max) {
                    max = input[c][h*2+1][w*2];
                }
                if (input[c][h*2+1][w*2+1] > max) {
                    max = input[c][h*2+1][w*2+1];
                }
                output[c][h][w] = max;
            }
        }
    }
}

void bn3(float input[50][4][4], float output[50][4][4]){
    for (int c = 0; c < 50; c++) {
        for (int h = 0; h < 4; h++) {
            for (int w = 0; w < 4; w++) {
                output[c][h][w] = ((input[c][h][w] - binip1_bn_runningmean[c])\
                                    /(sqrt(binip1_bn_runningvar[c]) + 0.00001))\
                                    * binip1_bn_weight[c] + binip1_bn_bias[c];
            }
        }
    }
}

// 3値化線形結合層
void ip1(int input[800], float output[500]) {
    for (int i = 0; i < 500; i++) {
        for (int j = 0; j < 800; j++) {
            output[i] = output[i] + input[j] * binip1_linear_weight[i][j];
        }
        
        // alpha と bias の処理
        output[i] = output[i] * binip1_alpha[i] + binip1_linear_bias[i];
    }
}

void relu3(float input[500], float output[500]) {
    for (int i = 0; i < 500; i++) {
        output[i] = fmaxf(0, input[i]);
    }
}

void ip2(float input[500], float output[10]) {
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 500; j++) {
            output[i] += input[j] * ip2_weight[i][j];
        }
    }
    
    // bias の処理
    for (int i = 0; i < 10; i++) {
        output[i] = output[i] + ip2_bias[i];
    }
}

    









































