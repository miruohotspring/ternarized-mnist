#include "layers.h"
#include "params.h"
#include "math.h"
#include <stdio.h>
#include "minitrace/minitrace.h"

void conv1(float input[28][28], float output[20][24][24]){
    MTR_BEGIN_FUNC();
    for (int c = 0; c < 20; c++) {
        MTR_BEGIN("conv1", "subloop1");
        for (int h = 0; h < 24; h++) {
            MTR_BEGIN("conv1", "subloop2");
            for (int w = 0; w < 24; w++) {
                MTR_BEGIN("conv1", "width");
                for (int hf = 0; hf < 5; hf++) {
                    for (int wf = 0; wf < 5; wf++) {
                        output[c][h][w] += input[h+hf][w+wf] * conv1_weight[c][0][hf][wf];
                    }
                }
                MTR_END("conv1", "width");
                output[c][h][w] += conv1_bias[c];
            }
            MTR_END("conv1", "height");
        }
        MTR_END("conv1", "channel");
    }
    MTR_END_FUNC();
}

void bn1(float input[20][24][24], float output[20][24][24]){
    MTR_BEGIN_FUNC();
    for (int c = 0; c < 20; c++) {
        for (int h = 0; h < 24; h++) {
            for (int w = 0; w < 24; w++) {
                output[c][h][w] = (input[c][h][w] - bnconv1_runningmean[c])\
                                    /(sqrt(bnconv1_runningvar[c]) + 0.0001);
            }
        }
    }
    MTR_END_FUNC();
}

void relu1(float input[20][24][24], float output[20][24][24]) {
    MTR_BEGIN_FUNC();
    for (int c = 0; c < 20; c++) {
        for (int h = 0; h < 24; h++) {
            for (int w = 0; w < 24; w++) {
                output[c][h][w] = fmaxf(0., input[c][h][w]);
            }
        }
    }
    MTR_END_FUNC();
}

void pool1(float input[20][24][24], float output[20][12][12]) {
    MTR_BEGIN_FUNC();
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
    MTR_END_FUNC();
}

void bn2(float input[20][12][12], float output[20][12][12]){
    MTR_BEGIN_FUNC();
    for (int c = 0; c < 20; c++) {
        for (int h = 0; h < 12; h++) {
            for (int w = 0; w < 12; w++) {
                output[c][h][w] = ((input[c][h][w] - binconv2_bn_runningmean[c])\
                                    / (sqrt(binconv2_bn_runningvar[c]) + 0.00001))\
                                    * binconv2_bn_weight[c] + binconv2_bn_bias[c];
            }
        }
    }
    MTR_END_FUNC();
}

// 畳み込みフィルタbinconv2_weight[50][20][5][5]を使って
// input[20][12][12] -> output[50][8][8]
void conv2(int input[20][12][12], float weight[50][20][5][5], float bias[50], float alpha[50], float output[50][8][8]){
    MTR_BEGIN_FUNC();
    for (int cout = 0; cout < 50; cout++) {
        for (int h = 0; h < 8; h++) {
            for (int w = 0; w < 8; w++) {
                for (int cin = 0; cin < 20; cin++) {
                    for (int fh = 0; fh < 5; fh++) {
                        for (int fw = 0; fw < 5; fw++) {
                            output[cout][h][w] += input[cin][h+fh][w+fw] * weight[cout][cin][fh][fw];
                        }
                    }
                }
                // alphaとbiasの処理
                output[cout][h][w] = output[cout][h][w] * alpha[cout] + bias[cout];
            }
        }
    }
    MTR_END_FUNC();
}

void conv2__(int input[20][12][12], float weight[50][20][5][5], float bias[50], float alpha[50], float output[50][8][8]){
    MTR_BEGIN_FUNC();
    for (int cout = 0; cout < 50; cout++) {
        MTR_BEGIN("conv2", "subloop1");
        for (int cin = 0; cin < 20; cin++) {
            MTR_BEGIN("conv2", "subloop2");
            for (int fh = 0; fh < 5; fh++){
                for (int fw = 0; fw < 5; fw++){
                    
        MTR_BEGIN("conv2", "hls1");
        for (int h = 0; h < 8; h++) {
#pragma HLS unroll
            output[cout][h][0] += input[cin][h+fh][0+fw] * binconv2_conv_weight[cout][cin][fh][fw];
            output[cout][h][1] += input[cin][h+fh][1+fw] * binconv2_conv_weight[cout][cin][fh][fw];
            output[cout][h][2] += input[cin][h+fh][2+fw] * binconv2_conv_weight[cout][cin][fh][fw];
            output[cout][h][3] += input[cin][h+fh][3+fw] * binconv2_conv_weight[cout][cin][fh][fw];
            output[cout][h][4] += input[cin][h+fh][4+fw] * binconv2_conv_weight[cout][cin][fh][fw];
            output[cout][h][5] += input[cin][h+fh][5+fw] * binconv2_conv_weight[cout][cin][fh][fw];
            output[cout][h][6] += input[cin][h+fh][6+fw] * binconv2_conv_weight[cout][cin][fh][fw];
            output[cout][h][7] += input[cin][h+fh][7+fw] * binconv2_conv_weight[cout][cin][fh][fw];
        }
        MTR_END("conv2", "hls1");
                }
            }
            MTR_END("conv2", "subloop2");
        }

        MTR_BEGIN("conv2", "hls2");
        for (int h = 0; h < 8; h++) {
#pragma HLS unroll
            output[cout][h][0] = output[cout][h][0] * alpha[cout] + bias[cout];
            output[cout][h][1] = output[cout][h][1] * alpha[cout] + bias[cout];
            output[cout][h][2] = output[cout][h][2] * alpha[cout] + bias[cout];
            output[cout][h][3] = output[cout][h][3] * alpha[cout] + bias[cout];
            output[cout][h][4] = output[cout][h][4] * alpha[cout] + bias[cout];
            output[cout][h][5] = output[cout][h][5] * alpha[cout] + bias[cout];
            output[cout][h][6] = output[cout][h][6] * alpha[cout] + bias[cout];
            output[cout][h][7] = output[cout][h][7] * alpha[cout] + bias[cout];
        }
        MTR_END("conv2", "hls2");
        MTR_END("conv2", "subloop1");
    }
    MTR_END_FUNC();
}

void relu2(float input[50][8][8], float output[50][8][8]) {
    MTR_BEGIN_FUNC();
    for (int c = 0; c < 50; c++) {
        for (int h = 0; h < 8; h++) {
            for (int w = 0; w < 8; w++) {
                output[c][h][w] = fmaxf(0., input[c][h][w]);
            }
        }
    }
    MTR_END_FUNC();
}

void pool2(float input[50][8][8], float output[50][4][4]) {
    MTR_BEGIN_FUNC();
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
    MTR_END_FUNC();
}

void bn3(float input[50][4][4], float output[50][4][4]){
    MTR_BEGIN_FUNC();
    for (int c = 0; c < 50; c++) {
        for (int h = 0; h < 4; h++) {
            for (int w = 0; w < 4; w++) {
                output[c][h][w] = ((input[c][h][w] - binip1_bn_runningmean[c])\
                                    /(sqrt(binip1_bn_runningvar[c]) + 0.00001))\
                                    * binip1_bn_weight[c] + binip1_bn_bias[c];
            }
        }
    }
    MTR_END_FUNC();
}

// 3値化線形結合層
void ip1(int input[800], float output[500]) {
    MTR_BEGIN_FUNC();
    for (int i = 0; i < 500; i++) {
        for (int j = 0; j < 800; j++) {
            output[i] = output[i] + input[j] * binip1_linear_weight[i][j];
        }
        
        // alpha と bias の処理
        output[i] = output[i] * binip1_alpha[i] + binip1_linear_bias[i];
    }
    MTR_END_FUNC();
}

void relu3(float input[500], float output[500]) {
    MTR_BEGIN_FUNC();
    for (int i = 0; i < 500; i++) {
        output[i] = fmaxf(0, input[i]);
    }
    MTR_END_FUNC();
}

void ip2(float input[500], float output[10]) {
    MTR_BEGIN_FUNC();
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 500; j++) {
            output[i] += input[j] * ip2_weight[i][j];
        }
    }
    
    // bias の処理
    for (int i = 0; i < 10; i++) {
        output[i] = output[i] + ip2_bias[i];
    }
    MTR_END_FUNC();
}

    









































