#include "layers.h"
#include "params.h"
#include "math.h"
#include "stdio.h"

void conv1(float input[28][28], float output[20][24][24]){
    for (int c = 0; c < 20; c++) {
        for (int i = 0; i < 24; i++) {
            for (int j = 0; j < 24; j++) {
                float ys[25];
                
                for (int wi = 0; wi < 5; wi++) {
                    for (int wj = 0; wj < 5; wj++) {
                        ys[wi*5+wj] = input[i+wi][j+wj] * conv1_weight[c][0][wi][wj];
                    }
                }
                
                for (int yi = 0; yi < 25; yi++) {
                    output[c][i][j] += ys[yi];
                }
                output[c][i][j] += conv1_bias[c];
            }
        }
    }
}

void bn1(float input[20][24][24], float output[20][24][24]){
    for (int c = 0; c < 20; c++) {
        for (int w = 0; w < 24; w++) {
            for (int h = 0; h < 24; h++) {
                output[c][w][h] = (input[c][w][h] - bnconv1_runningmean[c])\
                                    /(sqrt(bnconv1_runningvar[c]) + 0.00001);
            }
        }
    }
}

// 畳み込みフィルタbinconv2_weight[50][20][5][5]を使って
// input[20][12][12] -> output[50][8][8]
void conv2(int input[20][12][12], float output[50][8][8]){
    for (int j = 0; j < 50; j++) {
        for (int k = 0; k < 20; k++) {
            for (int w = 0; w < 8; w++) {
                for (int h = 0; h < 8; h++) {
                    
                    float ys[25];
                    for (int fw = 0; fw < 5; fw++) {
                        for (int fh = 0; fh < 5; fh++) {
                            ys[fw*5+fh] = input[k][w+fw][h+fh] * binconv2_conv_weight[j][k][fw][fh];
                        }
                    }
                    for (int yi = 0; yi < 25; yi++) {
                        output[j][w][h] += ys[yi];
                    }
                    
                    // バイアスとフィルタ係数αの処理
                    output[j][w][h] = (output[j][w][h] + binconv2_conv_bias[j])*binconv2_alpha[j];
                }
            }
        }
    }
}










































