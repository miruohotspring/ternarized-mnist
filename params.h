#ifndef PARAMS_H
#define PARAMS_H

extern float conv1_weight[20][1][5][5];
extern float conv1_bias[20];
extern float bnconv1_runningmean[20];
extern float bnconv1_runningvar[20];
extern float bnconv1_numbatchestracked;
extern float binconv2_bn_weight[20];
extern float binconv2_bn_bias[20];
extern float binconv2_bn_runningmean[20];
extern float binconv2_bn_runningvar[20];
extern float binconv2_bn_numbatchestracked;
extern int binconv2_conv_weight[50][20][5][5];
extern float binconv2_conv_bias[50];
extern float binip1_bn_weight[50];
extern float binip1_bn_bias[50];
extern float binip1_bn_runningmean[50];
extern float binip1_bn_runningvar[50];
extern float binip1_bn_numbatchestracked;
extern float binip1_linear_weight[500][800];
extern float binip1_linear_bias[500];
extern float ip2_weight[10][500];
extern float ip2_bias[10];
extern float binconv2_alpha[50];
extern float binip1_alpha[500];

#endif