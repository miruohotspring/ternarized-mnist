#include <stdio.h>
#include "const.h"
#include "data.h"
#include "layers.h"
#include "util.h"

int main() {
    
    int outputs[__DATANUM__];
    
    for (int n = 0; n < __DATANUM__; n++) {
        float input[28][28] = {{0}};
        normalize(supertest[n], input);
        float conv1_out[20][24][24] = {{{0}}};
        conv1(input, conv1_out);
        float bn1_out[20][24][24] = {{{0}}};
        bn1(conv1_out, bn1_out);
        float relu1_out[20][24][24] = {{{0}}};
        relu1(bn1_out, relu1_out);
        float pool1_out[20][12][12] = {{{0}}};
        pool1(relu1_out, pool1_out);
        float bn2_out[20][12][12] = {{{0}}};
        bn2(pool1_out, bn2_out);

        // 3値化畳み込み層
        int conv2_in[20][12][12] = {{{0}}};
        ternarize1(bn2_out, conv2_in);
        float conv2_out[50][8][8] = {{{0}}};
        conv2(conv2_in, conv2_out);
        
        float relu2_out[50][8][8] = {{{0}}};
        relu2(conv2_out, relu2_out);
        float pool2_out[50][4][4] = {{{0}}};
        pool2(relu2_out, pool2_out);
        float bn3_out[50][4][4];
        bn3(pool2_out, bn3_out);
        
        //3値化全結合層
        int ip1_in[800];
        ternarize2(bn3_out, ip1_in);
        float ip1_out[500];
        ip1(ip1_in, ip1_out);
        
        float relu3_out[500] = {0};
        relu3(ip1_out, relu3_out);
        float ip2_out[10];
        ip2(relu3_out, ip2_out);
        outputs[n] = argmax(ip2_out);

/*
        printf("[\n");
        for (int c = 0; c < 1; c++) {
            printf("[");
            for (int h = 0; h < 1; h++) {
                printf("[");
                for (int w = 0; w < 10; w++) {
                  if (w%6 == 5) {
                      printf("%f\n", ip2_out[w]);
                  } else {
                      printf("%f,\t", ip2_out[w]);
                  }
                }
               printf("]\n");
            }
            printf("]\n");
        }
        printf("]\n");
*/
    }
    
    return 0;
}










































