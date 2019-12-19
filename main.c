#include <stdio.h>
#include "const.h"
#include "data.h"
#include "layers.h"

int main() {
    //printf("%.8f\n", x_test[0][0]);
    
    int ouputs[__DATANUM__];
    
    for (int n = 0; n < __DATANUM__; n++) {
        //x_testは入力画像の配列
        //各層の順伝播関数は 層の名前(入力配列, 出力配列) というルールで実装してます
        float conv1_out[20][24][24] = {{{0}}};
        conv1(x_test[n], conv1_out);
        float bn1_out[20][24][24];
        bn1(conv1_out, bn1_out);

        //---------デバッグ用のコード-----------
        printf("[\n");
        for (int c = 0; c < 20; c++) {
            printf("[");
            for (int w = 0; w < 24; w++) {
                printf("%d, %d\t[", c, w);
                for (int h = 0; h < 24; h++) {
                    if (h%6 == 5) {
                        printf("%.8f\n", bn1_out[c][w][h]);
                    } else {
                        printf("%.8f,\t", bn1_out[c][w][h]);
                    }
                }
                printf("]\n");
            }
            printf("]\n");
        }
        printf("]\n");
        //-----------デバッグここまで-----------

/*
        float relu1_out[20][24][24];
        relu1(bn1_out, relu1_out);
        float pool1_out[20][12][12];
        pool1(relu1_out, pool1_out);
        float bn2_out[20][12][12];
        bn2(pool1_out, bn2_out);
        
        // 3値化畳み込み層
        int conv2_in[20][12][12];
        ternarize1(bn2_out, conv2_in);
        float conv2_out[50][8][8];
        conv2(conv2_in, conv2_out);
        
        float relu2_out[50][8][8];
        relu2(conv2_out, relu2_out);
        float pool2_out[50][4][4];
        pool2(relu2_out, pool2_out);
        float bn3_out[50][4][4];
        bn3(pool2_out, bn3_out);
        
        //3値化全結合層
        int ip1_in[800];
        ternarize2(bn3_out, ip1_in);
        float ip1_out[500];
        ip1(bn3_out, ip1_out);
        
        float ip2_out[10];
        ip2(ip1_out, ip2_out);
        outputs[n] = argmax(ip2_out);
*/
    }
    
    /*
    int count = 0;
    for (int n = 0; n < N; n++) {
        if (outputs[n] == y_test[n]) count++;
    }
    printf("accuracy: %f", (count/N)*100);
    */
    
    return 0;
}










































