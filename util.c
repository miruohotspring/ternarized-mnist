#include "util.h"

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
