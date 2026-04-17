#include <stdio.h>

int add(int a, int b) {
    return a + b;
}

int main(void) {
    int x = add(2, 3);
    printf("result=%d\n", x);
    return 0;
}
