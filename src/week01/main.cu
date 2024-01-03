
#include "common.h"


extern void run_add1_dim();

extern void run_add2_dim();

extern void run_add3_dim();
extern void run_transpose();

extern void run_shared_memory();
extern void run_const_memory();

int main(int argc, char **argv) {
    run_add1_dim();
    run_add2_dim();
    run_add3_dim();
    run_transpose();
    run_shared_memory();
    run_const_memory();
    return 0;
}


