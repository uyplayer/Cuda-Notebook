
#include "common/common.h"
#include "include/week01.h"


TEST(Week01, Add1Dim) {
    bool result = add1_dim();
    std::cout<< result << std::endl;
    ASSERT_TRUE(result);
}

TEST(Week01, Add2Dim) {
    bool result = add2_dim();
    std::cout<< result << std::endl;
    ASSERT_TRUE(result);
}

TEST(Week01, Add3Dim) {
    bool result = add3_dim();
    std::cout<< result << std::endl;
    ASSERT_TRUE(result);
}

int add(int a, int b) {
    return a + b;
}
TEST(Week01, AddFunction) {
    int result = add(3, 5);
    ASSERT_EQ(result, 8);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}