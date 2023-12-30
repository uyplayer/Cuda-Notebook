
#include "common.h"
#include "week01.h"


TEST(Week01, Add1Dim) {
    bool result = add1_dim();
    ASSERT_TRUE(result);
}

TEST(Week01, Add2Dim) {
    bool result = add2_dim();
    ASSERT_TRUE(result);
}

TEST(Week01, Add3Dim) {
    bool result = add3_dim();
    ASSERT_TRUE(result);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    return result;
}
