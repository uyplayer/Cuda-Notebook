
#include "../common/common.h"
#include "../week01/week01.h"


TEST(VectorAddTest, BasicTest) {
    bool res = runVectorAddition();
    ASSERT_EQ(true, res);
}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}