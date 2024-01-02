
#include "common.h"
#include "week01.h"
#include <catch2/catch_test_macros.hpp>



TEST_CASE("Week01 Add1Dim", "[Week01]") {
    WARN("Week01 Add1Dim");
    REQUIRE(add1_dim());
}

TEST_CASE("Week01 Add2Dim", "[Week01]") {
    WARN("Week01 Add2Dim");
    REQUIRE(add2_dim());
}

TEST_CASE("Week01 Add3Dim", "[Week01]") {
    WARN("Week01 Add3Dim");
    REQUIRE(add3_dim());
}
