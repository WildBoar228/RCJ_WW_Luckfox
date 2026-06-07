#include <cmath>
#include <gtest/gtest.h>
#include <sstream>
#include <string>

#include "angles.hpp"

namespace {

    using namespace ww_vision;

    constexpr double kEps = 1e-9;

    // ==================== Deg tests ====================

    TEST(DegTest, ConstructionAndAccessor) {
        Deg d{42};
        EXPECT_EQ(d.deg, 42);
    }

    TEST(DegTest, ExplicitIntConversion) {
        Deg d{42};
        int i = static_cast<int>(d);
        EXPECT_EQ(i, 42);
    }

    TEST(DegTest, ImplicitConversionToDoubleRadians) {
        Deg d{90};
        double rad = d;   // operator double()
        EXPECT_NEAR(rad, kPi / 2.0, kEps);
    }

    TEST(DegTest, ImplicitConversionToRad) {
        Rad r = Deg{180};
        EXPECT_NEAR(r.rad, kPi, kEps);
    }

    TEST(DegTest, Addition) {
        Deg a{30}, b{40};
        Deg sum = a + b;
        EXPECT_EQ(sum.deg, 70);
    }

    TEST(DegTest, Subtraction) {
        Deg a{90}, b{50};
        Deg diff = a - b;
        EXPECT_EQ(diff.deg, 40);
    }

    TEST(DegTest, UnaryMinus) {
        Deg d{15};
        EXPECT_EQ((-d).deg, -15);
    }

    TEST(DegTest, MultiplicationByInt) {
        Deg d{45};
        EXPECT_EQ((d * 3).deg, 135);
        EXPECT_EQ((2 * d).deg, 90);
    }

    TEST(DegTest, DivisionByInt) {
        Deg d{180};
        EXPECT_EQ((d / 3).deg, 60);
    }

    TEST(DegTest, ComparisonOperators) {
        Deg a{10}, b{20}, c{10};
        EXPECT_TRUE(a == c);
        EXPECT_FALSE(a == b);
        EXPECT_TRUE(a != b);
        EXPECT_FALSE(a != c);
        EXPECT_TRUE(a < b);
        EXPECT_TRUE(a <= b);
        EXPECT_TRUE(a <= c);
        EXPECT_FALSE(a > b);
        EXPECT_FALSE(a >= b);
        EXPECT_TRUE(a >= c);
        EXPECT_TRUE(b > a);
        EXPECT_TRUE(b >= a);
    }

    TEST(DegTest, OutputStream) {
        Deg d{90};
        std::ostringstream os;
        os << d;
        EXPECT_EQ(os.str(), "90 deg");
    }

    TEST(DegTest, NegativeAngle) {
        Deg d{-45};
        EXPECT_EQ(d.deg, -45);
        EXPECT_NEAR(static_cast<double>(d), -kPi / 4.0, kEps);
        Rad r = d;
        EXPECT_NEAR(r.rad, -kPi / 4.0, kEps);
        EXPECT_EQ((-d).deg, 45);
    }

    TEST(DegTest, ZeroDegree) {
        Deg z{0};
        EXPECT_EQ(z.deg, 0);
        EXPECT_DOUBLE_EQ(static_cast<double>(z), 0.0);
        Rad r = z;
        EXPECT_DOUBLE_EQ(r.rad, 0.0);
    }

    TEST(DegTest, LargeValues) {
        Deg d{360};
        EXPECT_NEAR(static_cast<double>(d), 2 * kPi, kEps);
        Deg neg{-720};
        EXPECT_NEAR(static_cast<double>(neg), -4 * kPi, kEps);
    }

    // ==================== Rad tests ====================

    TEST(RadTest, ConstructionAndAccessor) {
        Rad r{1.234};
        EXPECT_DOUBLE_EQ(r.rad, 1.234);
    }

    TEST(RadTest, ImplicitConversionToDouble) {
        Rad r{3.14};
        double val = r;
        EXPECT_DOUBLE_EQ(val, 3.14);
    }

    TEST(RadTest, ImplicitConversionToDegExact) {
        Rad r{kPi};            // pi rad = 180 deg
        Deg d = r;
        EXPECT_EQ(d.deg, 180);
    }

    TEST(RadTest, ConversionToDegRounding) {
        // 0.5 rad ~ 28.6479 deg -> rounds to 29 deg
        Deg d = Rad{0.5};
        EXPECT_EQ(d.deg, 29);
    }

    TEST(RadTest, ConversionToDegRoundingNegative) {
        // -0.5 rad ~ -28.6479 deg -> rounds to -29 deg
        Deg d = Rad{-0.5};
        EXPECT_EQ(d.deg, -29);
    }

    TEST(RadTest, ConversionToDegEdge) {
        // 0.499 rad ~ 28.59 deg -> still rounds to 29 deg
        Deg d = Rad{0.499};
        EXPECT_EQ(d.deg, 29);
    }

    TEST(RadTest, ConversionToDegAtHalf) {
        // exactly 0.5 rad -> should round to 29
        EXPECT_EQ(Deg{Rad{0.5}}.deg, 29);
    }

    TEST(RadTest, Addition) {
        Rad a{1.0}, b{2.5};
        EXPECT_DOUBLE_EQ((a + b).rad, 3.5);
    }

    TEST(RadTest, Subtraction) {
        Rad a{5.0}, b{1.5};
        EXPECT_DOUBLE_EQ((a - b).rad, 3.5);
    }

    TEST(RadTest, UnaryMinus) {
        EXPECT_DOUBLE_EQ((-Rad{2.0}).rad, -2.0);
    }

    TEST(RadTest, MultiplicationByDouble) {
        Rad r{2.0};
        EXPECT_DOUBLE_EQ((r * 3.0).rad, 6.0);
        EXPECT_DOUBLE_EQ((1.5 * r).rad, 3.0);
    }

    TEST(RadTest, DivisionByDouble) {
        EXPECT_DOUBLE_EQ((Rad{6.0} / 3.0).rad, 2.0);
    }

    TEST(RadTest, ComparisonOperators) {
        Rad a{0.5}, b{1.0}, c{0.5};
        EXPECT_TRUE(a == c);
        EXPECT_FALSE(a == b);
        EXPECT_TRUE(a != b);
        EXPECT_FALSE(a != c);
        EXPECT_TRUE(a < b);
        EXPECT_TRUE(a <= b);
        EXPECT_TRUE(a <= c);
        EXPECT_FALSE(a > b);
        EXPECT_FALSE(a >= b);
        EXPECT_TRUE(a >= c);
        EXPECT_TRUE(b > a);
        EXPECT_TRUE(b >= a);
    }

    TEST(RadTest, OutputStream) {
        Rad r{1.5708};
        std::ostringstream os;
        os << r;
        std::string s = os.str();
        EXPECT_NE(s.find(" rad"), std::string::npos);
        // verify suffix
        EXPECT_EQ(s.substr(s.size() - 4), " rad");
    }

    TEST(RadTest, NegativeRadian) {
        Rad r{-1.0};
        EXPECT_DOUBLE_EQ(r.rad, -1.0);
        EXPECT_EQ(Deg(r).deg, -57);   // -1 rad ~ -57.3 deg -> -57
    }

    TEST(RadTest, ZeroRadian) {
        Rad r{0.0};
        EXPECT_DOUBLE_EQ(r.rad, 0.0);
        EXPECT_EQ(Deg(r).deg, 0);
    }

    // ==================== Math integration ====================

    TEST(MathFunctions, SinOfDeg) {
        EXPECT_NEAR(std::sin(Deg{30}), 0.5, kEps);
    }

    TEST(MathFunctions, CosOfDeg) {
        EXPECT_NEAR(std::cos(Deg{60}), 0.5, kEps);
    }

    TEST(MathFunctions, TanOfDeg) {
        EXPECT_NEAR(std::tan(Deg{45}), 1.0, kEps);
    }

    TEST(MathFunctions, SinOfRad) {
        EXPECT_NEAR(std::sin(Rad{kPi / 2}), 1.0, kEps);
    }

    TEST(MathFunctions, CosOfRad) {
        EXPECT_NEAR(std::cos(Rad{kPi}), -1.0, kEps);
    }

    // ==================== Implicit conversion in function arguments ====================

    int AcceptDeg(Deg ang) { return ang.deg; }
    double AcceptRad(Rad ang) { return ang.rad; }

    TEST(ImplicitConversion, DegAsRadArgument) {
        EXPECT_EQ(AcceptDeg(Deg{90}), 90);
        EXPECT_NEAR(AcceptRad(Deg{90}), kPi / 2.0, kEps);
    }

    TEST(ImplicitConversion, RadAsDegArgument) {
        EXPECT_EQ(AcceptDeg(Rad{1}), 57);
        EXPECT_NEAR(AcceptRad(Rad{1}), 1, kEps);
    }

    // ==================== Compile‑time check for no ambiguity ====================

    TEST(NoAmbiguity, SinCallCompiles) {
        Deg d{90};
        volatile double s = std::sin(d);
        (void)s;   // just confirm it compiles and runs
    }

} // namespace
