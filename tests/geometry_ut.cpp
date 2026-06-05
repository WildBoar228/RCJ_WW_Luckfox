// geomtry_test.cpp
#include <gtest/gtest.h>
#include <cmath>
#include "geometry.hpp"

namespace {
    using namespace ww_geom;

    static constexpr double kEps = 1e-9;

    // Helper: expected endpoint with the new angle convention.
    // 0° → up ( +y ), 90° → right ( +x ), 180° → down ( -y ), 270° → left ( -x ).
    Point expectedEndpointFromAngle(Point start, Deg angle, int length) {
        double rad = static_cast<double>(angle);   // radians using Deg::operator double()
        int dx = static_cast<int>(std::round(length * std::sin(rad)));
        int dy = static_cast<int>(std::round(length * std::cos(rad)));
        return Point{start.x + dx, start.y + dy};
    }

    // -------------------------------------------------------------------
    // Point and Comp(Point)
    // -------------------------------------------------------------------

    TEST(PointTest, DefaultConstruction) {
        Point p;
        EXPECT_EQ(p.x, 0);
        EXPECT_EQ(p.y, 0);
    }

    TEST(PointTest, MemberInitialization) {
        Point p{3, -7};
        EXPECT_EQ(p.x, 3);
        EXPECT_EQ(p.y, -7);
    }

    TEST(PointCompTest, EqualPoints) {
        EXPECT_EQ(Comp(Point{2, 5}, Point{2, 5}), 0);
    }

    TEST(PointCompTest, XFirst) {
        EXPECT_LT(Comp(Point{1, 0}, Point{2, 0}), 0);
        EXPECT_GT(Comp(Point{2, 0}, Point{1, 0}), 0);
    }

    TEST(PointCompTest, XEqualYDifferent) {
        EXPECT_LT(Comp(Point{3, -1}, Point{3, 2}), 0);
        EXPECT_GT(Comp(Point{3, 2}, Point{3, -1}), 0);
    }

    TEST(PointCompTest, FullOrdering) {
        Point a{0, 0}, b{0, 1}, c{1, 0}, d{1, 1};
        EXPECT_LT(Comp(a, b), 0);
        EXPECT_LT(Comp(a, c), 0);
        EXPECT_LT(Comp(a, d), 0);
        EXPECT_LT(Comp(b, c), 0);
        EXPECT_LT(Comp(c, d), 0);
    }

    // -------------------------------------------------------------------
    // Segment from two points
    // -------------------------------------------------------------------

    TEST(SegmentTest, FromTwoPoints) {
        Segment s{Point{1, 2}, Point{4, 6}};
        EXPECT_EQ(s.Begin().x, 1);
        EXPECT_EQ(s.Begin().y, 2);
        EXPECT_EQ(s.End().x, 4);
        EXPECT_EQ(s.End().y, 6);
    }

    TEST(SegmentTest, LineEquationZeroForEndpoints) {
        Segment s{Point{0, 0}, Point{10, 0}};
        EXPECT_EQ(s.PointEquation(s.Begin()), 0);
        EXPECT_EQ(s.PointEquation(s.End()), 0);
        EXPECT_NE(s.PointEquation(Point{5, 1}), 0);
    }

    TEST(SegmentTest, EquationAccessors) {
        Segment s{Point{1, 1}, Point{4, 5}};
        EXPECT_EQ(s.Equation().a, s.A());
        EXPECT_EQ(s.Equation().b, s.B());
        EXPECT_EQ(s.Equation().c, s.C());
        EXPECT_EQ(s.A() * s.Begin().x + s.B() * s.Begin().y + s.C(), 0);
        EXPECT_EQ(s.A() * s.End().x + s.B() * s.End().y + s.C(), 0);
    }

    // -------------------------------------------------------------------
    // Segment from point, angle, length  (new convention)
    // -------------------------------------------------------------------

    TEST(SegmentTest, FromAngleZeroDeg) {
        Point start{5, 5};
        Segment s{start, Deg{0}, 10};
        auto expected = expectedEndpointFromAngle(start, Deg{0}, 10);
        EXPECT_EQ(s.End().x, expected.x);
        EXPECT_EQ(s.End().y, expected.y);   // (5, 15)
    }

    TEST(SegmentTest, FromAngle90Deg) {
        Point start{5, 5};
        Segment s{start, Deg{90}, 7};
        auto expected = expectedEndpointFromAngle(start, Deg{90}, 7);
        EXPECT_EQ(s.End().x, expected.x);   // 12
        EXPECT_EQ(s.End().y, expected.y);   // 5
    }

    TEST(SegmentTest, FromAngle180Deg) {
        Point start{5, 5};
        Segment s{start, Deg{180}, 3};
        auto expected = expectedEndpointFromAngle(start, Deg{180}, 3);
        EXPECT_EQ(s.End().x, expected.x);   // 5
        EXPECT_EQ(s.End().y, expected.y);   // 2
    }

    TEST(SegmentTest, FromAngleMinus90Deg) {
        Point start{0, 0};
        Segment s{start, Deg{-90}, 4};
        auto expected = expectedEndpointFromAngle(start, Deg{-90}, 4);
        EXPECT_EQ(s.End().x, expected.x);   // -4
        EXPECT_EQ(s.End().y, expected.y);   // 0
    }

    TEST(SegmentTest, FromAngle45DegRounding) {
        Point start{0, 0};
        Segment s{start, Deg{45}, 10};
        auto expected = expectedEndpointFromAngle(start, Deg{45}, 10);
        EXPECT_EQ(s.End().x, 7);
        EXPECT_EQ(s.End().y, 7);
    }

    TEST(SegmentTest, FromAngleSmallLengthRounding) {
        Segment s{Point{0, 0}, Deg{30}, 300};
        EXPECT_NEAR(s.End().x, 150, 1);
        EXPECT_NEAR(s.End().y, 260, 1);
    }

    TEST(SegmentTest, FromAngleNegativeLength) {
        Point start{2, 2};
        Segment s{start, Deg{0}, -5};
        auto expected = expectedEndpointFromAngle(start, Deg{0}, -5);
        // 0° with negative length goes straight down 5 units
        EXPECT_EQ(s.End().x, 2);
        EXPECT_EQ(s.End().y, -3);
    }

    // -------------------------------------------------------------------
    // Comp(Segment)
    // -------------------------------------------------------------------

    TEST(SegmentCompTest, EqualSegments) {
        Segment a{Point{0,0}, Point{1,1}};
        Segment b{Point{0,0}, Point{1,1}};
        EXPECT_EQ(Comp(a, b), 0);
    }

    TEST(SegmentCompTest, ByBeginThenEnd) {
        Segment s1{Point{0,0}, Point{2,2}};
        Segment s2{Point{0,0}, Point{3,3}};
        EXPECT_LT(Comp(s1, s2), 0);
        EXPECT_GT(Comp(s2, s1), 0);

        Segment s3{Point{1,1}, Point{0,0}};
        EXPECT_GT(Comp(s3, s1), 0);  // begin (1,1) > (0,0)
    }

    // -------------------------------------------------------------------
    // Distance functions
    // -------------------------------------------------------------------

    TEST(CalcPointDistanceSqTest, SamePoint) {
        EXPECT_EQ(CalcPointDistanceSq(Point{1,2}, Point{1,2}), 0);
    }

    TEST(CalcPointDistanceSqTest, Horizontal) {
        EXPECT_EQ(CalcPointDistanceSq(Point{0,0}, Point{5,0}), 25);
    }

    TEST(CalcPointDistanceSqTest, Vertical) {
        EXPECT_EQ(CalcPointDistanceSq(Point{0,0}, Point{0,-3}), 9);
    }

    TEST(CalcPointDistanceSqTest, Diagonal) {
        EXPECT_EQ(CalcPointDistanceSq(Point{0,0}, Point{3,4}), 25);
    }

    TEST(CalcPointDistanceTest, ExactInteger) {
        EXPECT_EQ(CalcPointDistance(Point{0,0}, Point{3,4}), 5);
    }

    TEST(CalcPointDistanceTest, Truncated) {
        EXPECT_EQ(CalcPointDistance(Point{0,0}, Point{1,1}), 1);
    }

    TEST(CalcPointDistanceTest, Zero) {
        EXPECT_EQ(CalcPointDistance(Point{7,7}, Point{7,7}), 0);
    }

    // -------------------------------------------------------------------
    // CalcSegmentDist (point to segment)
    // -------------------------------------------------------------------

    TEST(CalcSegmentDistTest, PointOnSegment) {
        Segment s{Point{0,0}, Point{10,0}};
        EXPECT_EQ(CalcSegmentDist(Point{5,0}, s), 0);
    }

    TEST(CalcSegmentDistTest, PerpendicularProjectionInside) {
        Segment s{Point{0,0}, Point{10,0}};
        EXPECT_EQ(CalcSegmentDist(Point{5,3}, s), 3);
    }

    TEST(CalcSegmentDistTest, ClosestToEndPointBeyondEnd) {
        Segment s{Point{0,0}, Point{10,0}};
        EXPECT_EQ(CalcSegmentDist(Point{15, 0}, s), 5);
    }

    TEST(CalcSegmentDistTest, ClosestToBeginPointBeyondBegin) {
        Segment s{Point{0,0}, Point{10,0}};
        EXPECT_EQ(CalcSegmentDist(Point{-4, 0}, s), 4);
    }

    TEST(CalcSegmentDistTest, DiagonalSegmentPointProjectionInside) {
        Segment s{Point{0,0}, Point{4,4}};
        EXPECT_EQ(CalcSegmentDist(Point{2,0}, s), 1);  // distance ≈1.414, truncated to 1
    }

    TEST(CalcSegmentDistTest, ZeroLengthSegment) {
        Segment s{Point{3,3}, Point{3,3}};
        EXPECT_EQ(CalcSegmentDist(Point{3,4}, s), 1);
    }

    // -------------------------------------------------------------------
    // SegmentIntersection
    // -------------------------------------------------------------------

    TEST(SegmentIntersectionTest, Crossing) {
        Segment a{Point{0,0}, Point{10,10}};
        Segment b{Point{0,10}, Point{10,0}};
        Point p;
        ASSERT_TRUE(SegmentIntersection(a, b, p));
        EXPECT_EQ(p.x, 5);
        EXPECT_EQ(p.y, 5);
    }

    TEST(SegmentIntersectionTest, EndpointTouch) {
        Segment a{Point{0,0}, Point{5,5}};
        Segment b{Point{5,5}, Point{10,10}};
        Point p;
        ASSERT_TRUE(SegmentIntersection(a, b, p));
        EXPECT_EQ(p.x, 5);
        EXPECT_EQ(p.y, 5);
    }

    TEST(SegmentIntersectionTest, ParallelNoIntersection) {
        Segment a{Point{0,0}, Point{10,0}};
        Segment b{Point{0,1}, Point{10,1}};
        Point p;
        EXPECT_FALSE(SegmentIntersection(a, b, p));
    }

    TEST(SegmentIntersectionTest, CollinearOverlap) {
        Segment a{Point{0,0}, Point{10,0}};
        Segment b{Point{5,0}, Point{15,0}};
        Point p;
        EXPECT_NO_THROW(SegmentIntersection(a, b, p));  // non‑crash test
    }

    TEST(SegmentIntersectionTest, NoIntersectionApart) {
        Segment a{Point{0,0}, Point{1,1}};
        Segment b{Point{2,2}, Point{3,3}};
        Point p;
        EXPECT_FALSE(SegmentIntersection(a, b, p));
    }

    // -------------------------------------------------------------------
    // Blob
    // -------------------------------------------------------------------

    TEST(BlobTest, VertexCount) {
        EXPECT_EQ(Blob::vert_cnt, 4);
    }

    TEST(BlobTest, CanSetVerticesAndCenter) {
        Blob blob;
        blob.p[0] = Point{0,0};
        blob.p[1] = Point{10,0};
        blob.p[2] = Point{10,10};
        blob.p[3] = Point{0,10};
        blob.center = Point{5,5};
        EXPECT_EQ(blob.p[0].x, 0);
        EXPECT_EQ(blob.center.x, 5);
    }

    // -------------------------------------------------------------------
    // DistToPolygon (ray to blob) — using two‑point segments to avoid angle
    // -------------------------------------------------------------------

    TEST(DistToPolygonTest, RayIntersectsBlobFromLeft) {
        Blob blob;
        blob.p[0] = Point{0,0};
        blob.p[1] = Point{10,0};
        blob.p[2] = Point{10,10};
        blob.p[3] = Point{0,10};
        blob.center = Point{5,5};

        // Horizontal ray entering the blob from the left
        Segment ray{Point{-1, 5}, Point{11, 5}};   // from x=-1 to x=11, y=5
        int dist = DistToPolygon(ray, blob);
        // Distance from ray start (-1,5) to first intersection (0,5) is 1
        EXPECT_EQ(dist, 1);
    }

    TEST(DistToPolygonTest, RayOutsideNoHit) {
        Blob blob;
        blob.p[0] = Point{0,0};
        blob.p[1] = Point{10,0};
        blob.p[2] = Point{10,10};
        blob.p[3] = Point{0,10};
        blob.center = Point{5,5};

        // Ray going straight down far away from blob
        Segment ray{Point{20, 20}, Point{20, 30}};
        int dist = DistToPolygon(ray, blob);
        EXPECT_GT(dist, 0);   // no intersection, distance may be a large sentinel
    }

    // -------------------------------------------------------------------
    // Constexprness (compile‑time check)
    // -------------------------------------------------------------------

    TEST(CompileTime, FunctionsAreConstexpr) {
        constexpr Point p1{1, 2};
        constexpr Point p2{3, 4};
        constexpr int cmp = Comp(p1, p2);
        static_assert(cmp < 0, "constexpr Comp");

        constexpr Segment s{Point{0,0}, Point{10,0}};
        constexpr int eq = s.PointEquation(Point{5, 0});
        static_assert(eq == 0, "constexpr PointEquation");
    }

} // namespace
