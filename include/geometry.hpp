#ifndef _RCJ_WW_LUCK_FOX_GEOMETRY_HPP_
#define _RCJ_WW_LUCK_FOX_GEOMETRY_HPP_

#include "angles.hpp"

namespace ww_geom {

    struct Point{
        int x = 0;
        int y = 0;
    };

    constexpr int Comp(Point p1, Point p2);
    
    struct LineEquation {
        int a = 0;
        int b = 0;
        int c = 0;
    };

    struct Segment{
    private:
        Point begin_;
        Point end_;
        LineEquation eq_;

    public:
        constexpr Segment(Point, Point);
        constexpr Segment(Point begin, Deg angle, int length);
        constexpr int PointEquation(Point) const;
        constexpr LineEquation Equation() const { return eq_; };

        constexpr Point Begin() const { return begin_; }
        constexpr Point End() const { return end_; }
        
        constexpr int A() const { return eq_.a; }
        constexpr int B() const { return eq_.b; }
        constexpr int C() const { return eq_.c; }
    };

    constexpr int Comp(const Segment& p1, const Segment& p2);

    struct Blob{
        static constexpr int vert_cnt = 4;
        Point p[vert_cnt];
        Point center;
    };

    constexpr int CalcPointDistanceSq(Point p1, Point p2);
    constexpr int CalcPointDistance(Point p1, Point p2);
    constexpr int CalcSegmentDist(Point p, Segment s);
    constexpr bool SegmentIntersection(Segment s1, Segment s2, Point &p);
    constexpr int DistToPolygon(Segment ray, const Blob& blob);

} // namespace ww_geom

#include "geom_impl.cpp"

#endif
