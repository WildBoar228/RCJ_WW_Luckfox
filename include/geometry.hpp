#ifndef _RCJ_WW_LUCKFOX_GEOMETRY_HPP_
#define _RCJ_WW_LUCKFOX_GEOMETRY_HPP_

#include "angles.hpp"

namespace ww {
namespace vision {

    struct Point{
        int x = 0;
        int y = 0;
    };

    template <typename CvPoint>
    Point CastToPoint(const CvPoint& p) {
        return Point{
            .x = static_cast<int>(p.x),
            .y = static_cast<int>(p.y)
        };
    }

    constexpr int Comp(Point p1, Point p2);
    constexpr Deg AngleTo(Point from, Point to);
    
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
        constexpr Segment() = default;
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

    struct BlobGeom{
        static constexpr int vert_cnt = 4;
        Point p[vert_cnt];
        Point center;
        int area;
    };

    constexpr int CalcPointDistanceSq(Point p1, Point p2);
    constexpr int CalcPointDistance(Point p1, Point p2);
    constexpr int CalcSegmentDist(Point p, Segment s);
    constexpr bool SegmentIntersection(Segment s1, Segment s2, Point &p);
    constexpr int CalcPolygonDist(Segment ray, const BlobGeom& blob);

} // namespace vision
} // namespace ww

#include "geom_impl.cpp"

#endif
