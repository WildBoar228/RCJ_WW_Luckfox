#include <cmath>
#include <utility>

#include "geometry.hpp"

namespace ww_vision {

    namespace utils {

        template <typename T>
        bool IsBetween(T a, T from, T to) {
            return from <= a && a <= to;
        }
        
    } // namespace utils

    constexpr int Comp(Point p1, Point p2) {
        if (p1.x != p2.x) {
            return p1.x < p2.x ? -1 : 1;
        } else if (p1.y != p2.y) {
            return p1.y < p2.y ? -1 : 1;
        }
        return 0;
    }

    constexpr Deg AngleTo(Point from, Point to) {
        return Rad(std::atan2(to.y - from.y, to.x - from.x));
    }

    constexpr int Comp(const Segment& s1, const Segment& s2) {
        if (Comp(s1.Begin(), s2.Begin()) != 0) {
            return Comp(s1.Begin(), s2.Begin());
        } else if (Comp(s1.End(), s2.End()) != 0) {
            return Comp(s1.End(), s2.End());
        }
        return 0;
    }

    constexpr Segment::Segment(Point p1, Point p2)
        : begin_(p1)
        , end_(p2) {
        if (p1.x > p2.x || (p1.x == p2.x && p1.y > p2.y)) {
            std::swap(p1, p2);
        }
        eq_.a = p2.y - p1.y;
        eq_.b = p1.x - p2.x;
        eq_.c = p1.y * p2.x - p1.x * p2.y;
    }

    constexpr Segment::Segment(Point begin, Deg angle, int length)
        : begin_(begin)
        , end_(Point{
            .x = static_cast<int>(begin.x + length * std::sin(angle)),
            .y = static_cast<int>(begin.y + length * std::cos(angle))
          })
    { }

    constexpr int Segment::PointEquation(Point p) const {
        return eq_.a * p.x + eq_.b * p.y + eq_.c;
    }

    constexpr int CalcPointDistanceSq(Point p1, Point p2) {
        return (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y);
    }

    constexpr int CalcPointDistance(Point p1, Point p2) {
        return std::sqrt(CalcPointDistanceSq(p1, p2));
    }

    constexpr int CalcSegmentDist(Point p, Segment s) {
        int edge_len_sq = CalcPointDistanceSq(s.Begin(), s.End());
        int p_dist1_sq = CalcPointDistanceSq(p, s.Begin());
        int p_dist2_sq = CalcPointDistanceSq(p, s.End());

        if (p_dist1_sq > edge_len_sq + p_dist2_sq) {
            return std::sqrt(p_dist2_sq);
        }
        if (p_dist2_sq > edge_len_sq + p_dist1_sq) {
            return std::sqrt(p_dist1_sq);
        }

        LineEquation eq_ = s.Equation();
        if (eq_.a * eq_.a + eq_.b * eq_.b == 0) {
            return std::sqrt(p_dist1_sq);
        }
        int dist = abs(s.PointEquation(p)) / (sqrt(eq_.a*eq_.a + eq_.b*eq_.b));
        return dist;
    }

    constexpr bool SegmentIntersection(Segment s1, Segment s2, Point& p) {
        if (Comp(s1, s2) > 0) {
            std::swap(s1, s2);
        } 
        bool cond1 = (s1.A() * s2.B() == s1.B() * s2.A());
        bool cond2 = (s1.A() * s2.C() == s1.C() * s2.A());
        bool cond3 = (s1.B() * s2.C() == s1.C() * s2.B());
        if (cond1 && cond2 && cond3) {
            if (s1.B() == 0 && utils::IsBetween(s2.Begin().y, s1.Begin().y, s1.End().y)) {
                p = s2.Begin();
                return true;
            } else if (s1.B() != 0 && utils::IsBetween(s2.Begin().x, s1.Begin().x, s1.End().x)) {
                p = s2.Begin();
                return true;
            } else {
                return false;
            } 
        } else {
            int s1_c = s1.PointEquation(s2.Begin());
            int s1_d = s1.PointEquation(s2.End());
            int s2_a = s2.PointEquation(s1.Begin());
            int s2_b = s2.PointEquation(s1.End());
            if (s1_c * s1_d <= 0 && s2_a * s2_b <= 0) {
                int vp = s1.A() * s2.B() - s2.A() * s1.B();
                p.x = -(s1.C() * s2.B() - s2.C() * s1.B()) / vp;
                p.y = -(s1.A() * s2.C() - s2.A() * s1.C()) / vp;
                return true;
            } else {
                return false;
            } 
        }
    }

    constexpr int CalcPolygonDist(Segment ray, const BlobGeom& blob) {
        Point intersect;
        int min_dist = 1000;
        for (int i = 0; i < blob.vert_cnt; ++i){
            Segment side(blob.p[i], blob.p[(i + 1) % 4]);
            
            if (SegmentIntersection(ray, side, intersect)){
                int temp = CalcPointDistance(intersect, ray.Begin());
                if (temp < min_dist)
                    min_dist = temp;
            }
        }

        return min_dist;
    }

} // namespace ww_vision
