#ifndef _RCJ_WW_LUCKFOX_INCLUDE_ANGLES_HPP_
#define _RCJ_WW_LUCKFOX_INCLUDE_ANGLES_HPP_

#include <cmath>
#include <iostream>

namespace ww_geom {

    static constexpr double kPi = 3.14159265358979323846;
    static constexpr double kDoublePi = 6.28318530717958647692;
    static constexpr double kRadToDeg = 180. / kPi;
    static constexpr double kDegToRad = kPi / 180.;

    // Для Миши:

    // Angles increase clockwise, zero is the Y axis ("up")
    //        0
    //   -45  ^  45
    //      \ | /
    // -90 <-   -> 90
    //      / | \ 
    //  -135  v  135
    //      +-180

    struct Rad;

    struct Deg {
        int deg;

        explicit constexpr Deg(int d) : deg(d) {}
        constexpr operator double() const { return deg * kDegToRad; }
        explicit constexpr operator int() const { return deg; }

        constexpr operator Rad() const;
    };

    struct Rad {
        double rad;

        explicit constexpr Rad(double r) : rad(r) {}
        constexpr operator double() const { return rad; }

        constexpr operator Deg() const {
            return Deg(static_cast<int>(rad * kRadToDeg + (rad > 0 ? 0.5 : -0.5)));
        }
    };

    // Deg operators
    constexpr Deg::operator Rad() const {
        return Rad(deg * kDegToRad);
    }

    constexpr Deg operator+(Deg lhs, Deg rhs) {
        return Deg(lhs.deg + rhs.deg);
    }
    constexpr Deg operator-(Deg lhs, Deg rhs) {
        return Deg(lhs.deg - rhs.deg);
    }
    constexpr Deg operator*(Deg lhs, int scale) {
        return Deg(lhs.deg * scale);
    }
    constexpr Deg operator*(int scale, Deg rhs) {
        return Deg(scale * rhs.deg);
    }
    constexpr Deg operator/(Deg lhs, int div) {
        return Deg(lhs.deg / div);
    }
    constexpr Deg operator-(Deg d) {
        return Deg(-d.deg);
    }

    constexpr bool operator==(Deg lhs, Deg rhs) { return lhs.deg == rhs.deg; }
    constexpr bool operator!=(Deg lhs, Deg rhs) { return !(lhs == rhs); }
    constexpr bool operator< (Deg lhs, Deg rhs) { return lhs.deg <  rhs.deg; }
    constexpr bool operator<=(Deg lhs, Deg rhs) { return lhs.deg <= rhs.deg; }
    constexpr bool operator> (Deg lhs, Deg rhs) { return lhs.deg >  rhs.deg; }
    constexpr bool operator>=(Deg lhs, Deg rhs) { return lhs.deg >= rhs.deg; }

    inline std::ostream& operator<<(std::ostream& os, Deg d) {
        return os << d.deg << " deg";
    }

    constexpr Deg operator""_deg(unsigned long long num) {
        return Deg(static_cast<int>(num));
    }


    // Rad operators
    constexpr Rad operator+(Rad lhs, Rad rhs) {
        return Rad(lhs.rad + rhs.rad);
    }
    constexpr Rad operator-(Rad lhs, Rad rhs) {
        return Rad(lhs.rad - rhs.rad);
    }
    constexpr Rad operator*(Rad lhs, double s) {
        return Rad(lhs.rad * s);
    }
    constexpr Rad operator*(double s, Rad rhs) {
        return Rad(s * rhs.rad);
    }
    constexpr Rad operator/(Rad lhs, double d) {
        return Rad(lhs.rad / d);
    }
    constexpr Rad operator-(Rad r) {
        return Rad(-r.rad);
    }

    constexpr bool operator==(Rad lhs, Rad rhs) { return lhs.rad == rhs.rad; }
    constexpr bool operator!=(Rad lhs, Rad rhs) { return !(lhs == rhs); }
    constexpr bool operator< (Rad lhs, Rad rhs) { return lhs.rad <  rhs.rad; }
    constexpr bool operator<=(Rad lhs, Rad rhs) { return lhs.rad <= rhs.rad; }
    constexpr bool operator> (Rad lhs, Rad rhs) { return lhs.rad >  rhs.rad; }
    constexpr bool operator>=(Rad lhs, Rad rhs) { return lhs.rad >= rhs.rad; }

    inline std::ostream& operator<<(std::ostream& os, Rad r) {
        return os << r.rad << " rad";
    }

    constexpr Rad operator""_rad(unsigned long long num) {
        return Rad(static_cast<double>(num));
    }
    constexpr Rad operator""_rad(long double num) {
        return Rad(static_cast<double>(num));
    }


    constexpr Deg FitAngle(Deg angle) {
        int ang = (angle.deg % 360 + 360) % 360;
        if (ang > 180) {
            ang -= 180;
        }
        return Deg(ang);
    }
    
    constexpr Rad FitAngle(Rad angle) {
        double ang = std::remainder(angle.rad, kDoublePi);
        return Rad(ang);
    }
    
} // namespace ww_geom

#endif
