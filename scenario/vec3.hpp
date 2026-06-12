#pragma once
// Minimal 3-D double vector shared by the scenario geometry layer: world
// positions (m), UE velocities (m/s), and unit directions. Right-handed, +z up
// (Spec G §G.3 world frame). Header-only.

namespace orca {
namespace scenario {

struct Vec3 {
    double x, y, z;
};

inline Vec3 operator+(const Vec3& a, const Vec3& b) {
    return Vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}
inline Vec3 operator-(const Vec3& a, const Vec3& b) {
    return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}
inline Vec3 operator*(const Vec3& a, double s) {
    return Vec3{a.x * s, a.y * s, a.z * s};
}
inline double dot3(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

}  // namespace scenario
}  // namespace orca
