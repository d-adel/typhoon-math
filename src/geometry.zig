//! Core geometry primitives: shapes, poses, bounding volumes, and spatial queries.
//!
//! This module provides reusable geometry types for rendering, collision detection,
//! raycasting, culling, and other spatial operations. These types are pure geometry
//! with no physics-specific state (mass, velocity, contact caches, etc.).
//!
//! Generic over the floating-point type T (typically f32 or f64).

const std = @import("std");

const Matrix = @import("matrix.zig").Matrix;
const Quaternion = @import("quaternion.zig").Quaternion;
const Vector = @import("vector.zig").Vector;

pub fn Sphere(comptime T: type) type {
    return struct {
        radius: T,
    };
}

pub fn Box(comptime T: type) type {
    return struct {
        half_extents: Vector(3, T),
    };
}

pub fn Capsule(comptime T: type) type {
    return struct {
        half_height: T,
        radius: T,
    };
}

pub fn Cylinder(comptime T: type) type {
    return struct {
        half_height: T,
        radius: T,
    };
}

pub fn Hull(comptime T: type) type {
    return struct {
        verts: []const Vector(3, T),
    };
}

pub fn Scaled(comptime T: type) type {
    const ShapeT = Shape(T);
    return struct {
        base: *const ShapeT,
        scale: Vector(3, T),
    };
}

pub fn Shape(comptime T: type) type {
    return union(enum) {
        sphere: Sphere(T),
        box: Box(T),
        capsule: Capsule(T),
        cylinder: Cylinder(T),
        hull: Hull(T),
        scaled: Scaled(T),
    };
}

pub fn Pose(comptime T: type) type {
    const Vec3 = Vector(3, T);
    const Mat4 = Matrix(3, 4, T);
    const Quat = Quaternion(T);

    return struct {
        position: Vec3,
        rotation: Quat,
        transform: Mat4,
        inv_transform: Mat4,

        pub fn from(position: Vec3, rotation: Quat) @This() {
            const transform = buildTransform(T, position, rotation);
            var inv = transform;
            inv.invert();
            return .{
                .position = position,
                .rotation = rotation,
                .transform = transform,
                .inv_transform = inv,
            };
        }

        pub inline fn worldToLocalPoint(self: @This(), p: Vec3) Vec3 {
            return self.inv_transform.transform(p);
        }

        pub inline fn localToWorldPoint(self: @This(), p: Vec3) Vec3 {
            return self.transform.transform(p);
        }

        pub inline fn localToWorldDir(self: @This(), v: Vec3) Vec3 {
            return self.transform.transformDirection(v);
        }

        pub inline fn worldToLocalDir(self: @This(), v: Vec3) Vec3 {
            return self.inv_transform.transformDirection(v);
        }
    };
}

fn buildTransform(comptime T: type, position: Vector(3, T), rotation: Quaternion(T)) Matrix(3, 4, T) {
    const w = rotation.data[0];
    const x = rotation.data[1];
    const y = rotation.data[2];
    const z = rotation.data[3];

    const xx = x * x;
    const yy = y * y;
    const zz = z * z;
    const xy = x * y;
    const xz = x * z;
    const yz = y * z;
    const wx = w * x;
    const wy = w * y;
    const wz = w * z;

    return Matrix(3, 4, T).fromArray(.{
        1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy),       position.data[0],
        2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),       position.data[1],
        2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy), position.data[2],
    });
}

pub fn AABB(comptime T: type) type {
    const Vec3 = Vector(3, T);

    return struct {
        lowerBound: Vec3,
        upperBound: Vec3,

        pub inline fn merged(a: @This(), b: @This()) @This() {
            return .{
                .lowerBound = Vec3.min(a.lowerBound, b.lowerBound),
                .upperBound = Vec3.max(a.upperBound, b.upperBound),
            };
        }

        pub inline fn normalized(a: @This()) @This() {
            return .{
                .lowerBound = Vec3.min(a.lowerBound, a.upperBound),
                .upperBound = Vec3.max(a.lowerBound, a.upperBound),
            };
        }

        pub inline fn overlaps(a: @This(), b: @This()) bool {
            const ax0 = a.lowerBound.data[0];
            const ax1 = a.upperBound.data[0];
            const ay0 = a.lowerBound.data[1];
            const ay1 = a.upperBound.data[1];
            const az0 = a.lowerBound.data[2];
            const az1 = a.upperBound.data[2];

            const bx0 = b.lowerBound.data[0];
            const bx1 = b.upperBound.data[0];
            const by0 = b.lowerBound.data[1];
            const by1 = b.upperBound.data[1];
            const bz0 = b.lowerBound.data[2];
            const bz1 = b.upperBound.data[2];

            if (ax1 < bx0 or ax0 > bx1) return false;
            if (ay1 < by0 or ay0 > by1) return false;
            if (az1 < bz0 or az0 > bz1) return false;
            return true;
        }

        pub inline fn contains(outer: @This(), inner: @This()) bool {
            const ax0 = outer.lowerBound.data[0];
            const ax1 = outer.upperBound.data[0];
            const ay0 = outer.lowerBound.data[1];
            const ay1 = outer.upperBound.data[1];
            const az0 = outer.lowerBound.data[2];
            const az1 = outer.upperBound.data[2];

            const bx0 = inner.lowerBound.data[0];
            const bx1 = inner.upperBound.data[0];
            const by0 = inner.lowerBound.data[1];
            const by1 = inner.upperBound.data[1];
            const bz0 = inner.lowerBound.data[2];
            const bz1 = inner.upperBound.data[2];

            return (ax0 <= bx0 and ax1 >= bx1) and (ay0 <= by0 and ay1 >= by1) and (az0 <= bz0 and az1 >= bz1);
        }

        pub inline fn expanded(a: @This(), margin: T) @This() {
            const delta = Vec3.fromArray(.{ margin, margin, margin });
            var lower = a.lowerBound;
            lower.sub(delta);
            var upper = a.upperBound;
            upper.add(delta);
            return .{ .lowerBound = lower, .upperBound = upper };
        }

        pub fn area(a: @This()) T {
            const d: Vec3 = Vec3.subbed(a.upperBound, a.lowerBound);
            return 2.0 * (d.data[0] * d.data[1] + d.data[1] * d.data[2] + d.data[2] * d.data[0]);
        }
    };
}
