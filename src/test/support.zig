const std = @import("std");
const testing = std.testing;

const geometry = @import("../geometry.zig");
const support = @import("../support.zig");
const Vector = @import("../vector.zig").Vector;
const Quaternion = @import("../quaternion.zig").Quaternion;

test "supportLocal sphere and box" {
    const Vec3 = Vector(3, f64);
    const Shape = geometry.Shape(f64);

    const sphere = Shape{ .sphere = .{ .radius = 2.0 } };
    const dir = Vec3.fromArray(.{ 1, 2, 3 });
    const sphere_support = support.supportLocal(f64, sphere, dir);
    try testing.expectApproxEqAbs(sphere_support.magnitude(), 2.0, 1e-9);

    const box = Shape{ .box = .{ .half_extents = Vec3.fromArray(.{ 1.0, 2.0, 0.5 }) } };
    const box_support = support.supportLocal(f64, box, Vec3.fromArray(.{ -1, 1, -0.5 }));
    inline for (0..3) |axis| {
        const he = box.box.half_extents.data[axis];
        const expected = if (axis == 0) -he else if (axis == 1) he else -he;
        try testing.expectApproxEqAbs(box_support.data[axis], expected, 1e-12);
    }
}

test "supportScaled reuses base shape" {
    const Vec3 = Vector(3, f64);
    const Shape = geometry.Shape(f64);

    var base_box = Shape{ .box = .{ .half_extents = Vec3.fromArray(.{ 1, 2, 3 }) } };
    const scaled_shape = Shape{ .scaled = .{ .base = &base_box, .scale = Vec3.fromArray(.{ 2, 0.5, 1.0 }) } };
    const dir = Vec3.fromArray(.{ 1, -1, 3 });
    const support_point = support.supportLocal(f64, scaled_shape, dir);
    // Expect base support scaled component-wise
    const base_point = support.supportLocal(f64, base_box, Vec3.fromArray(.{ dir.data[0] / 2.0, dir.data[1] / 0.5, dir.data[2] / 1.0 }));
    var expected = base_point;
    expected.mul(Vec3.fromArray(.{ 2, 0.5, 1.0 }));
    inline for (0..3) |axis| {
        try testing.expectApproxEqAbs(expected.data[axis], support_point.data[axis], 1e-12);
    }
}

test "supportWorld applies pose transform" {
    const Vec3 = Vector(3, f64);
    const Pose = geometry.Pose(f64);
    const Shape = geometry.Shape(f64);

    const sphere = Shape{ .sphere = .{ .radius = 1.0 } };
    const pose = Pose.from(Vec3.fromArray(.{ 10, 0, 0 }), Quaternion(f64).identity());
    const dir_world = Vec3.fromArray(.{ 0, 0, 1 });
    const support_point = support.supportWorld(f64, sphere, pose, dir_world);
    const expected = Vec3.fromArray(.{ 10, 0, 1 });
    inline for (0..3) |axis| {
        try testing.expectApproxEqAbs(support_point.data[axis], expected.data[axis], 1e-9);
    }
}
