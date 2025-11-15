const std = @import("std");
const testing = std.testing;

const geometry = @import("../geometry.zig");
const Vector = @import("../vector.zig").Vector;
const Quaternion = @import("../quaternion.zig").Quaternion;

test "Pose world/local conversions round-trip" {
    const Vec3 = Vector(3, f64);
    const Pose = geometry.Pose(f64);

    var q = Quaternion(f64).fromArray(.{ 0.8, 0.2, -0.1, 0.5 });
    q.normalize();
    const pos = Vec3.fromArray(.{ 3.0, -2.5, 1.25 });
    const pose = Pose.from(pos, q);

    const local_point = Vec3.fromArray(.{ 0.5, -1.0, 2.0 });
    const world_point = pose.localToWorldPoint(local_point);
    const back_point = pose.worldToLocalPoint(world_point);
    inline for (0..3) |axis| {
        try testing.expectApproxEqAbs(local_point.data[axis], back_point.data[axis], 1e-9);
    }

    const local_dir = Vec3.fromArray(.{ -0.2, 0.8, 0.1 });
    const world_dir = pose.localToWorldDir(local_dir);
    const back_dir = pose.worldToLocalDir(world_dir);
    inline for (0..3) |axis| {
        try testing.expectApproxEqAbs(local_dir.data[axis], back_dir.data[axis], 1e-9);
    }
}

test "AABB normalized merge and contains" {
    const Vec3 = Vector(3, f64);
    const AABB = geometry.AABB(f64);

    const raw = AABB{
        .lowerBound = Vec3.fromArray(.{ 5, 3, -2 }),
        .upperBound = Vec3.fromArray(.{ 1, 8, 4 }),
    };
    const normalized = raw.normalized();
    try testing.expect(normalized.lowerBound.data[0] <= normalized.upperBound.data[0]);
    try testing.expect(normalized.lowerBound.data[1] <= normalized.upperBound.data[1]);
    try testing.expect(normalized.lowerBound.data[2] <= normalized.upperBound.data[2]);

    const other = AABB{
        .lowerBound = Vec3.fromArray(.{ -1, 0, -3 }),
        .upperBound = Vec3.fromArray(.{ 2, 2, -1 }),
    };
    const merged = normalized.merged(other);
    try testing.expect(merged.contains(normalized));
    try testing.expect(merged.contains(other));
    try testing.expect(merged.overlaps(other));
}

test "Pose transform aligns with matrix determinant" {
    const Vec3 = Vector(3, f64);
    const Pose = geometry.Pose(f64);
    const Quaternionf = Quaternion(f64);

    const rotation = Quaternionf.fromArray(.{ 0.3, 0.4, 0.5, 0.2 });
    var q = rotation;
    q.normalize();
    const pose = Pose.from(Vec3.zero(), q);
    const det = pose.transform.determinant();
    try testing.expectApproxEqAbs(det, 1.0, 1e-9);
}
