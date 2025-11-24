//! Support mapping functions for collision detection (GJK/EPA).
//!
//! Support functions map a direction vector to the farthest point on a shape
//! in that direction. These are fundamental primitives for convex collision
//! detection algorithms like GJK (Gilbert-Johnson-Keerthi) and EPA (Expanding
//! Polytope Algorithm).

const std = @import("std");

const geometry = @import("geometry.zig");
const Vector = @import("vector.zig").Vector;

// ============================================================================
// Public API (Generic)
// ============================================================================

pub fn supportWorld(comptime T: type, shape: geometry.Shape(T), pose: geometry.Pose(T), dir_world: Vector(3, T)) Vector(3, T) {
    const Vec3 = Vector(3, T);
    if (dir_world.magnitudeSq() == 0) {
        return pose.localToWorldPoint(Vec3.zero());
    }
    const dir_local = pose.worldToLocalDir(dir_world);
    const local_point = supportLocal(T, shape, dir_local);
    return pose.localToWorldPoint(local_point);
}

pub fn supportLocal(comptime T: type, shape: geometry.Shape(T), dir: Vector(3, T)) Vector(3, T) {
    return switch (shape) {
        .sphere => |s| supportSphere(T, s, dir),
        .box => |b| supportBox(T, b, dir),
        .capsule => |c| supportCapsule(T, c, dir),
        .cylinder => |c| supportCylinder(T, c, dir),
        .tapered_capsule => |c| supportTaperedCapsule(T, c, dir),
        .tapered_cylinder => |c| supportTaperedCylinder(T, c, dir),
        .hull => |h| supportHull(T, h, dir),
        .mesh => unreachable,
        .heightfield => unreachable,
        .scaled => |s| supportScaled(T, s, dir),
        .compound => |c| supportCompoundChildren(T, c.children, dir),
        .mutable_compound => |c| supportCompoundChildren(T, c.children, dir),
    };
}

// ============================================================================
// Shape-Specific Support Functions
// ============================================================================

fn safeNormalize(comptime T: type, vec_in: Vector(3, T)) Vector(3, T) {
    const len = vec_in.magnitude();
    if (len <= 1e-12) return Vector(3, T).zero();
    var v = vec_in;
    v.mulScalar(1.0 / len);
    return v;
}

fn supportSphere(comptime T: type, s: geometry.Sphere(T), dir: Vector(3, T)) Vector(3, T) {
    const n = safeNormalize(T, dir);
    var scaled = n;
    scaled.mulScalar(s.radius);
    return scaled;
}

fn supportBox(comptime T: type, b: geometry.Box(T), dir: Vector(3, T)) Vector(3, T) {
    const Vec3 = Vector(3, T);
    var result = Vec3.zero();
    const he = b.half_extents;
    result.data[0] = if (dir.data[0] >= 0) he.data[0] else -he.data[0];
    result.data[1] = if (dir.data[1] >= 0) he.data[1] else -he.data[1];
    result.data[2] = if (dir.data[2] >= 0) he.data[2] else -he.data[2];
    return result;
}

fn supportCapsule(comptime T: type, c: geometry.Capsule(T), dir: Vector(3, T)) Vector(3, T) {
    const Vec3 = Vector(3, T);
    const axis_point = Vec3.fromArray(.{ 0.0, if (dir.data[1] >= 0.0) c.half_height else -c.half_height, 0.0 });
    const n = safeNormalize(T, dir);
    var radial = n;
    radial.mulScalar(c.radius);
    var point = axis_point;
    point.add(radial);
    return point;
}

fn supportCylinder(comptime T: type, c: geometry.Cylinder(T), dir: Vector(3, T)) Vector(3, T) {
    const Vec3 = Vector(3, T);
    const half_height = c.half_height;
    const radius = c.radius;

    const axial = if (dir.data[1] >= 0.0) half_height else -half_height;
    const planar = Vec3.fromArray(.{ dir.data[0], 0.0, dir.data[2] });
    const planar_len = planar.magnitude();

    var px: T = 0;
    var pz: T = 0;
    if (planar_len > 1e-12) {
        const scale = radius / planar_len;
        px = planar.data[0] * scale;
        pz = planar.data[2] * scale;
    }

    return Vec3.fromArray(.{ px, axial, pz });
}

fn supportTaperedCylinder(comptime T: type, c: geometry.TaperedCylinder(T), dir: Vector(3, T)) Vector(3, T) {
    const Vec3 = Vector(3, T);
    const planar_len = @sqrt(dir.data[0] * dir.data[0] + dir.data[2] * dir.data[2]);

    const top_y = c.half_height;
    const bottom_y = -c.half_height;

    var top_point = Vec3.fromArray(.{ 0.0, top_y, 0.0 });
    var bottom_point = Vec3.fromArray(.{ 0.0, bottom_y, 0.0 });

    if (planar_len > 1e-12) {
        const top_scale = c.top_radius / planar_len;
        top_point.data[0] = dir.data[0] * top_scale;
        top_point.data[2] = dir.data[2] * top_scale;

        const bottom_scale = c.bottom_radius / planar_len;
        bottom_point.data[0] = dir.data[0] * bottom_scale;
        bottom_point.data[2] = dir.data[2] * bottom_scale;
    }

    const top_dot = Vec3.dot(top_point, dir);
    const bottom_dot = Vec3.dot(bottom_point, dir);
    return if (top_dot >= bottom_dot) top_point else bottom_point;
}

fn supportTaperedCapsule(comptime T: type, c: geometry.TaperedCapsule(T), dir: Vector(3, T)) Vector(3, T) {
    const Vec3 = Vector(3, T);
    const n = safeNormalize(T, dir);

    const top_center = Vec3.fromArray(.{ 0.0, c.half_height, 0.0 });
    var top = top_center;
    var scaled_top = n;
    scaled_top.mulScalar(c.top_radius);
    top.add(scaled_top);

    const bottom_center = Vec3.fromArray(.{ 0.0, -c.half_height, 0.0 });
    var bottom = bottom_center;
    var scaled_bottom = n;
    scaled_bottom.mulScalar(c.bottom_radius);
    bottom.add(scaled_bottom);

    const side = supportTaperedCylinder(T, .{
        .half_height = c.half_height,
        .top_radius = c.top_radius,
        .bottom_radius = c.bottom_radius,
    }, dir);

    const top_dot = Vec3.dot(top, dir);
    const bottom_dot = Vec3.dot(bottom, dir);
    const side_dot = Vec3.dot(side, dir);

    var best = top;
    var best_dot = top_dot;
    if (bottom_dot > best_dot) {
        best = bottom;
        best_dot = bottom_dot;
    }
    if (side_dot > best_dot) best = side;
    return best;
}

fn supportHull(comptime T: type, h: geometry.Hull(T), dir: Vector(3, T)) Vector(3, T) {
    const Vec3 = Vector(3, T);
    if (h.verts.len == 0) return Vec3.zero();
    var best_index: usize = 0;
    var best_dot: T = -std.math.inf(T);
    for (h.verts, 0..) |v, idx| {
        const d = Vec3.dot(v, dir);
        if (d > best_dot) {
            best_dot = d;
            best_index = idx;
        }
    }
    return h.verts[best_index];
}

fn supportScaled(comptime T: type, s: geometry.Scaled(T), dir: Vector(3, T)) Vector(3, T) {
    const Vec3 = Vector(3, T);
    const scale = s.scale;
    const inv_dir = Vec3.fromArray(.{
        if (@abs(scale.data[0]) > 1e-12) dir.data[0] / scale.data[0] else 0.0,
        if (@abs(scale.data[1]) > 1e-12) dir.data[1] / scale.data[1] else 0.0,
        if (@abs(scale.data[2]) > 1e-12) dir.data[2] / scale.data[2] else 0.0,
    });
    const base_support = supportLocal(T, s.base.*, inv_dir);
    var point = base_support;
    _ = point.mul(scale);
    return point;
}

fn supportCompoundChildren(comptime T: type, children: []const geometry.CompoundChild(T), dir: Vector(3, T)) Vector(3, T) {
    const Vec3 = Vector(3, T);
    if (children.len == 0) return Vec3.zero();

    var best = Vec3.zero();
    var best_dot: T = -std.math.inf(T);
    for (children) |child| {
        const local_dir = child.pose.worldToLocalDir(dir);
        const local_support = supportLocal(T, child.shape.*, local_dir);
        const world_support = child.pose.localToWorldPoint(local_support);
        const dot_val = Vec3.dot(world_support, dir);
        if (dot_val > best_dot) {
            best_dot = dot_val;
            best = world_support;
        }
    }
    return best;
}
