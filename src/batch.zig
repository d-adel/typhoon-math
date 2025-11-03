const std = @import("std");
const Vector = @import("vector.zig").Vector;
const Matrix = @import("matrix.zig").Matrix;
const Quaternion = @import("quaternion.zig").Quaternion;

/// Batch SIMD operations for processing multiple vectors/matrices/quaternions simultaneously.
/// Uses Structure-of-Arrays layout for optimal SIMD performance.
/// Default batch size is 4 elements per batch operation.

/// Vec3Batch: Structure-of-Arrays layout for batched 3D vector operations
pub fn Vec3Batch(comptime Batch: usize, comptime T: type) type {
    return struct {
        const Self = @This();
        const SimdVec = @Vector(Batch, T);

        x: SimdVec,
        y: SimdVec,
        z: SimdVec,

        pub inline fn fromVectors(vectors: [Batch]Vector(3, T)) Self {
            var xs: [Batch]T = undefined;
            var ys: [Batch]T = undefined;
            var zs: [Batch]T = undefined;
            inline for (0..Batch) |lane| {
                const v = vectors[lane];
                xs[lane] = v.getUnchecked(0);
                ys[lane] = v.getUnchecked(1);
                zs[lane] = v.getUnchecked(2);
            }
            return .{
                .x = @bitCast(xs),
                .y = @bitCast(ys),
                .z = @bitCast(zs),
            };
        }

        pub inline fn toVectors(self: Self) [Batch]Vector(3, T) {
            const xs: [Batch]T = @bitCast(self.x);
            const ys: [Batch]T = @bitCast(self.y);
            const zs: [Batch]T = @bitCast(self.z);
            var result: [Batch]Vector(3, T) = undefined;
            inline for (0..Batch) |lane| {
                result[lane] = Vector(3, T).fromArray(.{ xs[lane], ys[lane], zs[lane] });
            }
            return result;
        }

        pub inline fn add(a: Self, b: Self) Self {
            return .{
                .x = a.x + b.x,
                .y = a.y + b.y,
                .z = a.z + b.z,
            };
        }

        pub inline fn sub(a: Self, b: Self) Self {
            return .{
                .x = a.x - b.x,
                .y = a.y - b.y,
                .z = a.z - b.z,
            };
        }

        pub inline fn dot(a: Self, b: Self) SimdVec {
            return a.x * b.x + a.y * b.y + a.z * b.z;
        }

        pub inline fn cross(a: Self, b: Self) Self {
            return .{
                .x = a.y * b.z - a.z * b.y,
                .y = a.z * b.x - a.x * b.z,
                .z = a.x * b.y - a.y * b.x,
            };
        }

        pub inline fn normalize(v: Self) Self {
            const epsilon: SimdVec = @splat(@as(T, 1e-12));
            const zero: SimdVec = @splat(@as(T, 0.0));
            const one: SimdVec = @splat(@as(T, 1.0));
            const len_sq = dot(v, v);
            const safe_len_sq = @max(len_sq, epsilon);
            const inv_len = one / @sqrt(safe_len_sq);
            const mask = len_sq > epsilon;
            const selected = @select(T, mask, inv_len, zero);
            return .{
                .x = v.x * selected,
                .y = v.y * selected,
                .z = v.z * selected,
            };
        }

        pub inline fn lengthSquared(v: Self) SimdVec {
            return dot(v, v);
        }

        pub inline fn length(v: Self) SimdVec {
            return @sqrt(lengthSquared(v));
        }

        pub inline fn mulScalar(v: Self, scalar: SimdVec) Self {
            return .{
                .x = v.x * scalar,
                .y = v.y * scalar,
                .z = v.z * scalar,
            };
        }

        pub inline fn negate(v: Self) Self {
            const neg_one: SimdVec = @splat(@as(T, -1.0));
            return mulScalar(v, neg_one);
        }

        pub inline fn addScaled(v: Self, other: Self, scalar: SimdVec) Self {
            const scaled = mulScalar(other, scalar);
            return add(v, scaled);
        }

        pub inline fn reduceBits(v: SimdVec) u32 {
            const bits: @Vector(Batch, u32) = @bitCast(v);
            return @reduce(.Add, bits);
        }

        pub inline fn reduceComponent(self: Self, lane_index: usize) u32 {
            const component = switch (lane_index % 3) {
                0 => self.x,
                1 => self.y,
                else => self.z,
            };
            return reduceBits(component);
        }
    };
}

/// Mat3Batch: Structure-of-Arrays layout for batched 3x3 matrix operations
pub fn Mat3Batch(comptime Batch: usize, comptime T: type) type {
    return struct {
        const Self = @This();
        const SimdVec = @Vector(Batch, T);

        m00: SimdVec,
        m01: SimdVec,
        m02: SimdVec,
        m10: SimdVec,
        m11: SimdVec,
        m12: SimdVec,
        m20: SimdVec,
        m21: SimdVec,
        m22: SimdVec,

        pub inline fn fromMatrices(matrices: [Batch]Matrix(3, 3, T)) Self {
            var columns: [9][Batch]T = undefined;
            inline for (0..Batch) |lane| {
                const mat = matrices[lane];
                columns[0][lane] = mat.get1D(0);
                columns[1][lane] = mat.get1D(1);
                columns[2][lane] = mat.get1D(2);
                columns[3][lane] = mat.get1D(3);
                columns[4][lane] = mat.get1D(4);
                columns[5][lane] = mat.get1D(5);
                columns[6][lane] = mat.get1D(6);
                columns[7][lane] = mat.get1D(7);
                columns[8][lane] = mat.get1D(8);
            }
            return .{
                .m00 = @bitCast(columns[0]),
                .m01 = @bitCast(columns[1]),
                .m02 = @bitCast(columns[2]),
                .m10 = @bitCast(columns[3]),
                .m11 = @bitCast(columns[4]),
                .m12 = @bitCast(columns[5]),
                .m20 = @bitCast(columns[6]),
                .m21 = @bitCast(columns[7]),
                .m22 = @bitCast(columns[8]),
            };
        }

        pub inline fn mul(a: Self, b: Self) Self {
            return .{
                .m00 = a.m00 * b.m00 + a.m01 * b.m10 + a.m02 * b.m20,
                .m01 = a.m00 * b.m01 + a.m01 * b.m11 + a.m02 * b.m21,
                .m02 = a.m00 * b.m02 + a.m01 * b.m12 + a.m02 * b.m22,
                .m10 = a.m10 * b.m00 + a.m11 * b.m10 + a.m12 * b.m20,
                .m11 = a.m10 * b.m01 + a.m11 * b.m11 + a.m12 * b.m21,
                .m12 = a.m10 * b.m02 + a.m11 * b.m12 + a.m12 * b.m22,
                .m20 = a.m20 * b.m00 + a.m21 * b.m10 + a.m22 * b.m20,
                .m21 = a.m20 * b.m01 + a.m21 * b.m11 + a.m22 * b.m21,
                .m22 = a.m20 * b.m02 + a.m21 * b.m12 + a.m22 * b.m22,
            };
        }

        pub inline fn mulVec(m: Self, v: Vec3Batch(Batch, T)) Vec3Batch(Batch, T) {
            return .{
                .x = m.m00 * v.x + m.m01 * v.y + m.m02 * v.z,
                .y = m.m10 * v.x + m.m11 * v.y + m.m12 * v.z,
                .z = m.m20 * v.x + m.m21 * v.y + m.m22 * v.z,
            };
        }

        pub inline fn reduceBits(v: SimdVec) u32 {
            const bits: @Vector(Batch, u32) = @bitCast(v);
            return @reduce(.Add, bits);
        }

        pub inline fn reduceComponent(self: Self, lane: usize) u32 {
            const components = switch (lane % 9) {
                0 => self.m00,
                1 => self.m01,
                2 => self.m02,
                3 => self.m10,
                4 => self.m11,
                5 => self.m12,
                6 => self.m20,
                7 => self.m21,
                else => self.m22,
            };
            return reduceBits(components);
        }
    };
}

/// QuatBatch: Structure-of-Arrays layout for batched quaternion operations
pub fn QuatBatch(comptime Batch: usize, comptime T: type) type {
    return struct {
        const Self = @This();
        const SimdVec = @Vector(Batch, T);

        w: SimdVec,
        x: SimdVec,
        y: SimdVec,
        z: SimdVec,

        pub inline fn fromQuaternions(quaternions: [Batch]Quaternion(T)) Self {
            var ws: [Batch]T = undefined;
            var xs: [Batch]T = undefined;
            var ys: [Batch]T = undefined;
            var zs: [Batch]T = undefined;
            inline for (0..Batch) |lane| {
                const q = quaternions[lane];
                ws[lane] = q.w();
                xs[lane] = q.x();
                ys[lane] = q.y();
                zs[lane] = q.z();
            }
            return .{
                .w = @bitCast(ws),
                .x = @bitCast(xs),
                .y = @bitCast(ys),
                .z = @bitCast(zs),
            };
        }

        pub inline fn mul(a: Self, b: Self) Self {
            const a_wwww = a.w;
            const b_wwww = b.w;

            const a_zxyy_w = a.z;
            const a_zxyy_x = a.x;
            const a_zxyy_y = a.y;
            const a_zxyy_z = a.y;

            const b_yxzz_w = b.y;
            const b_yxzz_x = b.x;
            const b_yxzz_y = b.z;
            const b_yxzz_z = b.z;

            const a_yzxx_w = a.y;
            const a_yzxx_x = a.z;
            const a_yzxx_y = a.x;
            const a_yzxx_z = a.x;

            const b_zyxx_w = b.z;
            const b_zyxx_x = b.y;
            const b_zyxx_y = b.x;
            const b_zyxx_z = b.x;

            const t1_w = a_wwww * b.w;
            const t1_x = a_wwww * b.x;
            const t1_y = a_wwww * b.y;
            const t1_z = a_wwww * b.z;

            const t2_w = b_wwww * a.w;
            const t2_x = b_wwww * a.x;
            const t2_y = b_wwww * a.y;
            const t2_z = b_wwww * a.z;

            const t3_w = a_zxyy_w * b_yxzz_w;
            const t3_x = a_zxyy_x * b_yxzz_x;
            const t3_y = a_zxyy_y * b_yxzz_y;
            const t3_z = a_zxyy_z * b_yxzz_z;

            const t4_w = a_yzxx_w * b_zyxx_w;
            const t4_x = a_yzxx_x * b_zyxx_x;
            const t4_y = a_yzxx_y * b_zyxx_y;
            const t4_z = a_yzxx_z * b_zyxx_z;

            const res_w = t1_w + t2_w + (t3_w - t4_w) * @as(SimdVec, @splat(-1.0));
            const res_x = t1_x + t2_x + (t3_x - t4_x) * @as(SimdVec, @splat(1.0));
            const res_y = t1_y + t2_y + (t3_y - t4_y) * @as(SimdVec, @splat(1.0));
            const res_z = t1_z + t2_z + (t3_z - t4_z) * @as(SimdVec, @splat(1.0));

            return .{ .w = res_w, .x = res_x, .y = res_y, .z = res_z };
        }

        pub inline fn rotate(q: Self, v: Vec3Batch(Batch, T)) Vec3Batch(Batch, T) {
            const q_vec = Vec3Batch(Batch, T){ .x = q.x, .y = q.y, .z = q.z };
            const uv = Vec3Batch(Batch, T).cross(q_vec, v);
            const uuv = Vec3Batch(Batch, T).cross(q_vec, uv);
            const two: SimdVec = @splat(@as(T, 2.0));
            const w_scale = q.w * two;
            return Vec3Batch(Batch, T).add(Vec3Batch(Batch, T).add(v, .{
                .x = uv.x * w_scale,
                .y = uv.y * w_scale,
                .z = uv.z * w_scale,
            }), .{
                .x = uuv.x * two,
                .y = uuv.y * two,
                .z = uuv.z * two,
            });
        }

        pub inline fn normalize(q: Self) Self {
            const epsilon: SimdVec = @splat(@as(T, 1e-12));
            const zero: SimdVec = @splat(@as(T, 0.0));
            const one: SimdVec = @splat(@as(T, 1.0));
            const len_sq = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
            const safe_len_sq = @max(len_sq, epsilon);
            const inv_len = one / @sqrt(safe_len_sq);
            const mask = len_sq > epsilon;
            const selected = @select(T, mask, inv_len, zero);
            return .{
                .w = q.w * selected,
                .x = q.x * selected,
                .y = q.y * selected,
                .z = q.z * selected,
            };
        }

        pub inline fn conjugate(q: Self) Self {
            const neg_one: SimdVec = @splat(@as(T, -1.0));
            return .{
                .w = q.w,
                .x = q.x * neg_one,
                .y = q.y * neg_one,
                .z = q.z * neg_one,
            };
        }

        pub inline fn reduceBits(v: SimdVec) u32 {
            const bits: @Vector(Batch, u32) = @bitCast(v);
            return @reduce(.Add, bits);
        }
    };
}
