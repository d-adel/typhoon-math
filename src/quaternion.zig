const std = @import("std");

const Vector = @import("vector.zig").Vector;

/// Quaternion for 3D rotations using SIMD operations.
/// Data layout: [w, x, y, z] where w is the scalar part.
pub fn Quaternion(comptime T: type) type {
    return struct {
        const Self = @This();
        data: @Vector(4, T),

        inline fn mutData(self: *Self) *[4]T {
            return @ptrCast(&self.data);
        }

        inline fn valueArray(self: Self) [4]T {
            return @bitCast(self.data);
        }

        const W: usize = 0;
        const X: usize = 1;
        const Y: usize = 2;
        const Z: usize = 3;

        // ========== Construction ==========

        pub inline fn fromArray(values: [4]T) Self {
            const info = @typeInfo(T);
            switch (info) {
                .bool => unreachable,
                else => return .{ .data = @bitCast(values) },
            }
        }

        pub inline fn fromSlice(slice: []const T) Self {
            if (slice.len != 4) @panic("fromSlice: wrong length");
            var tmp: [4]T = undefined;
            inline for (0..4) |i| tmp[i] = slice[i];
            return .{ .data = @bitCast(tmp) };
        }

        pub inline fn fromPtrArray(p: *const [4]T) Self {
            return .{ .data = @bitCast(p.*) };
        }

        pub inline fn fromManyPtr(p: [*]const T) Self {
            var tmp: [4]T = undefined;
            inline for (0..4) |i| tmp[i] = p[i];
            return .{ .data = @bitCast(tmp) };
        }

        pub inline fn fromSequence(seq: anytype) Self {
            const SeqT = @TypeOf(seq);
            const info = @typeInfo(SeqT);
            switch (info) {
                .array => |ai| {
                    if (ai.len != 4 or ai.child != T)
                        @compileError("fromSequence: expected [4]" ++ @typeName(T));
                    return Self.fromArray(seq);
                },
                .pointer => |pi| switch (pi.size) {
                    .one => {
                        const ci = @typeInfo(pi.child);
                        if (ci == .array and ci.array.len == 4 and ci.array.child == T)
                            return Self.fromPtrArray(seq);
                        @compileError("fromSequence: expected *[4]" ++ @typeName(T));
                    },
                    .many => return Self.fromManyPtr(seq),
                    .slice => return Self.fromSlice(seq),
                    .c => @compileError("fromSequence: C pointer unsupported; use fromManyPtr"),
                },
                else => @compileError("fromSequence: unsupported input type"),
            }
        }

        pub inline fn identity() Self {
            return .{ .data = .{ @as(T, 1), @as(T, 0), @as(T, 0), @as(T, 0) } };
        }

        pub inline fn zero() Self {
            return .{ .data = @bitCast([4]T{ @as(T, 0), @as(T, 0), @as(T, 0), @as(T, 0) }) };
        }

        pub inline fn fromAxisAngle(axis: Vector(3, T), angle: T) Self {
            const half_angle = angle * 0.5;
            const s = @sin(half_angle);
            const c = @cos(half_angle);
            return .{
                .data = .{
                    c,
                    axis.data[0] * s,
                    axis.data[1] * s,
                    axis.data[2] * s,
                },
            };
        }

        // ========== Element Access ==========

        pub inline fn w(self: Self) T {
            return self.data[W];
        }

        pub inline fn x(self: Self) T {
            return self.data[X];
        }

        pub inline fn y(self: Self) T {
            return self.data[Y];
        }

        pub inline fn z(self: Self) T {
            return self.data[Z];
        }

        // ========== Magnitude ==========

        pub inline fn magnitudeSq(self: Self) T {
            const arr = self.valueArray();
            return dot4(arr, arr);
        }

        pub inline fn normalize(self: *Self) void {
            const d = self.magnitudeSq();
            if (d == 0) {
                self.* = Self.identity();
                return;
            }
            const inv_len = @as(T, 1) / @sqrt(d);
            const dst = self.mutData();
            inline for (0..4) |i| dst.*[i] *= inv_len;
        }

        // ========== Multiplication ==========

        pub inline fn mul(self: Self, b: Self) Self {
            // SIMD quaternion multiplication using the Hamilton product formula:
            //   rw = aw*bw - ax*bx - ay*by - az*bz
            //   rx = aw*bx + ax*bw + ay*bz - az*by
            //   ry = aw*by - ax*bz + ay*bw + az*bx
            //   rz = aw*bz + ax*by - ay*bx + az*bw
            //
            // Use @mulAdd for FMA instructions where available, and let the
            // compiler auto-vectorize. Explicit scalar form is clearer and
            // easier to verify correctness.

            const aw = self.data[W];
            const ax = self.data[X];
            const ay = self.data[Y];
            const az = self.data[Z];

            const bw = b.data[W];
            const bx = b.data[X];
            const by = b.data[Y];
            const bz = b.data[Z];

            // Use @mulAdd for FMA optimization
            const rw = aw * bw - ax * bx - ay * by - az * bz;
            const rx = @mulAdd(T, ay, bz, @mulAdd(T, ax, bw, aw * bx)) - az * by;
            const ry = @mulAdd(T, az, bx, @mulAdd(T, ay, bw, aw * by)) - ax * bz;
            const rz = @mulAdd(T, ax, by, @mulAdd(T, az, bw, aw * bz)) - ay * bx;

            return .{ .data = .{ rw, rx, ry, rz } };
        }

        pub inline fn mulAssign(self: *Self, b: Self) void {
            self.* = self.mul(b);
        }

        // ========== Rotation Operations ==========

        pub inline fn rotateByVector(self: *Self, v: Vector(3, T)) void {
            var qtmp = Self.fromArray(.{ @as(T, 0), v.data[0], v.data[1], v.data[2] });
            qtmp.mulAssign(self.*);
            const half: T = 0.5;
            self.data[W] += qtmp.data[W] * half;
            self.data[X] += qtmp.data[X] * half;
            self.data[Y] += qtmp.data[Y] * half;
            self.data[Z] += qtmp.data[Z] * half;
        }

        pub inline fn addScaledVector(self: *Self, v: Vector(3, T), scale: T) void {
            var qtmp = Self.fromArray(.{
                @as(T, 0),
                v.data[0] * scale,
                v.data[1] * scale,
                v.data[2] * scale,
            });
            qtmp.mulAssign(self.*);
            const half: T = 0.5;
            self.data[W] += qtmp.data[W] * half;
            self.data[X] += qtmp.data[X] * half;
            self.data[Y] += qtmp.data[Y] * half;
            self.data[Z] += qtmp.data[Z] * half;
        }

        pub inline fn rotate(self: Self, v: Vector(3, T)) Vector(3, T) {
            const Vec3 = Vector(3, T);
            const q_vec = Vec3.fromArray(.{ self.data[X], self.data[Y], self.data[Z] });
            var uv = Vec3.cross(q_vec, v);
            var uuv = Vec3.cross(q_vec, uv);
            const two = @as(T, 2);
            uv.mulScalar(self.data[W] * two);
            uuv.mulScalar(two);
            var result = v;
            result.add(uv);
            result.add(uuv);
            return result;
        }

        // ========== Conjugate and Inverse ==========

        pub inline fn conjugate(self: *Self) void {
            self.data[X] = -self.data[X];
            self.data[Y] = -self.data[Y];
            self.data[Z] = -self.data[Z];
        }

        pub inline fn inverse(self: *Self) void {
            const m2 = self.magnitudeSq();
            if (m2 == 0) {
                self.* = Self.identity();
                return;
            }
            self.conjugate();
            const inv = @as(T, 1) / m2;
            const dst = self.mutData();
            inline for (0..4) |i| dst.*[i] *= inv;
        }

        pub inline fn nlerp(a: Self, b: Self, t: T) Self {
            const one_minus_t = @as(T, 1) - t;
            const lhs = a.valueArray();
            const rhs = b.valueArray();
            var result: [4]T = undefined;
            inline for (0..4) |i| result[i] = lhs[i] * one_minus_t + rhs[i] * t;
            var q = Self.fromArray(result);
            q.normalize();
            return q;
        }

        pub inline fn slerp(a: Self, b: Self, t: T) Self {
            var cos_theta = dot4(a.valueArray(), b.valueArray());

            if (cos_theta > @as(T, 0.9995)) {
                return nlerp(a, b, t);
            }

            var b_adjusted = b;
            if (cos_theta < 0) {
                const dst = b_adjusted.mutData();
                inline for (0..4) |i| dst.*[i] = -dst.*[i];
                cos_theta = -cos_theta;
            }

            if (cos_theta > @as(T, 1)) cos_theta = @as(T, 1);

            const theta = std.math.acos(cos_theta);
            const sin_theta = @sin(theta);

            if (@abs(sin_theta) < @as(T, 0.001)) {
                return nlerp(a, b_adjusted, t);
            }

            const ratio_a = @sin((@as(T, 1) - t) * theta) / sin_theta;
            const ratio_b = @sin(t * theta) / sin_theta;

            const lhs = a.valueArray();
            const rhs = b_adjusted.valueArray();
            var result: [4]T = undefined;
            inline for (0..4) |i| result[i] = lhs[i] * ratio_a + rhs[i] * ratio_b;
            return .{ .data = @bitCast(result) };
        }

        pub inline fn angleBetween(a: Self, b: Self) T {
            var dot_product = dot4(a.valueArray(), b.valueArray());
            if (dot_product > @as(T, 1)) dot_product = @as(T, 1);
            if (dot_product < @as(T, -1)) dot_product = @as(T, -1);
            return @as(T, 2) * std.math.acos(@abs(dot_product));
        }

        inline fn dot4(a_vals: [4]T, b_vals: [4]T) T {
            var sum: T = 0;
            inline for (0..4) |i| sum = @mulAdd(T, a_vals[i], b_vals[i], sum);
            return sum;
        }
    };
}
