const std = @import("std");

/// Generic N-dimensional vector using SIMD operations.
/// Provides common vector operations with optimal SIMD performance.
pub fn Vector(comptime N: usize, comptime T: type) type {
    return struct {
        const Self = @This();
        data: @Vector(N, T),

        // ========== Construction ==========

        pub inline fn fromArray(values: [N]T) Self {
            const info = @typeInfo(T);
            switch (info) {
                .bool => unreachable,
                else => return .{ .data = @bitCast(values) },
            }
        }

        pub inline fn fromSlice(slice: []const T) Self {
            if (slice.len != N) @panic("fromSlice: wrong length");
            var tmp: [N]T = undefined;
            inline for (0..N) |i| tmp[i] = slice[i];
            return .{ .data = @bitCast(tmp) };
        }

        pub inline fn fromPtrArray(p: *const [N]T) Self {
            return .{ .data = @bitCast(p.*) };
        }

        pub inline fn fromManyPtr(p: [*]const T) Self {
            var tmp: [N]T = undefined;
            inline for (0..N) |i| tmp[i] = p[i];
            return .{ .data = @bitCast(tmp) };
        }

        pub inline fn fromSequence(seq: anytype) Self {
            const SeqT = @TypeOf(seq);
            const info = @typeInfo(SeqT);
            switch (info) {
                .array => |ai| {
                    if (ai.len != N or ai.child != T)
                        @compileError("fromSequence: expected [N]" ++ @typeName(T));
                    return Self.fromArray(seq);
                },

                .pointer => |pi| switch (pi.size) {
                    .one => {
                        const ci = @typeInfo(pi.child);
                        if (ci == .array and ci.array.len == N and ci.array.child == T)
                            return Self.fromPtrArray(seq);
                        @compileError("fromSequence: expected *[N]" ++ @typeName(T));
                    },
                    .many => return Self.fromManyPtr(seq),
                    .slice => return Self.fromSlice(seq),
                    .c => @compileError("fromSequence: C pointer unsupported; use fromManyPtr"),
                },

                else => @compileError("fromSequence: unsupported input type"),
            }
        }

        pub inline fn zero() Self {
            return .{ .data = @splat(@as(T, 0)) };
        }

        // ========== Element Access ==========

        pub inline fn get(self: Self, idx: usize) T {
            const tmp: [N]T = @bitCast(self.data);
            return tmp[idx];
        }

        pub inline fn set(self: *Self, idx: usize, value: T) void {
            var tmp: [N]T = @bitCast(self.data);
            tmp[idx] = value;
            self.data = @bitCast(tmp);
        }

        // ========== Basic Operations (Mutating) ==========

        pub inline fn clear(self: *Self) void {
            self.data = @splat(@as(T, 0));
        }

        pub inline fn add(self: *Self, other: Self) void {
            self.data += other.data;
        }

        pub inline fn sub(self: *Self, other: Self) void {
            self.data -= other.data;
        }

        pub inline fn mul(self: *Self, other: Self) void {
            self.data *= other.data;
        }

        pub inline fn mulScalar(self: *Self, k: T) void {
            self.data *= @as(@TypeOf(self.data), @splat(k));
        }

        pub inline fn negate(self: *Self) void {
            self.data = -self.data;
        }

        pub inline fn normalize(self: *Self) void {
            const mag = self.magnitude();
            if (mag == 0) return;
            const inv = @as(T, 1) / mag;
            self.data *= @as(@TypeOf(self.data), @splat(inv));
        }

        // ========== Basic Operations (Non-mutating) ==========

        pub inline fn added(a: Self, b: Self) Self {
            return .{ .data = a.data + b.data };
        }

        pub inline fn subbed(a: Self, b: Self) Self {
            return .{ .data = a.data - b.data };
        }

        pub inline fn scaled(a: Self, k: T) Self {
            return .{ .data = a.data * @as(@TypeOf(a.data), @splat(k)) };
        }

        pub inline fn normalized(a: Self) Self {
            const mag = a.magnitude();
            if (mag == 0) return a;
            const inv = @as(T, 1) / mag;
            return .{ .data = a.data * @as(@TypeOf(a.data), @splat(inv)) };
        }

        // ========== Vector Products ==========

        pub inline fn dot(a: Self, b: Self) T {
            return @reduce(.Add, a.data * b.data);
        }

        pub inline fn cross(a: Self, b: Self) Self {
            comptime {
                if (N != 3) @compileError("cross is only defined for 3D vectors");
            }
            const ax = a.data[0];
            const ay = a.data[1];
            const az = a.data[2];
            const bx = b.data[0];
            const by = b.data[1];
            const bz = b.data[2];
            return Self.fromArray(.{
                ay * bz - az * by,
                az * bx - ax * bz,
                ax * by - ay * bx,
            });
        }

        // ========== Magnitude ==========

        pub inline fn magnitude(a: Self) T {
            return @sqrt(@reduce(.Add, a.data * a.data));
        }

        pub inline fn magnitudeSq(a: Self) T {
            return @reduce(.Add, a.data * a.data);
        }

        // ========== Comparison ==========

        pub inline fn min(a: Self, b: Self) Self {
            return .{ .data = @min(a.data, b.data) };
        }

        pub inline fn max(a: Self, b: Self) Self {
            return .{ .data = @max(a.data, b.data) };
        }

        pub inline fn lerp(a: Self, b: Self, t: T) Self {
            const delta = b.subbed(a);
            return a.added(delta.scaled(t));
        }

        pub inline fn distance(a: Self, b: Self) T {
            return a.subbed(b).magnitude();
        }

        pub inline fn distanceSq(a: Self, b: Self) T {
            return a.subbed(b).magnitudeSq();
        }

        pub inline fn project(a: Self, b: Self) Self {
            const b_mag_sq = b.magnitudeSq();
            if (b_mag_sq == 0) return Self.zero();
            const scale = Self.dot(a, b) / b_mag_sq;
            return b.scaled(scale);
        }

        pub inline fn reflect(v: Self, normal: Self) Self {
            const factor = @as(T, 2) * Self.dot(v, normal);
            return v.subbed(normal.scaled(factor));
        }

        pub inline fn clampLength(a: Self, max_length: T) Self {
            const mag_sq = a.magnitudeSq();
            if (mag_sq <= max_length * max_length) return a;
            if (mag_sq == 0) return a;
            const scale = max_length / @sqrt(mag_sq);
            return a.scaled(scale);
        }

        pub inline fn normalizeOrZero(a: Self, epsilon: T) Self {
            const len = a.magnitude();
            if (len <= epsilon) return Self.zero();
            return a.scaled(@as(T, 1.0) / len);
        }
    };
}
