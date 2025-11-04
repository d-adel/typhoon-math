const std = @import("std");
const builtin = @import("builtin");

/// Generic N-dimensional vector using SIMD operations.
/// Provides common vector operations with optimal SIMD performance.
pub fn Vector(comptime N: usize, comptime T: type) type {
    return struct {
        const Self = @This();
        // Map logical vector size N to a SIMD-friendly backing width (ActualN).
        // We support backing widths of 2, 4, and 8 lanes (common SIMD widths).
        // Vec3 (N == 3) is padded to 4 lanes so 3D ops can be implemented with
        // 4-lane SIMD shuffles without extracting scalar lanes.
        const ActualN = switch (N) {
            1 => 2,
            2 => 2,
            3 => 4,
            4 => 4,
            5 => 8,
            6 => 8,
            7 => 8,
            8 => 8,
            else => @compileError("Vector size > 8 is unsupported by this SIMD implementation"),
        };
        const Simd = @Vector(ActualN, T);
        const SimdAlignment = @sizeOf(Simd);

        // Explicitly align SIMD backing to 32 bytes where possible. For f64
        // with 4 lanes this yields 32-byte (256-bit) alignment which is a good
        // target for AVX2/FMA codegen. The align(32) is a hint to the compiler
        // and helps prevent misaligned vector loads/stores on some targets.
        data: Simd align(SimdAlignment),

        inline fn mutData(self: *Self) *[ActualN]T {
            return @ptrCast(&self.data);
        }

        inline fn constData(self: *const Self) *const [ActualN]T {
            return @ptrCast(&self.data);
        }

        inline fn valueArray(self: Self) [ActualN]T {
            return @bitCast(self.data);
        }

        // ========== Construction ==========

        pub inline fn fromArray(values: [N]T) Self {
            const info = @typeInfo(T);
            switch (info) {
                .bool => unreachable,
                else => {
                    var tmp: [ActualN]T = undefined;
                    // copy supplied values and pad remaining lanes with zero
                    inline for (0..N) |i| tmp[i] = values[i];
                    inline for (N..ActualN) |i| tmp[i] = @as(T, 0);
                    return .{ .data = @bitCast(tmp) };
                },
            }
        }

        pub inline fn fromSlice(slice: []const T) Self {
            if (slice.len != N) @panic("fromSlice: wrong length");
            var tmp: [ActualN]T = undefined;
            inline for (0..N) |i| tmp[i] = slice[i];
            inline for (N..ActualN) |i| tmp[i] = @as(T, 0);
            return .{ .data = @bitCast(tmp) };
        }

        pub inline fn fromPtrArray(p: *const [N]T) Self {
            var tmp: [ActualN]T = undefined;
            inline for (0..N) |i| tmp[i] = p.*[i];
            inline for (N..ActualN) |i| tmp[i] = @as(T, 0);
            return .{ .data = @bitCast(tmp) };
        }

        pub inline fn fromManyPtr(p: [*]const T) Self {
            var tmp: [ActualN]T = undefined;
            inline for (0..N) |i| tmp[i] = p[i];
            inline for (N..ActualN) |i| tmp[i] = @as(T, 0);
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
            var tmp: [ActualN]T = undefined;
            inline for (0..ActualN) |i| tmp[i] = @as(T, 0);
            return .{ .data = @bitCast(tmp) };
        }

        // ========== Element Access ==========

        pub inline fn get(self: Self, idx: usize) T {
            if (idx >= N) @panic("get: index out of range");
            const tmp: [ActualN]T = @bitCast(self.data);
            return tmp[idx];
        }

        /// Fast unchecked accessor for hot inner loops where the index is known
        /// to be valid. Avoids the bounds-check present in `get`.
        pub inline fn getUnchecked(self: Self, idx: usize) T {
            const tmp: [ActualN]T = @bitCast(self.data);
            return tmp[idx];
        }

        pub inline fn set(self: *Self, idx: usize, value: T) void {
            if (idx >= N) @panic("set: index out of range");
            var tmp: [ActualN]T = @bitCast(self.data);
            tmp[idx] = value;
            self.data = @bitCast(tmp);
        }

        // ========== Basic Operations (Mutating) ==========

        pub inline fn clear(self: *Self) void {
            const dst = self.mutData();
            inline for (0..ActualN) |i| dst.*[i] = @as(T, 0);
        }

        pub inline fn add(self: *Self, other: Self) void {
            const dst = self.mutData();
            const src = other.valueArray();
            inline for (0..N) |i| dst.*[i] += src[i];
        }

        pub inline fn sub(self: *Self, other: Self) void {
            const dst = self.mutData();
            const src = other.valueArray();
            inline for (0..N) |i| dst.*[i] -= src[i];
        }

        pub inline fn mul(self: *Self, other: Self) void {
            const dst = self.mutData();
            const src = other.valueArray();
            inline for (0..N) |i| dst.*[i] *= src[i];
        }

        pub inline fn mulScalar(self: *Self, k: T) void {
            const dst = self.mutData();
            inline for (0..N) |i| dst.*[i] *= k;
        }

        pub inline fn negate(self: *Self) void {
            const dst = self.mutData();
            inline for (0..N) |i| dst.*[i] = -dst.*[i];
        }

        pub inline fn normalize(self: *Self) void {
            const dst = self.mutData();
            if (N == 3 and @typeInfo(T) == .float) {
                const x = dst.*[0];
                const y = dst.*[1];
                const z = dst.*[2];
                const len_sq = @mulAdd(T, z, z, @mulAdd(T, y, y, x * x));
                if (len_sq == 0) return;
                const inv_len = @as(T, 1) / @sqrt(len_sq);
                dst.*[0] = x * inv_len;
                dst.*[1] = y * inv_len;
                dst.*[2] = z * inv_len;
                if (ActualN > 3) dst.*[3] = @as(T, 0);
                return;
            }
            const sum = sumSquaresPtr(self.constData());
            if (sum == 0) return;
            const inv = fastRsqrt(sum);
            inline for (0..N) |i| dst.*[i] *= inv;
        }

        // ========== Basic Operations (Non-mutating) ==========

        pub inline fn added(a: Self, b: Self) Self {
            const lhs = a.valueArray();
            const rhs = b.valueArray();
            var out: [ActualN]T = undefined;
            inline for (0..N) |i| out[i] = lhs[i] + rhs[i];
            inline for (N..ActualN) |i| out[i] = lhs[i];
            return .{ .data = @bitCast(out) };
        }

        pub inline fn subbed(a: Self, b: Self) Self {
            const lhs = a.valueArray();
            const rhs = b.valueArray();
            var out: [ActualN]T = undefined;
            inline for (0..N) |i| out[i] = lhs[i] - rhs[i];
            inline for (N..ActualN) |i| out[i] = lhs[i];
            return .{ .data = @bitCast(out) };
        }

        pub inline fn scaled(a: Self, k: T) Self {
            const src = a.valueArray();
            var out: [ActualN]T = undefined;
            inline for (0..N) |i| out[i] = src[i] * k;
            inline for (N..ActualN) |i| out[i] = src[i];
            return .{ .data = @bitCast(out) };
        }

        pub inline fn normalized(a: Self) Self {
            const src = a.valueArray();
            if (N == 3 and @typeInfo(T) == .float) {
                const x = src[0];
                const y = src[1];
                const z = src[2];
                const len_sq = @mulAdd(T, z, z, @mulAdd(T, y, y, x * x));
                if (len_sq == 0) return a;
                const inv_len = @as(T, 1) / @sqrt(len_sq);
                var out: [ActualN]T = src;
                out[0] = x * inv_len;
                out[1] = y * inv_len;
                out[2] = z * inv_len;
                if (ActualN > 3) out[3] = @as(T, 0);
                return .{ .data = @bitCast(out) };
            }
            const sum = sumSquaresPtr(&src);
            if (sum == 0) return a;
            const inv = fastRsqrt(sum);
            var out: [ActualN]T = src;
            inline for (0..N) |i| out[i] *= inv;
            return .{ .data = @bitCast(out) };
        }

        // ========== Vector Products ==========

        pub inline fn dot(a: Self, b: Self) T {
            const lhs = a.valueArray();
            const rhs = b.valueArray();
            var sum: T = 0;
            inline for (0..N) |i| sum = @mulAdd(T, lhs[i], rhs[i], sum);
            return sum;
        }

        pub inline fn cross(a: Self, b: Self) Self {
            comptime {
                if (N != 3) @compileError("cross is only defined for 3D vectors");
            }
            const lhs = a.valueArray();
            const rhs = b.valueArray();
            var out: [ActualN]T = lhs;
            out[0] = lhs[1] * rhs[2] - lhs[2] * rhs[1];
            out[1] = lhs[2] * rhs[0] - lhs[0] * rhs[2];
            out[2] = lhs[0] * rhs[1] - lhs[1] * rhs[0];
            inline for (3..ActualN) |i| out[i] = lhs[i];
            return .{ .data = @bitCast(out) };
        }

        // ========== Magnitude ==========

        pub inline fn magnitude(a: Self) T {
            const arr = a.valueArray();
            return @sqrt(sumSquaresPtr(&arr));
        }

        pub inline fn magnitudeSq(a: Self) T {
            const arr = a.valueArray();
            return sumSquaresPtr(&arr);
        }

        // ========== Comparison ==========

        pub inline fn min(a: Self, b: Self) Self {
            const lhs = a.valueArray();
            const rhs = b.valueArray();
            var out: [ActualN]T = undefined;
            inline for (0..N) |i| out[i] = if (lhs[i] < rhs[i]) lhs[i] else rhs[i];
            inline for (N..ActualN) |i| out[i] = lhs[i];
            return .{ .data = @bitCast(out) };
        }

        pub inline fn max(a: Self, b: Self) Self {
            const lhs = a.valueArray();
            const rhs = b.valueArray();
            var out: [ActualN]T = undefined;
            inline for (0..N) |i| out[i] = if (lhs[i] > rhs[i]) lhs[i] else rhs[i];
            inline for (N..ActualN) |i| out[i] = lhs[i];
            return .{ .data = @bitCast(out) };
        }

        pub inline fn abs(a: Self) Self {
            const src = a.valueArray();
            var out: [ActualN]T = undefined;
            inline for (0..N) |i| {
                const v = src[i];
                out[i] = if (v < 0) -v else v;
            }
            inline for (N..ActualN) |i| out[i] = src[i];
            return .{ .data = @bitCast(out) };
        }

        pub inline fn lerp(a: Self, b: Self, t: T) Self {
            const one_minus_t = @as(T, 1) - t;
            const lhs = a.valueArray();
            const rhs = b.valueArray();
            var out: [ActualN]T = undefined;
            inline for (0..N) |i| out[i] = lhs[i] * one_minus_t + rhs[i] * t;
            inline for (N..ActualN) |i| out[i] = lhs[i];
            return .{ .data = @bitCast(out) };
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

        inline fn sumSquares(vec: Simd) T {
            const arr: [ActualN]T = @bitCast(vec);
            return sumSquaresPtr(&arr);
        }

        inline fn sumSquaresPtr(ptr: *const [ActualN]T) T {
            var sum: T = 0;
            inline for (0..N) |i| sum = @mulAdd(T, ptr.*[i], ptr.*[i], sum);
            return sum;
        }

        inline fn fastRsqrt(value: T) T {
            const info = @typeInfo(T);
            if (info != .float) {
                @compileError("fastRsqrt is only defined for floating point Vector types");
            }
            if (value <= 0) return @as(T, 0);
            if (T == f32) {
                if (builtin.target.cpu.arch == .x86 or builtin.target.cpu.arch == .x86_64) {
                    var approx: f32 = undefined;
                    asm volatile ("rsqrtss %[value], %[result]"
                        : [result] "=x" (approx),
                        : [value] "x" (value),
                    );
                    const half = value * @as(T, 0.5);
                    approx = approx * (@as(T, 1.5) - half * approx * approx);
                    return @as(T, approx);
                }
                var bits: u32 = @bitCast(value);
                bits = 0x5f3759df - (bits >> 1);
                var y: f32 = @bitCast(bits);
                const half = value * @as(T, 0.5);
                y = y * (@as(T, 1.5) - half * y * y);
                return @as(T, y);
            }
            return @as(T, 1) / @sqrt(value);
        }
    };
}
