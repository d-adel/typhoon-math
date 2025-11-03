const std = @import("std");
const builtin = @import("builtin");
const x86 = std.Target.x86;

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
            return .{ .data = @splat(@as(T, 0)) };
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
            self.data = @splat(@as(T, 0));
        }

        pub inline fn add(self: *Self, other: Self) void {
            // Use a local temporary to make combined load/add/store patterns
            // more explicit for the optimizer.
            var tmp = self.data;
            tmp += other.data;
            self.data = tmp;
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
            if (N == 3 and @typeInfo(T) == .float) {
                const x = self.data[0];
                const y = self.data[1];
                const z = self.data[2];
                const len_sq = @mulAdd(T, z, z, @mulAdd(T, y, y, x * x));
                if (len_sq == 0) return;
                const inv_len = @as(T, 1) / @sqrt(len_sq);
                self.data = Simd{ x * inv_len, y * inv_len, z * inv_len, @as(T, 0) };
                return;
            }
            const sum = sumSquares(self.data);
            if (sum == 0) return;
            const inv = fastRsqrt(sum);
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
            if (N == 3 and @typeInfo(T) == .float) {
                const x = a.data[0];
                const y = a.data[1];
                const z = a.data[2];
                const len_sq = @mulAdd(T, z, z, @mulAdd(T, y, y, x * x));
                if (len_sq == 0) return a;
                const inv_len = @as(T, 1) / @sqrt(len_sq);
                return .{ .data = Simd{ x * inv_len, y * inv_len, z * inv_len, @as(T, 0) } };
            }
            const sum = sumSquares(a.data);
            if (sum == 0) return a;
            const inv = fastRsqrt(sum);
            return .{ .data = a.data * @as(@TypeOf(a.data), @splat(inv)) };
        }

        // ========== Vector Products ==========

        pub inline fn dot(a: Self, b: Self) T {
            return dotSimd(a.data, b.data);
        }

        pub inline fn cross(a: Self, b: Self) Self {
            comptime {
                if (N != 3) @compileError("cross is only defined for 3D vectors");
            }
            // SIMD implementation using two shuffles and vector ops.
            // For N==3 we back the vector with ActualN==4 lanes, so we
            // perform 4-lane shuffles where the 4th lane is a padding lane
            // (kept zero). Masks pick y,z,x in lanes 0..2 and leave lane 3
            // untouched so the padding remains zero.
            const m_yzx: [ActualN]i32 = if (ActualN == 4) .{ 1, 2, 0, 3 } else if (ActualN == 8) .{ 1, 2, 0, 3, 4, 5, 6, 7 } else .{ 1, 2 };
            const m_zxy: [ActualN]i32 = if (ActualN == 4) .{ 2, 0, 1, 3 } else if (ActualN == 8) .{ 2, 0, 1, 3, 4, 5, 6, 7 } else .{ 2, 0 };

            const a_yzx = @shuffle(T, a.data, undefined, m_yzx);
            const a_zxy = @shuffle(T, a.data, undefined, m_zxy);
            const b_yzx = @shuffle(T, b.data, undefined, m_yzx);
            const b_zxy = @shuffle(T, b.data, undefined, m_zxy);

            const res: @TypeOf(a.data) = a_yzx * b_zxy - a_zxy * b_yzx;
            return .{ .data = res };
        }

        // ========== Magnitude ==========

        pub inline fn magnitude(a: Self) T {
            return @sqrt(sumSquares(a.data));
        }

        pub inline fn magnitudeSq(a: Self) T {
            return sumSquares(a.data);
        }

        // ========== Comparison ==========

        pub inline fn min(a: Self, b: Self) Self {
            return .{ .data = @min(a.data, b.data) };
        }

        pub inline fn max(a: Self, b: Self) Self {
            return .{ .data = @max(a.data, b.data) };
        }

        pub inline fn lerp(a: Self, b: Self, t: T) Self {
            const t_vec = @as(@TypeOf(a.data), @splat(t));
            const one_minus_t_vec = @as(@TypeOf(a.data), @splat(@as(T, 1) - t));
            return .{ .data = a.data * one_minus_t_vec + b.data * t_vec };
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

        inline fn sumSquares(vec: @Vector(ActualN, T)) T {
            return dotSimd(vec, vec);
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

        inline fn dotSimd(a_vec: Simd, b_vec: Simd) T {
            return @reduce(.Add, a_vec * b_vec);
        }
    };
}
