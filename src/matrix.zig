const std = @import("std");

const Vector = @import("vector.zig").Vector;

/// Generic NxM matrix using SIMD operations.
/// Data is stored in row-major order.
/// Supports square and non-square matrices with specialized operations for 3x3 and 3x4 transforms.
pub fn Matrix(comptime N: usize, comptime M: usize, comptime T: type) type {
    return struct {
        const Self = @This();
        data: @Vector(N * M, T),

        // ========== Construction ==========

        pub inline fn fromArray(values: [N * M]T) Self {
            const info = @typeInfo(T);
            switch (info) {
                .bool => unreachable,
                else => return .{ .data = @bitCast(values) },
            }
        }

        pub inline fn fromSlice(slice: []const T) Self {
            if (slice.len != N * M) @panic("fromSlice: wrong length");
            var tmp: [N * M]T = undefined;
            inline for (0..N * M) |i| tmp[i] = slice[i];
            return .{ .data = @bitCast(tmp) };
        }

        pub inline fn fromPtrArray(p: *const [N * M]T) Self {
            return .{ .data = @bitCast(p.*) };
        }

        pub inline fn fromManyPtr(p: [*]const T) Self {
            var tmp: [N * M]T = undefined;
            inline for (0..N * M) |i| tmp[i] = p[i];
            return .{ .data = @bitCast(tmp) };
        }

        pub inline fn fromSequence(seq: anytype) Self {
            const SeqT = @TypeOf(seq);
            const info = @typeInfo(SeqT);
            switch (info) {
                .array => |ai| {
                    if (ai.len != N * M or ai.child != T)
                        @compileError("fromSequence: expected [N]" ++ @typeName(T));
                    return Self.fromArray(seq);
                },

                .pointer => |pi| switch (pi.size) {
                    .one => {
                        const ci = @typeInfo(pi.child);
                        if (ci == .array and ci.array.len == N * M and ci.array.child == T)
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

        pub inline fn get1D(self: Self, index: usize) T {
            const tmp: [N * M]T = @bitCast(self.data);
            return tmp[index];
        }

        pub inline fn set1D(self: *Self, index: usize, value: T) void {
            var tmp: [N * M]T = @bitCast(self.data);
            tmp[index] = value;
            self.data = @bitCast(tmp);
        }

        pub inline fn get(self: Self, row: usize, col: usize) T {
            const idx_lin = row * M + col;
            return self.get1D(idx_lin);
        }

        pub inline fn set(self: *Self, row: usize, col: usize, value: T) void {
            const idx_lin = row * M + col;
            self.set1D(idx_lin, value);
        }

        // ========== Matrix Operations ==========

        pub inline fn identity() Self {
            comptime {
                if (N != M) @compileError("identity() requires a square matrix");
            }

            var arr: [N * N]T = undefined;
            inline for (0..N) |r| {
                inline for (0..N) |c| {
                    arr[r * N + c] = if (r == c) @as(T, 1) else @as(T, 0);
                }
            }

            return .{ .data = @bitCast(arr) };
        }

        pub inline fn isSquare() bool {
            return N == M;
        }

        pub inline fn determinant(self: Self) T {
            if (N == 3 and M == 3) return self.det3();
            if (N == 3 and M == 4) {
                const R: [3][3]T = .{
                    .{ self.data[idx(0, 0)], self.data[idx(0, 1)], self.data[idx(0, 2)] },
                    .{ self.data[idx(1, 0)], self.data[idx(1, 1)], self.data[idx(1, 2)] },
                    .{ self.data[idx(2, 0)], self.data[idx(2, 1)], self.data[idx(2, 2)] },
                };
                const a = R[0][0];
                const b = R[0][1];
                const c = R[0][2];
                const d = R[1][0];
                const e = R[1][1];
                const f = R[1][2];
                const g = R[2][0];
                const h = R[2][1];
                const i = R[2][2];
                return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
            }

            if (N != M) @compileError("determinant: non-square size (except 3x4 affine) not supported");

            var A: [N][N]T = undefined;
            inline for (0..N) |r| inline for (0..N) |c| {
                A[r][c] = self.data[idx(r, c)];
            };

            var det: T = 1;
            var sign: T = 1;
            inline for (0..N) |k| {
                var piv_row: usize = k;
                var piv_val: T = @abs(A[piv_row][k]);
                inline for (k + 1..N) |r| {
                    const v = @abs(A[r][k]);
                    if (v > piv_val) {
                        piv_val = v;
                        piv_row = r;
                    }
                }
                if (piv_val == 0) return @as(T, 0);

                if (piv_row != k) {
                    inline for (0..N) |c| std.mem.swap(T, &A[k][c], &A[piv_row][c]);
                    sign = -sign;
                }

                const piv = A[k][k];

                inline for (k + 1..N) |r| {
                    const f = A[r][k] / piv;
                    inline for (k..N) |c| {
                        A[r][c] -= f * A[k][c];
                    }
                }
                det *= piv;
            }
            return sign * det;
        }

        // ========== Vector Multiplication ==========

        pub inline fn mulVec(self: Self, vec: Vector(M, T)) Vector(N, T) {
            var out_arr: [N]T = undefined;
            inline for (0..N) |r| {
                const mask = comptime blk: {
                    var m: [M]i32 = undefined;
                    for (0..M) |c| m[c] = @intCast(r * M + c);
                    break :blk m;
                };
                const row: @Vector(M, T) = @shuffle(T, self.data, undefined, mask);
                out_arr[r] = @reduce(.Add, row * vec.data);
            }
            return .{ .data = @bitCast(out_arr) };
        }

        // ========== Transpose ==========

        pub inline fn transpose(self: *Self) void {
            comptime {
                if (N != M) @compileError("transpose(self: *Self) only valid for square matrices");
            }
            inline for (0..N) |r| {
                inline for (r + 1..N) |c| {
                    const a_idx = idx(r, c);
                    const b_idx = idx(c, r);
                    const tmp = self.data[a_idx];
                    self.data[a_idx] = self.data[b_idx];
                    self.data[b_idx] = tmp;
                }
            }
        }

        pub inline fn transposed(self: Self) Matrix(M, N, T) {
            var out: [M * N]T = undefined;

            inline for (0..N) |r| {
                inline for (0..M) |c| {
                    out[c * N + r] = self.data[r * M + c];
                }
            }

            return .{ .data = @bitCast(out) };
        }

        // ========== Inverse ==========

        pub inline fn invert(self: *Self) void {
            if (N == 3 and M == 3) {
                const inv = self.inverse3();
                self.* = inv;
            } else if (N == 3 and M == 4) {
                const eps_ortho: T = 1e-5;
                const eps_det: T = 1e-4;

                const detR: T = self.determinant();
                const det_ok = @abs(detR - @as(T, 1)) <= eps_det;

                const inv = if (det_ok and self.isRigid3x4(eps_ortho))
                    self.inverseRigid3x4()
                else
                    self.inverseAffine3x4();
                self.* = inv;
            } else {
                @compileError("invert(self: *Self) not implemented for this Matrix size");
            }
        }

        pub inline fn inverted(self: Self) Self {
            if (N == 3 and M == 3) {
                return self.inverse3();
            } else {
                @compileError("inverted() currently only supports 3x3 matrices");
            }
        }

        // ========== 3D Transform Operations (3x4 matrices) ==========

        pub inline fn transform(self: Self, p: Vector(3, T)) Vector(3, T) {
            comptime {
                if (!(N == 3 and M == 4)) @compileError("transform requires Matrix(3,4,T)");
            }
            const i = idx;
            const x = p.data[0];
            const y = p.data[1];
            const z = p.data[2];

            const rx = @mulAdd(T, z, self.data[i(0, 2)], @mulAdd(T, y, self.data[i(0, 1)], @mulAdd(T, x, self.data[i(0, 0)], self.data[i(0, 3)])));
            const ry = @mulAdd(T, z, self.data[i(1, 2)], @mulAdd(T, y, self.data[i(1, 1)], @mulAdd(T, x, self.data[i(1, 0)], self.data[i(1, 3)])));
            const rz = @mulAdd(T, z, self.data[i(2, 2)], @mulAdd(T, y, self.data[i(2, 1)], @mulAdd(T, x, self.data[i(2, 0)], self.data[i(2, 3)])));

            return Vector(3, T).fromArray(.{ rx, ry, rz });
        }

        pub inline fn transformInverse(self: Self, p: Vector(3, T)) Vector(3, T) {
            comptime {
                if (!(N == 3 and M == 4)) @compileError("transformInverse requires Matrix(3,4,T)");
            }
            const i = idx;
            const tx = p.data[0] - self.data[i(0, 3)];
            const ty = p.data[1] - self.data[i(1, 3)];
            const tz = p.data[2] - self.data[i(2, 3)];

            const rx = @mulAdd(T, tz, self.data[i(2, 0)], @mulAdd(T, ty, self.data[i(1, 0)], tx * self.data[i(0, 0)]));
            const ry = @mulAdd(T, tz, self.data[i(2, 1)], @mulAdd(T, ty, self.data[i(1, 1)], tx * self.data[i(0, 1)]));
            const rz = @mulAdd(T, tz, self.data[i(2, 2)], @mulAdd(T, ty, self.data[i(1, 2)], tx * self.data[i(0, 2)]));

            return Vector(3, T).fromArray(.{ rx, ry, rz });
        }

        pub inline fn transformDirection(self: Self, v: Vector(3, T)) Vector(3, T) {
            comptime {
                if (!(N == 3 and M == 4)) @compileError("transformDirection requires Matrix(3,4,T)");
            }
            const i = idx;
            const x = v.data[0];
            const y = v.data[1];
            const z = v.data[2];

            const rx = @mulAdd(T, z, self.data[i(0, 2)], @mulAdd(T, y, self.data[i(0, 1)], x * self.data[i(0, 0)]));
            const ry = @mulAdd(T, z, self.data[i(1, 2)], @mulAdd(T, y, self.data[i(1, 1)], x * self.data[i(1, 0)]));
            const rz = @mulAdd(T, z, self.data[i(2, 2)], @mulAdd(T, y, self.data[i(2, 1)], x * self.data[i(2, 0)]));

            return Vector(3, T).fromArray(.{ rx, ry, rz });
        }

        pub inline fn transformInverseDirection(self: Self, v: Vector(3, T)) Vector(3, T) {
            comptime {
                if (!(N == 3 and M == 4)) @compileError("transformInverseDirection requires Matrix(3,4,T)");
            }
            const i = idx;
            const x = v.data[0];
            const y = v.data[1];
            const z = v.data[2];

            const rx = @mulAdd(T, z, self.data[i(2, 0)], @mulAdd(T, y, self.data[i(1, 0)], x * self.data[i(0, 0)]));
            const ry = @mulAdd(T, z, self.data[i(2, 1)], @mulAdd(T, y, self.data[i(1, 1)], x * self.data[i(0, 1)]));
            const rz = @mulAdd(T, z, self.data[i(2, 2)], @mulAdd(T, y, self.data[i(1, 2)], x * self.data[i(0, 2)]));

            return Vector(3, T).fromArray(.{ rx, ry, rz });
        }

        // ========== Internal Helpers ==========

        inline fn idx(r: usize, c: usize) usize {
            return r * M + c;
        }

        inline fn dot3(a: @Vector(3, T), b: @Vector(3, T)) T {
            return @reduce(.Add, a * b);
        }
        inline fn cross3(a: @Vector(3, T), b: @Vector(3, T)) @Vector(3, T) {
            const m_yzx: [3]i32 = .{ 1, 2, 0 };
            const m_zxy: [3]i32 = .{ 2, 0, 1 };
            const a_yzx = @shuffle(T, a, undefined, m_yzx);
            const a_zxy = @shuffle(T, a, undefined, m_zxy);
            const b_yzx = @shuffle(T, b, undefined, m_yzx);
            const b_zxy = @shuffle(T, b, undefined, m_zxy);
            return a_yzx * b_zxy - a_zxy * b_yzx;
        }

        inline fn det3(self: Self) T {
            comptime {
                if (!(N == 3 and M == 3)) @compileError("det3 requires 3x3");
            }

            const m0: [3]i32 = .{ 0, 1, 2 };
            const m1: [3]i32 = .{ 3, 4, 5 };
            const m2: [3]i32 = .{ 6, 7, 8 };
            const r0: @Vector(3, T) = @shuffle(T, self.data, undefined, m0);
            const r1: @Vector(3, T) = @shuffle(T, self.data, undefined, m1);
            const r2: @Vector(3, T) = @shuffle(T, self.data, undefined, m2);
            return dot3(r0, cross3(r1, r2));
        }

        inline fn isRigid3x4(self: Self, eps: T) bool {
            comptime {
                if (!(N == 3 and M == 4)) @compileError("isRigid3x4 requires 3x4");
            }
            const i = idx;

            const c0: @Vector(3, T) = .{ self.data[i(0, 0)], self.data[i(1, 0)], self.data[i(2, 0)] };
            const c1: @Vector(3, T) = .{ self.data[i(0, 1)], self.data[i(1, 1)], self.data[i(2, 1)] };
            const c2: @Vector(3, T) = .{ self.data[i(0, 2)], self.data[i(1, 2)], self.data[i(2, 2)] };

            const one: T = 1;
            const n0 = dot3(c0, c0);
            const n1 = dot3(c1, c1);
            const n2 = dot3(c2, c2);
            if (@abs(n0 - one) > eps or @abs(n1 - one) > eps or @abs(n2 - one) > eps) return false;

            const z: T = 0;
            if (@abs(dot3(c0, c1) - z) > eps) return false;
            if (@abs(dot3(c0, c2) - z) > eps) return false;
            if (@abs(dot3(c1, c2) - z) > eps) return false;

            return true;
        }

        inline fn inverse3(self: Self) Self {
            comptime {
                if (!(N == 3 and M == 3)) @compileError("inverse3 requires 3x3");
            }
            const m0: [3]i32 = .{ 0, 1, 2 };
            const m1: [3]i32 = .{ 3, 4, 5 };
            const m2: [3]i32 = .{ 6, 7, 8 };
            const r0: @Vector(3, T) = @shuffle(T, self.data, undefined, m0);
            const r1: @Vector(3, T) = @shuffle(T, self.data, undefined, m1);
            const r2: @Vector(3, T) = @shuffle(T, self.data, undefined, m2);

            const c0: @Vector(3, T) = cross3(r1, r2);
            const c1: @Vector(3, T) = cross3(r2, r0);
            const c2: @Vector(3, T) = cross3(r0, r1);

            const det: T = dot3(r0, c0);
            if (det == 0) return self;
            const invd: T = @as(T, 1) / det;

            const invdV: @Vector(3, T) = @splat(invd);
            const rc0: @Vector(3, T) = c0 * invdV;
            const rc1: @Vector(3, T) = c1 * invdV;
            const rc2: @Vector(3, T) = c2 * invdV;

            const out: [9]T = .{
                rc0[0], rc1[0], rc2[0],
                rc0[1], rc1[1], rc2[1],
                rc0[2], rc1[2], rc2[2],
            };
            return .{ .data = @bitCast(out) };
        }

        inline fn inverseRigid3x4(self: Self) Self {
            comptime {
                if (!(N == 3 and M == 4)) @compileError("inverseRigid3x4 requires 3x4");
            }
            const i = idx;
            const rt0: @Vector(3, T) = .{ self.data[i(0, 0)], self.data[i(1, 0)], self.data[i(2, 0)] };
            const rt1: @Vector(3, T) = .{ self.data[i(0, 1)], self.data[i(1, 1)], self.data[i(2, 1)] };
            const rt2: @Vector(3, T) = .{ self.data[i(0, 2)], self.data[i(1, 2)], self.data[i(2, 2)] };
            const t: @Vector(3, T) = .{ self.data[i(0, 3)], self.data[i(1, 3)], self.data[i(2, 3)] };

            const ntx: T = -dot3(rt0, t);
            const nty: T = -dot3(rt1, t);
            const ntz: T = -dot3(rt2, t);

            const out: [12]T = .{
                rt0[0], rt0[1], rt0[2], ntx,
                rt1[0], rt1[1], rt1[2], nty,
                rt2[0], rt2[1], rt2[2], ntz,
            };
            return .{ .data = @bitCast(out) };
        }

        inline fn inverseAffine3x4(self: Self) Self {
            comptime {
                if (!(N == 3 and M == 4)) @compileError("inverseAffine3x4 requires 3x4");
            }
            const i = idx;

            const r0: @Vector(3, T) = .{ self.data[i(0, 0)], self.data[i(0, 1)], self.data[i(0, 2)] };
            const r1: @Vector(3, T) = .{ self.data[i(1, 0)], self.data[i(1, 1)], self.data[i(1, 2)] };
            const r2: @Vector(3, T) = .{ self.data[i(2, 0)], self.data[i(2, 1)], self.data[i(2, 2)] };

            const c0: @Vector(3, T) = cross3(r1, r2);
            const c1: @Vector(3, T) = cross3(r2, r0);
            const c2: @Vector(3, T) = cross3(r0, r1);

            const det: T = dot3(r0, c0);
            if (det == 0) return self;
            const invd: T = @as(T, 1) / det;

            const invdV: @Vector(3, T) = @splat(invd);

            const rti0: @Vector(3, T) = c0 * invdV;
            const rti1: @Vector(3, T) = c1 * invdV;
            const rti2: @Vector(3, T) = c2 * invdV;

            const t: @Vector(3, T) = .{ self.data[i(0, 3)], self.data[i(1, 3)], self.data[i(2, 3)] };

            const ntx: T = -(rti0[0] * t[0] + rti1[0] * t[1] + rti2[0] * t[2]);
            const nty: T = -(rti0[1] * t[0] + rti1[1] * t[1] + rti2[1] * t[2]);
            const ntz: T = -(rti0[2] * t[0] + rti1[2] * t[1] + rti2[2] * t[2]);

            const out: [12]T = .{
                rti0[0], rti1[0], rti2[0], ntx,
                rti0[1], rti1[1], rti2[1], nty,
                rti0[2], rti1[2], rti2[2], ntz,
            };
            return .{ .data = @bitCast(out) };
        }

        // ========== 3x3 Utilities ==========

        pub inline fn diagonal(d0: T, d1: T, d2: T) Self {
            comptime {
                if (N != 3 or M != 3) @compileError("diagonal requires 3x3");
            }
            var arr: [9]T = undefined;
            arr[0] = d0;
            arr[1] = @as(T, 0);
            arr[2] = @as(T, 0);
            arr[3] = @as(T, 0);
            arr[4] = d1;
            arr[5] = @as(T, 0);
            arr[6] = @as(T, 0);
            arr[7] = @as(T, 0);
            arr[8] = d2;
            return .{ .data = @bitCast(arr) };
        }
    };
}
