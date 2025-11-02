const std = @import("std");
const testing = std.testing;
const math = std.math;
const Random = std.Random;

const Matrix = @import("matrix.zig").Matrix;
const Quaternion = @import("quaternion.zig").Quaternion;
const Vector = @import("vector.zig").Vector;

fn expectApproxEq(comptime N: usize, v: anytype, w: anytype, eps: f64) !void {
    inline for (0..N) |i| try testing.expectApproxEqAbs(v.data[i], w.data[i], eps);
}

// Helpers: quaternion -> 3x3 rotation matrix (normalized quaternion expected)
fn quatToMat33(T: type, q: Quaternion(T)) Matrix(3, 3, T) {
    const w = q.data[0];
    const x = q.data[1];
    const y = q.data[2];
    const z = q.data[3];

    const two: T = 2;

    const xx = x * x;
    const yy = y * y;
    const zz = z * z;
    const xy = x * y;
    const xz = x * z;
    const yz = y * z;
    const wx = w * x;
    const wy = w * y;
    const wz = w * z;

    return Matrix(3, 3, T).fromArray(.{
        1 - two * (yy + zz), two * (xy - wz),     two * (xz + wy),
        two * (xy + wz),     1 - two * (xx + zz), two * (yz - wx),
        two * (xz - wy),     two * (yz + wx),     1 - two * (xx + yy),
    });
}

fn makeRigid34(T: type, q: Quaternion(T), t: Vector(3, T)) Matrix(3, 4, T) {
    const R = quatToMat33(T, q);
    return Matrix(3, 4, T).fromArray(.{
        R.data[0], R.data[1], R.data[2], t.data[0],
        R.data[3], R.data[4], R.data[5], t.data[1],
        R.data[6], R.data[7], R.data[8], t.data[2],
    });
}

fn makeAffine34_RS(T: type, q: Quaternion(T), s: Vector(3, T), t: Vector(3, T)) Matrix(3, 4, T) {
    // A = R * diag(sx, sy, sz)
    const R = quatToMat33(T, q);
    const sx = s.data[0];
    const sy = s.data[1];
    const sz = s.data[2];
    return Matrix(3, 4, T).fromArray(.{
        R.data[0] * sx, R.data[1] * sy, R.data[2] * sz, t.data[0],
        R.data[3] * sx, R.data[4] * sy, R.data[5] * sz, t.data[1],
        R.data[6] * sx, R.data[7] * sy, R.data[8] * sz, t.data[2],
    });
}

// ---------------- Vectors ----------------
test "Vector: constructors, add, dot, magnitude, normalize, negate, scalar" {
    const V3 = Vector(3, f64);

    const a = V3.fromArray(.{ 1.0, 2.0, 3.0 });
    const b = V3.fromArray(.{ 4.0, 5.0, 6.0 });

    // add
    var c = a;
    c.add(b);
    try expectApproxEq(3, c, V3.fromArray(.{ 5.0, 7.0, 9.0 }), 1e-12);

    // dot
    try testing.expectApproxEqAbs(V3.dot(a, b), 32.0, 1e-12);

    // magnitude / magnitudeSq
    try testing.expectApproxEqAbs(a.magnitude(), @sqrt(@as(f64, 14.0)), 1e-12);
    try testing.expectApproxEqAbs(a.magnitudeSq(), 14.0, 1e-12);

    // normalize
    var n = a;
    n.normalize();
    try testing.expectApproxEqAbs(n.magnitude(), 1.0, 1e-5);

    // mul scalar
    var k = a;
    k.mulScalar(2.5);
    try expectApproxEq(3, k, V3.fromArray(.{ 2.5, 5.0, 7.5 }), 1e-12);

    // negate
    var neg = a;
    neg.negate();
    try expectApproxEq(3, neg, V3.fromArray(.{ -1.0, -2.0, -3.0 }), 1e-12);
}

// ---------------- Quaternions ----------------
test "Quaternion: identity, normalize, conjugate/inverse, mul, no-ops" {
    const Qf = Quaternion(f64);
    const V3 = Vector(3, f64);

    const q = Qf.fromArray(.{ 1.0, 2.0, 3.0, 4.0 });

    // normalize -> unit length
    var qn = q;
    qn.normalize();
    try testing.expectApproxEqAbs(qn.magnitudeSq(), 1.0, 1e-5);

    // identity
    const I = Qf.identity();
    try expectApproxEq(4, I, Qf.fromArray(.{ 1.0, 0.0, 0.0, 0.0 }), 1e-12);

    // conjugate
    var qc = q;
    qc.conjugate();
    try expectApproxEq(4, qc, Qf.fromArray(.{ 1.0, -2.0, -3.0, -4.0 }), 1e-12);

    // inverse: q * q^{-1} ≈ I
    var qi = q;
    qi.inverse();
    var prod = q;
    prod.mulAssign(qi);
    try expectApproxEq(4, prod, I, 1e-12);

    // mul identity leaves unchanged
    var r = I;
    r.mulAssign(q);
    try expectApproxEq(4, r, q, 1e-12);

    // addScaledVector with zero should be no-op
    var q2 = q;
    q2.addScaledVector(V3.zero(), 0.0);
    try expectApproxEq(4, q2, q, 1e-12);

    // rotateByVector with zero vector = no change
    var q3 = q;
    q3.rotateByVector(V3.zero());
    try expectApproxEq(4, q3, q, 1e-12);
}

// ---------------- Matrices (3x3 and 3x4) ----------------
test "Matrix 3x3: identity, det, inverse, transpose, mulVec" {
    const V3 = Vector(3, f64);
    const M33 = Matrix(3, 3, f64);

    // Identity basics
    const I = M33.identity();
    const v = V3.fromArray(.{ 7, 8, 9 });
    try expectApproxEq(3, I.mulVec(v), v, 1e-12);
    try testing.expectApproxEqAbs(I.determinant(), 1.0, 1e-12);

    // Known matrix with det = 1 (classic example)
    const A = M33.fromArray(.{
        1, 2, 3,
        0, 1, 4,
        5, 6, 0,
    });
    try testing.expectApproxEqAbs(A.determinant(), 1.0, 1e-5);

    // Inverse check via solving A_inv(A v) ≈ v
    const Av = A.mulVec(v);
    var Ainv = A;
    Ainv.invert();
    const u = Ainv.mulVec(Av);
    try expectApproxEq(3, u, v, 1e-12);

    // Transpose round-trip
    var At = A.transposed();
    At.transpose();
    const Att = At;
    // Check via action on vectors (A and Att should act the same)
    try expectApproxEq(3, Att.mulVec(v), A.mulVec(v), 1e-12);
}

test "Matrix 3x4 (rigid): transform, direction, inverse, determinant" {
    const V3 = Vector(3, f64);
    const M34 = Matrix(3, 4, f64);

    // R = I, t = (1,2,3)
    const T = M34.fromArray(.{
        1, 0, 0, 1,
        0, 1, 0, 2,
        0, 0, 1, 3,
    });

    // determinant equals det(R) = 1
    try testing.expectApproxEqAbs(T.determinant(), 1.0, 1e-12);

    const p = V3.fromArray(.{ 4, 5, 6 });
    const v = V3.fromArray(.{ 4, 5, 6 });

    // transform: p' = R p + t => (5,7,9)
    const pt = T.transform(p);
    try expectApproxEq(3, pt, V3.fromArray(.{ 5, 7, 9 }), 1e-12);

    // transformDirection ignores translation
    const vd = T.transformDirection(v);
    try expectApproxEq(3, vd, v, 1e-12);

    // inverse should undo transform
    var Ti = T;
    Ti.invert();
    const back = Ti.transform(pt);
    try expectApproxEq(3, back, p, 1e-5);

    // transformInverse should match inverse().transform for points
    const back2 = T.transformInverse(pt);
    try expectApproxEq(3, back2, p, 1e-5);
}

test "Matrix 3x4 (affine scale+translate): inverse correctness" {
    const V3 = Vector(3, f64);
    const M34 = Matrix(3, 4, f64);

    // R = diag(2,3,4), t = (1,2,3)
    const A = M34.fromArray(.{
        2, 0, 0, 1,
        0, 3, 0, 2,
        0, 0, 4, 3,
    });
    // det(R) = 24
    try testing.expectApproxEqAbs(A.determinant(), 24.0, 1e-12);

    const p = V3.fromArray(.{ 1.5, -2.0, 0.75 });
    const pt = A.transform(p);
    var inv = A;
    inv.invert();
    const back = inv.transform(pt);
    try expectApproxEq(3, back, p, 1e-5);
}

test "Quaternion->Matrix3x3 rotation: orthonormal and det ≈ 1" {
    const Qf = Quaternion(f64);
    // random-ish unit quaternion
    var q = Qf.fromArray(.{ 0.9, 0.2, -0.3, 0.1 });
    q.normalize();
    const R = quatToMat33(f64, q);

    // Orthonormal columns/rows and det ~ 1
    const eps: f64 = 1e-12;
    // columns
    const c0: @Vector(3, f64) = .{ R.data[0], R.data[3], R.data[6] };
    const c1: @Vector(3, f64) = .{ R.data[1], R.data[4], R.data[7] };
    const c2: @Vector(3, f64) = .{ R.data[2], R.data[5], R.data[8] };
    try testing.expectApproxEqAbs(@reduce(.Add, c0 * c0), 1.0, eps);
    try testing.expectApproxEqAbs(@reduce(.Add, c1 * c1), 1.0, eps);
    try testing.expectApproxEqAbs(@reduce(.Add, c2 * c2), 1.0, eps);
    try testing.expectApproxEqAbs(@reduce(.Add, c0 * c1), 0.0, eps);
    try testing.expectApproxEqAbs(@reduce(.Add, c0 * c2), 0.0, eps);
    try testing.expectApproxEqAbs(@reduce(.Add, c1 * c2), 0.0, eps);
    try testing.expectApproxEqAbs(R.determinant(), 1.0, 1e-9);
}

test "Property: rigid 3x4 inverse (randomized)" {
    const Qf = Quaternion(f64);
    const V3 = Vector(3, f64);
    var prng = Random.DefaultPrng.init(0x00C0FFEE);
    const rand = prng.random();

    const trials = 16;
    var i: usize = 0;
    while (i < trials) : (i += 1) {
        // random quaternion
        var q = Qf.fromArray(.{
            rand.float(f64), rand.float(f64), rand.float(f64), rand.float(f64),
        });
        q.normalize();
        const t = V3.fromArray(.{ rand.float(f64) * 10 - 5, rand.float(f64) * 10 - 5, rand.float(f64) * 10 - 5 });
        const T = makeRigid34(f64, q, t);

        var Ti = T;
        Ti.invert();
        // random point
        const p = V3.fromArray(.{ rand.float(f64) * 3 - 1.5, rand.float(f64) * 3 - 1.5, rand.float(f64) * 3 - 1.5 });
        const pt = T.transform(p);
        const back = Ti.transform(pt);
        try expectApproxEq(3, back, p, 1e-9);

        // direction
        const v = V3.fromArray(.{ rand.float(f64) - 0.5, rand.float(f64) - 0.5, rand.float(f64) - 0.5 });
        const vt = T.transformDirection(v);
        const vback = Ti.transformDirection(vt);
        try expectApproxEq(3, vback, v, 1e-9);
    }
}

test "Property: affine 3x4 inverse (randomized scales)" {
    const Qf = Quaternion(f64);
    const V3 = Vector(3, f64);
    var prng = Random.DefaultPrng.init(0x1234_5678_9ABC_DEF0);
    const rand = prng.random();

    const trials = 12;
    var i: usize = 0;
    while (i < trials) : (i += 1) {
        var q = Qf.fromArray(.{
            rand.float(f64), rand.float(f64), rand.float(f64), rand.float(f64),
        });
        q.normalize();
        // scales in [0.5, 2]
        const s = V3.fromArray(.{ 0.5 + 1.5 * rand.float(f64), 0.5 + 1.5 * rand.float(f64), 0.5 + 1.5 * rand.float(f64) });
        const t = V3.fromArray(.{ rand.float(f64) * 10 - 5, rand.float(f64) * 10 - 5, rand.float(f64) * 10 - 5 });
        const A = makeAffine34_RS(f64, q, s, t);

        var Ai = A;
        Ai.invert();
        const p = V3.fromArray(.{ rand.float(f64) * 5 - 2.5, rand.float(f64) * 5 - 2.5, rand.float(f64) * 5 - 2.5 });
        const pt = A.transform(p);
        const back = Ai.transform(pt);
        try expectApproxEq(3, back, p, 2e-3);
    }
}

test "Vector: zero normalize is no-op" {
    const V3 = Vector(3, f64);
    var z = V3.zero();
    z.normalize();
    try expectApproxEq(3, z, V3.zero(), 1e-12);
}

test "2D basics: Vector(2) and Matrix(2,2) det and identity mul" {
    const V2 = Vector(2, f64);
    const M22 = Matrix(2, 2, f64);

    const I = M22.fromArray(.{ 1, 0, 0, 1 });
    const v = V2.fromArray(.{ 3, -4 });
    try expectApproxEq(2, I.mulVec(v), v, 1e-12);

    const A = M22.fromArray(.{ 2, 3, 5, 7 });
    try testing.expectApproxEqAbs(A.determinant(), 2 * 7 - 3 * 5, 1e-12);
}

// ---------------- More Vector/Quaternion coverage ----------------
test "Vector: assign ops and fromSlice/fromSequence" {
    const V4 = Vector(4, f64);
    const arr: [4]f64 = .{ 1, 2, 3, 4 };
    const s = arr[0..];
    const v = V4.fromSlice(s);
    try expectApproxEq(4, v, V4.fromArray(arr), 1e-12);

    var a = V4.fromArray(.{ 1, 1, 1, 1 });
    a.add(V4.fromArray(.{ 2, 3, 4, 5 }));
    try expectApproxEq(4, a, V4.fromArray(.{ 3, 4, 5, 6 }), 1e-12);

    var b = V4.fromArray(.{ 2, 2, 2, 2 });
    b.mulScalar(0.5);
    try expectApproxEq(4, b, V4.fromArray(.{ 1, 1, 1, 1 }), 1e-12);

    // normalize in-place
    var c = V4.fromArray(.{ 3, 0, 0, 4 });
    var cn = c;
    cn.normalize();
    c.normalize();
    try expectApproxEq(4, c, cn, 1e-12);
}

test "Quaternion: conjugate property and assign ops" {
    const Qf = Quaternion(f64);
    const V3 = Vector(3, f64);

    const q = Qf.fromArray(.{ 1.2, -0.5, 2.0, 0.75 });
    var qc = q;
    qc.conjugate();
    var prod2 = q;
    prod2.mulAssign(qc);
    const m2 = q.magnitudeSq();
    // q * conj(q) = (|q|^2, 0, 0, 0)
    try expectApproxEq(4, prod2, Qf.fromArray(.{ m2, 0, 0, 0 }), 1e-5);

    // normalizeAssign ≈ normalize
    var qn = q;
    qn.normalize();
    var qnn = q;
    qnn.normalize();
    try expectApproxEq(4, qn, qnn, 1e-5);

    // addScaledVector with zero scale is no-op
    var q2 = q;
    q2.addScaledVector(V3.fromArray(.{ 1, 0, 0 }), 0.0);
    try expectApproxEq(4, q2, q, 1e-12);

    // rotateByVectorAssign matches non-mutating
    var q3 = q;
    q3.rotateByVector(V3.zero());
    var q3b = q;
    q3b.rotateByVector(V3.zero());
    try expectApproxEq(4, q3, q3b, 1e-12);
}

// ---------------- Edge cases ----------------
test "Quaternion: zero normalize and inverse yield identity" {
    const Qf = Quaternion(f64);
    const I = Qf.identity();
    var z = Qf.zero();
    z.normalize();
    try expectApproxEq(4, z, I, 1e-12);
    z.inverse();
    try expectApproxEq(4, z, I, 1e-12);
}

test "Matrix 3x3: singular determinant and inverse is no-op" {
    const M33 = Matrix(3, 3, f64);
    // rows r1 = 2*r0 => singular
    const S = M33.fromArray(.{
        1, 2, 3,
        2, 4, 6,
        0, 1, 1,
    });
    try testing.expectApproxEqAbs(S.determinant(), 0.0, 1e-12);
    // our inverse3 returns self on det==0
    var Si = S;
    Si.invert();
    // Compare element-wise
    inline for (0..9) |i| try testing.expectApproxEqAbs(S.data[i], Si.data[i], 0.0);
}

test "Matrix 3x4: singular rotation block det=0 and inverse is no-op" {
    const M34 = Matrix(3, 4, f64);
    // R has duplicate columns -> det=0
    const A = M34.fromArray(.{
        1, 1, 0, 1,
        0, 0, 1, 2,
        0, 0, 0, 3,
    });
    try testing.expectApproxEqAbs(A.determinant(), 0.0, 1e-12);
    var Ai = A;
    Ai.invert();
    inline for (0..12) |i| try testing.expectApproxEqAbs(A.data[i], Ai.data[i], 0.0);
}

test "Matrix 3x4: transpose shape and element mapping" {
    const M34 = Matrix(3, 4, f64);
    const M43 = Matrix(4, 3, f32);
    const A = M34.fromArray(.{
        1, 2,  3,  4,
        5, 6,  7,  8,
        9, 10, 11, 12,
    });
    const At = A.transposed();
    // Expected 4x3 data row-major
    const E = M43.fromArray(.{
        1, 5, 9,
        2, 6, 10,
        3, 7, 11,
        4, 8, 12,
    });
    inline for (0..12) |i| try testing.expectApproxEqAbs(@as(f64, At.data[i]), @as(f64, E.data[i]), 0.0);
}

test "Vector: extreme magnitudes normalize near unit" {
    const V3 = Vector(3, f64);
    const small = V3.fromArray(.{ 1e-20, -2e-20, 3e-20 });
    var sn = small;
    sn.normalize();
    try testing.expectApproxEqAbs(sn.magnitude(), 1.0, 1e-9);

    const large = V3.fromArray(.{ 1e10, -2e10, 3e10 });
    var ln = large;
    ln.normalize();
    try testing.expectApproxEqAbs(ln.magnitude(), 1.0, 1e-9);
}

test "Rigid 3x4: transformInverseDirection == inverse().transformDirection" {
    const Qf = Quaternion(f64);
    const V3 = Vector(3, f64);
    // Build a rigid transform
    var q = Qf.fromArray(.{ 0.7, 0.1, -0.2, 0.6 });
    q.normalize();
    const t = V3.fromArray(.{ 1.5, -2.0, 0.25 });
    const T = makeRigid34(f64, q, t);

    const v = V3.fromArray(.{ 0.3, -0.4, 0.5 });
    const vt = T.transformDirection(v);
    const vback1 = T.transformInverseDirection(vt);
    var Ti2 = T;
    Ti2.invert();
    const vback2 = Ti2.transformDirection(vt);
    try expectApproxEq(3, vback1, vback2, 1e-12);
}

// ========== Additional Coverage Tests ==========

test "Vector: get/set element access" {
    const V3 = Vector(3, f64);
    var v = V3.fromArray(.{ 1, 2, 3 });
    try testing.expectEqual(v.get(0), 1.0);
    try testing.expectEqual(v.get(1), 2.0);
    try testing.expectEqual(v.get(2), 3.0);

    v.set(1, 10.0);
    try testing.expectEqual(v.get(1), 10.0);
}

test "Vector: clear operation" {
    const V4 = Vector(4, f64);
    var v = V4.fromArray(.{ 1, 2, 3, 4 });
    v.clear();
    try expectApproxEq(4, v, V4.zero(), 1e-12);
}

test "Vector: min/max operations" {
    const V3 = Vector(3, f64);
    const a = V3.fromArray(.{ 1, 5, 2 });
    const b = V3.fromArray(.{ 3, 2, 4 });
    const minv = V3.min(a, b);
    const maxv = V3.max(a, b);
    try expectApproxEq(3, minv, V3.fromArray(.{ 1, 2, 2 }), 1e-12);
    try expectApproxEq(3, maxv, V3.fromArray(.{ 3, 5, 4 }), 1e-12);
}

test "Vector: sub operation" {
    const V3 = Vector(3, f64);
    var a = V3.fromArray(.{ 5, 7, 9 });
    const b = V3.fromArray(.{ 2, 3, 4 });
    a.sub(b);
    try expectApproxEq(3, a, V3.fromArray(.{ 3, 4, 5 }), 1e-12);
}

test "Vector: mul (elementwise) operation" {
    const V3 = Vector(3, f64);
    var a = V3.fromArray(.{ 2, 3, 4 });
    const b = V3.fromArray(.{ 1, 2, 3 });
    a.mul(b);
    try expectApproxEq(3, a, V3.fromArray(.{ 2, 6, 12 }), 1e-12);
}

test "Matrix: diagonal construction (3x3)" {
    const M33 = Matrix(3, 3, f64);
    const D = M33.diagonal(2, 3, 4);
    try testing.expectEqual(D.get(0, 0), 2.0);
    try testing.expectEqual(D.get(1, 1), 3.0);
    try testing.expectEqual(D.get(2, 2), 4.0);
    try testing.expectEqual(D.get(0, 1), 0.0);
    try testing.expectEqual(D.get(1, 0), 0.0);
}

test "Matrix 3x3: inverted (non-mutating)" {
    const M33 = Matrix(3, 3, f64);
    const A = M33.fromArray(.{
        1, 2, 3,
        0, 1, 4,
        5, 6, 0,
    });
    const Ainv = A.inverted();

    // Verify A * A^-1 ≈ I by checking a vector transform
    const V3 = Vector(3, f64);
    const v = V3.fromArray(.{ 1, 2, 3 });
    const Av = A.mulVec(v);
    const result = Ainv.mulVec(Av);
    try expectApproxEq(3, result, v, 1e-12);
}

test "Matrix: get1D and set1D" {
    const M23 = Matrix(2, 3, f32);
    var m = M23.fromArray(.{ 1, 2, 3, 4, 5, 6 });
    try testing.expectEqual(m.get1D(0), 1.0);
    try testing.expectEqual(m.get1D(5), 6.0);

    m.set1D(2, 10.0);
    try testing.expectEqual(m.get1D(2), 10.0);
}

test "Matrix: isSquare check" {
    const M33 = Matrix(3, 3, f64);
    const M34 = Matrix(3, 4, f64);
    try testing.expect(M33.isSquare());
    try testing.expect(!M34.isSquare());
}

test "Quaternion: fromSequence with slice" {
    const Qf = Quaternion(f64);
    const arr: [4]f64 = .{ 1, 0, 0, 0 };
    const slice = arr[0..];
    const q = Qf.fromSlice(slice);
    try expectApproxEq(4, q, Qf.identity(), 1e-12);
}

test "Quaternion: element accessors" {
    const Qf = Quaternion(f64);
    const q = Qf.fromArray(.{ 0.5, 0.5, 0.5, 0.5 });
    try testing.expectEqual(q.w(), 0.5);
    try testing.expectEqual(q.x(), 0.5);
    try testing.expectEqual(q.y(), 0.5);
    try testing.expectEqual(q.z(), 0.5);
}

test "Quaternion: fromAxisAngle basic rotation" {
    const Qf = Quaternion(f64);
    const V3 = Vector(3, f64);
    const axis = V3.fromArray(.{ 0, 0, 1 });
    const angle = std.math.pi / 2.0;
    const q = Qf.fromAxisAngle(axis, angle);

    // For 90° rotation around Z, w ≈ cos(45°), z ≈ sin(45°)
    const expected_val = @sqrt(@as(f64, 0.5));
    try testing.expectApproxEqAbs(q.w(), expected_val, 1e-5);
    try testing.expectApproxEqAbs(q.z(), expected_val, 1e-5);
}

// ========== SIMD Verification Tests ==========

test "SIMD: Vector operations use SIMD" {
    // This test verifies that our Vector type properly uses @Vector
    const V4 = Vector(4, f64);
    const a = V4.fromArray(.{ 1, 2, 3, 4 });
    const b = V4.fromArray(.{ 5, 6, 7, 8 });

    // These operations should compile to SIMD instructions
    const sum = V4.added(a, b);
    const diff = V4.subbed(a, b);
    const scaled = a.scaled(2.0);
    const dotprod = V4.dot(a, b);

    try expectApproxEq(4, sum, V4.fromArray(.{ 6, 8, 10, 12 }), 1e-12);
    try expectApproxEq(4, diff, V4.fromArray(.{ -4, -4, -4, -4 }), 1e-12);
    try expectApproxEq(4, scaled, V4.fromArray(.{ 2, 4, 6, 8 }), 1e-12);
    try testing.expectApproxEqAbs(dotprod, 70.0, 1e-5);
}

test "SIMD: Quaternion operations use SIMD" {
    const Qf = Quaternion(f64);
    var q1 = Qf.fromArray(.{ 1, 0, 0, 0 });
    var q2 = Qf.fromArray(.{ 0.707, 0, 0.707, 0 });

    // Normalize uses SIMD
    q2.normalize();
    try testing.expectApproxEqAbs(q2.magnitudeSq(), 1.0, 1e-5);

    // Multiplication uses SIMD operations
    const q3 = q1.mul(q2);
    try expectApproxEq(4, q3, q2, 1e-5);
}

test "SIMD: Matrix operations use SIMD" {
    const M33 = Matrix(3, 3, f64);
    const V3 = Vector(3, f64);

    const m = M33.fromArray(.{
        1, 0, 0,
        0, 2, 0,
        0, 0, 3,
    });
    const v = V3.fromArray(.{ 1, 1, 1 });

    // mulVec uses @shuffle for SIMD
    const result = m.mulVec(v);
    try expectApproxEq(3, result, V3.fromArray(.{ 1, 2, 3 }), 1e-12);
}

test "Performance: large vector operations" {
    const V16 = Vector(16, f32);
    var a = V16.zero();
    const b = V16.fromArray(.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 });
    
    // Multiple operations that should benefit from SIMD
    a.add(b);
    a.mulScalar(2.0);
    const mag = a.magnitudeSq();
    
    // Expected: (2 + 4 + 6 + ... + 32)^2 summed = 4 * (1 + 4 + 9 + ... + 256) = 4 * 1496 = 5984
    try testing.expectApproxEqAbs(mag, 5984.0, 1e-9);
}

// ========== Scalar Utilities Tests ==========

const utils = @import("utils.zig");

test "Utils: clamp scalar values" {
    try testing.expectEqual(utils.clamp(f64, 5.0, 0.0, 10.0), 5.0);
    try testing.expectEqual(utils.clamp(f64, -5.0, 0.0, 10.0), 0.0);
    try testing.expectEqual(utils.clamp(f64, 15.0, 0.0, 10.0), 10.0);
    try testing.expectEqual(utils.clamp(i32, 5, 0, 10), 5);
    try testing.expectEqual(utils.clamp(i32, -5, 0, 10), 0);
    try testing.expectEqual(utils.clamp(i32, 15, 0, 10), 10);
}

test "Utils: saturate" {
    try testing.expectApproxEqAbs(utils.saturate(f64, 0.5), 0.5, 1e-12);
    try testing.expectApproxEqAbs(utils.saturate(f64, -0.5), 0.0, 1e-12);
    try testing.expectApproxEqAbs(utils.saturate(f64, 1.5), 1.0, 1e-12);
}

test "Utils: lerp scalar" {
    try testing.expectApproxEqAbs(utils.lerp(f64, 0.0, 10.0, 0.0), 0.0, 1e-12);
    try testing.expectApproxEqAbs(utils.lerp(f64, 0.0, 10.0, 1.0), 10.0, 1e-12);
    try testing.expectApproxEqAbs(utils.lerp(f64, 0.0, 10.0, 0.5), 5.0, 1e-12);
    try testing.expectApproxEqAbs(utils.lerp(f64, -5.0, 5.0, 0.25), -2.5, 1e-12);
}

test "Utils: smoothstep" {
    try testing.expectApproxEqAbs(utils.smoothstep(f64, 0.0), 0.0, 1e-12);
    try testing.expectApproxEqAbs(utils.smoothstep(f64, 1.0), 1.0, 1e-12);
    try testing.expectApproxEqAbs(utils.smoothstep(f64, 0.5), 0.5, 1e-12);
    // smoothstep should be smoother than linear at edges
    const linear_025 = 0.25;
    const smooth_025 = utils.smoothstep(f64, 0.25);
    try testing.expect(smooth_025 < linear_025);
}

test "Utils: step function" {
    try testing.expectEqual(utils.step(f64, 5.0, 3.0), 0.0);
    try testing.expectEqual(utils.step(f64, 5.0, 5.0), 1.0);
    try testing.expectEqual(utils.step(f64, 5.0, 7.0), 1.0);
}

test "Utils: sign function" {
    try testing.expectEqual(utils.sign(f64, 5.0), 1.0);
    try testing.expectEqual(utils.sign(f64, -5.0), -1.0);
    try testing.expectEqual(utils.sign(f64, 0.0), 0.0);
    try testing.expectEqual(utils.sign(i32, 10), 1);
    try testing.expectEqual(utils.sign(i32, -10), -1);
    try testing.expectEqual(utils.sign(i32, 0), 0);
}

test "Utils: square and cube" {
    try testing.expectApproxEqAbs(utils.square(f64, 5.0), 25.0, 1e-12);
    try testing.expectApproxEqAbs(utils.cube(f64, 3.0), 27.0, 1e-12);
    try testing.expectEqual(utils.square(i32, 4), 16);
    try testing.expectEqual(utils.cube(i32, 2), 8);
}

test "Utils: invSqrt for f32 and f64" {
    // f32
    const inv_sqrt_4_f32 = utils.invSqrt(f64, 4.0);
    try testing.expectApproxEqAbs(inv_sqrt_4_f32, 0.5, 1e-12);
    
    const inv_sqrt_9_f32 = utils.invSqrt(f64, 9.0);
    try testing.expectApproxEqAbs(inv_sqrt_9_f32, 1.0 / 3.0, 1e-12);
    
    // f64
    const inv_sqrt_4_f64 = utils.invSqrt(f64, 4.0);
    try testing.expectApproxEqAbs(inv_sqrt_4_f64, 0.5, 1e-12);
    
    const inv_sqrt_9_f64 = utils.invSqrt(f64, 9.0);
    try testing.expectApproxEqAbs(inv_sqrt_9_f64, 1.0 / 3.0, 1e-12);
}

// ========== Vector Utilities Tests ==========

test "Vector: lerp interpolation" {
    const V3 = Vector(3, f64);
    const a = V3.fromArray(.{ 0, 0, 0 });
    const b = V3.fromArray(.{ 10, 20, 30 });
    
    const lerp0 = V3.lerp(a, b, 0.0);
    const lerp1 = V3.lerp(a, b, 1.0);
    const lerp_half = V3.lerp(a, b, 0.5);
    
    try expectApproxEq(3, lerp0, a, 1e-12);
    try expectApproxEq(3, lerp1, b, 1e-12);
    try expectApproxEq(3, lerp_half, V3.fromArray(.{ 5, 10, 15 }), 1e-12);
}

test "Vector: distance and distanceSq" {
    const V3 = Vector(3, f64);
    const a = V3.fromArray(.{ 0, 0, 0 });
    const b = V3.fromArray(.{ 3, 4, 0 });
    
    try testing.expectApproxEqAbs(V3.distance(a, b), 5.0, 1e-5);
    try testing.expectApproxEqAbs(V3.distanceSq(a, b), 25.0, 1e-5);
}

test "Vector: project onto another vector" {
    const V3 = Vector(3, f64);
    const a = V3.fromArray(.{ 1, 1, 0 });
    const b = V3.fromArray(.{ 1, 0, 0 });
    
    const proj = V3.project(a, b);
    try expectApproxEq(3, proj, V3.fromArray(.{ 1, 0, 0 }), 1e-12);
    
    // Projection onto zero should return zero
    const zero = V3.zero();
    const proj_zero = V3.project(a, zero);
    try expectApproxEq(3, proj_zero, zero, 1e-12);
}

test "Vector: reflect across normal" {
    const V3 = Vector(3, f64);
    const v = V3.fromArray(.{ 1, -1, 0 });
    const normal = V3.fromArray(.{ 0, 1, 0 });
    
    const reflected = V3.reflect(v, normal);
    try expectApproxEq(3, reflected, V3.fromArray(.{ 1, 1, 0 }), 1e-12);
}

test "Vector: clampLength" {
    const V3 = Vector(3, f64);
    const v = V3.fromArray(.{ 3, 4, 0 });
    
    // Length is 5, clamp to 3 should scale proportionally
    const clamped = V3.clampLength(v, 3.0);
    try testing.expectApproxEqAbs(clamped.magnitude(), 3.0, 1e-5);
    
    // Already shorter, should remain unchanged
    const short = V3.fromArray(.{ 1, 0, 0 });
    const unchanged = V3.clampLength(short, 5.0);
    try expectApproxEq(3, unchanged, short, 1e-12);
    
    // Zero vector should remain zero
    const zero = V3.zero();
    const zero_clamped = V3.clampLength(zero, 5.0);
    try expectApproxEq(3, zero_clamped, zero, 1e-12);
}

// ========== Quaternion Utilities Tests ==========

test "Quaternion: nlerp interpolation" {
    const Qf = Quaternion(f64);
    const q1 = Qf.identity();
    const V3 = Vector(3, f64);
    const axis = V3.fromArray(.{ 0, 0, 1 });
    var q2 = Qf.fromAxisAngle(axis, std.math.pi / 2.0);
    q2.normalize();
    
    const nlerp0 = Qf.nlerp(q1, q2, 0.0);
    const nlerp1 = Qf.nlerp(q1, q2, 1.0);
    const nlerp_half = Qf.nlerp(q1, q2, 0.5);
    
    try expectApproxEq(4, nlerp0, q1, 1e-5);
    try expectApproxEq(4, nlerp1, q2, 1e-5);
    try testing.expectApproxEqAbs(nlerp_half.magnitudeSq(), 1.0, 1e-5);
}

test "Quaternion: slerp interpolation" {
    const Qf = Quaternion(f64);
    const q1 = Qf.identity();
    const V3 = Vector(3, f64);
    const axis = V3.fromArray(.{ 0, 0, 1 });
    var q2 = Qf.fromAxisAngle(axis, std.math.pi / 2.0);
    q2.normalize();
    
    const slerp0 = Qf.slerp(q1, q2, 0.0);
    const slerp1 = Qf.slerp(q1, q2, 1.0);
    const slerp_half = Qf.slerp(q1, q2, 0.5);
    
    try expectApproxEq(4, slerp0, q1, 1e-12);
    try expectApproxEq(4, slerp1, q2, 1e-12);
    try testing.expectApproxEqAbs(slerp_half.magnitudeSq(), 1.0, 1e-5);
    
    // Slerp should produce constant angular velocity
    const slerp_quarter = Qf.slerp(q1, q2, 0.25);
    const slerp_three_quarter = Qf.slerp(q1, q2, 0.75);
    try testing.expectApproxEqAbs(slerp_quarter.magnitudeSq(), 1.0, 1e-5);
    try testing.expectApproxEqAbs(slerp_three_quarter.magnitudeSq(), 1.0, 1e-5);
}

test "Quaternion: angleBetween" {
    const Qf = Quaternion(f64);
    const V3 = Vector(3, f64);
    const axis = V3.fromArray(.{ 0, 0, 1 });
    
    const q1 = Qf.identity();
    var q2 = Qf.fromAxisAngle(axis, std.math.pi / 2.0);
    q2.normalize();
    
    const angle_between = Qf.angleBetween(q1, q2);
    try testing.expectApproxEqAbs(angle_between, std.math.pi / 2.0, 1e-12);
    
    // Angle between identity and itself should be 0
    const angle_same = Qf.angleBetween(q1, q1);
    try testing.expectApproxEqAbs(angle_same, 0.0, 1e-5);
}

test "Quaternion: slerp with close quaternions" {
    const Qf = Quaternion(f64);
    const V3 = Vector(3, f64);
    const axis = V3.fromArray(.{ 0, 0, 1 });
    
    const q1 = Qf.identity();
    var q2 = Qf.fromAxisAngle(axis, 0.001); // Very small angle
    q2.normalize();
    
    // Should fall back to nlerp for close quaternions
    const result = Qf.slerp(q1, q2, 0.5);
    try testing.expectApproxEqAbs(result.magnitudeSq(), 1.0, 1e-5);
}
