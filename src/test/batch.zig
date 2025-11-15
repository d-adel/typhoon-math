const std = @import("std");
const testing = std.testing;

const batch = @import("../batch.zig");
const Vector = @import("../vector.zig").Vector;
const Matrix = @import("../matrix.zig").Matrix;

test "Vec3Batch add/dot/toVectors round-trip" {
    const Vec3 = Vector(3, f64);
    const Batch = batch.Vec3Batch(4, f64);

    const inputs_a = [_]Vec3{
        Vec3.fromArray(.{ 1, 2, 3 }),
        Vec3.fromArray(.{ 4, 5, 6 }),
        Vec3.fromArray(.{ -1, 0.5, 2 }),
        Vec3.fromArray(.{ 0, 0, 1 }),
    };
    const inputs_b = [_]Vec3{
        Vec3.fromArray(.{ -2, 1, 0 }),
        Vec3.fromArray(.{ 3, -3, 1 }),
        Vec3.fromArray(.{ 0.25, 0.25, 0.25 }),
        Vec3.fromArray(.{ 5, 1, -2 }),
    };

    const batch_a = Batch.fromVectors(inputs_a);
    const batch_b = Batch.fromVectors(inputs_b);
    const batch_sum = Batch.add(batch_a, batch_b);
    const output = batch_sum.toVectors();
    inline for (0..4) |lane| {
        const expected = inputs_a[lane].added(inputs_b[lane]);
        inline for (0..3) |axis| {
            try testing.expectApproxEqAbs(output[lane].data[axis], expected.data[axis], 1e-12);
        }
    }

    const dots = Batch.dot(batch_a, batch_b);
    const dots_array: [4]f64 = @bitCast(dots);
    inline for (0..4) |lane| {
        try testing.expectApproxEqAbs(dots_array[lane], Vector(3, f64).dot(inputs_a[lane], inputs_b[lane]), 1e-12);
    }
}

test "Mat3Batch mulVec matches scalar mat-vec" {
    const Vec3 = Vector(3, f64);
    const Mat3 = Matrix(3, 3, f64);
    const VecBatch = batch.Vec3Batch(4, f64);
    const MatBatch = batch.Mat3Batch(4, f64);

    const mats = [_]Mat3{
        Mat3.identity(),
        Mat3.fromArray(.{ 2, 0, 0, 0, 3, 0, 0, 0, 4 }),
        Mat3.fromArray(.{ 1, 2, 3, 0, 1, 4, 5, 6, 0 }),
        Mat3.fromArray(.{ 0, -1, 2, 3, 0, 1, -2, 4, 0 }),
    };

    const vecs = [_]Vec3{
        Vec3.fromArray(.{ 1, 0, 0 }),
        Vec3.fromArray(.{ -1, 1, 2 }),
        Vec3.fromArray(.{ 0.5, -0.25, 2 }),
        Vec3.fromArray(.{ 3, 3, -1 }),
    };

    const mat_batch = MatBatch.fromMatrices(mats);
    const vec_batch = VecBatch.fromVectors(vecs);
    const result_batch = MatBatch.mulVec(mat_batch, vec_batch);
    const result_vecs = result_batch.toVectors();

    inline for (0..4) |lane| {
        const expected = mats[lane].mulVec(vecs[lane]);
        inline for (0..3) |axis| {
            try testing.expectApproxEqAbs(result_vecs[lane].data[axis], expected.data[axis], 1e-12);
        }
    }
}
