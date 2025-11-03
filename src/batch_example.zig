const std = @import("std");
const typhoon_math = @import("typhoon-math");

/// Example demonstrating the public batch SIMD API

pub fn main() !void {
    const Vec3 = typhoon_math.Vector(3, f32);
    const Mat3 = typhoon_math.Matrix(3, 3, f32);
    const Quat = typhoon_math.Quaternion(f32);

    const BatchSize = 4;
    const Vec3Batch = typhoon_math.Vec3Batch(BatchSize, f32);
    const Mat3Batch = typhoon_math.Mat3Batch(BatchSize, f32);
    const QuatBatch = typhoon_math.QuatBatch(BatchSize, f32);

    std.debug.print("\n=== Typhoon Math Batch API Example ===\n\n", .{});

    // Example 1: Batch vector operations
    {
        std.debug.print("1. Batch Vector Operations:\n", .{});
        var vectors: [BatchSize]Vec3 = undefined;
        for (&vectors, 0..) |*v, i| {
            const fi: f32 = @floatFromInt(i);
            v.* = Vec3.fromArray(.{ fi + 1.0, fi + 2.0, fi + 3.0 });
        }

        const batch = Vec3Batch.fromVectors(vectors);
        const normalized = Vec3Batch.normalize(batch);
        const dot_products = Vec3Batch.dot(batch, normalized);

        std.debug.print("   Input vectors: 4 vectors processed in parallel\n", .{});
        std.debug.print("   Dot products (batch result): ", .{});
        const results: [BatchSize]f32 = @bitCast(dot_products);
        for (results) |r| {
            std.debug.print("{d:.3} ", .{r});
        }
        std.debug.print("\n\n", .{});
    }

    // Example 2: Batch matrix operations
    {
        std.debug.print("2. Batch Matrix Operations:\n", .{});
        var matrices: [BatchSize]Mat3 = undefined;
        for (&matrices, 0..) |*m, i| {
            const fi: f32 = @floatFromInt(i + 1);
            const scale = fi * 0.1;
            m.* = Mat3.fromArray(.{
                scale, 0.0,   0.0,
                0.0,   scale, 0.0,
                0.0,   0.0,   scale,
            });
        }

        const batch_a = Mat3Batch.fromMatrices(matrices);
        const batch_b = Mat3Batch.fromMatrices(matrices);
        const product = Mat3Batch.mul(batch_a, batch_b);

        std.debug.print("   Multiplied 4 diagonal matrices in parallel\n", .{});
        std.debug.print("   Result diagonal elements (m00): ", .{});
        const m00_results: [BatchSize]f32 = @bitCast(product.m00);
        for (m00_results) |r| {
            std.debug.print("{d:.4} ", .{r});
        }
        std.debug.print("\n\n", .{});
    }

    // Example 3: Batch quaternion operations
    {
        std.debug.print("3. Batch Quaternion Operations:\n", .{});
        var quaternions: [BatchSize]Quat = undefined;
        for (&quaternions, 0..) |*q, i| {
            const angle = @as(f32, @floatFromInt(i)) * std.math.pi / 4.0;
            const axis = Vec3.fromArray(.{ 0.0, 0.0, 1.0 }); // Z-axis rotation
            q.* = Quat.fromAxisAngle(axis, angle);
        }

        const quat_batch = QuatBatch.fromQuaternions(quaternions);
        const normalized_quats = QuatBatch.normalize(quat_batch);

        // Rotate vectors in batch
        var test_vectors: [BatchSize]Vec3 = undefined;
        for (&test_vectors) |*v| {
            v.* = Vec3.fromArray(.{ 1.0, 0.0, 0.0 });
        }

        const vec_batch = Vec3Batch.fromVectors(test_vectors);
        const rotated = QuatBatch.rotate(normalized_quats, vec_batch);

        std.debug.print("   Rotated 4 vectors around Z-axis in parallel\n", .{});
        std.debug.print("   Resulting X components: ", .{});
        const x_results: [BatchSize]f32 = @bitCast(rotated.x);
        for (x_results) |r| {
            std.debug.print("{d:.3} ", .{r});
        }
        std.debug.print("\n\n", .{});
    }

    std.debug.print("=== Key Benefits ===\n", .{});
    std.debug.print("- Process {d} elements simultaneously using SIMD\n", .{BatchSize});
    std.debug.print("- 2-6x speedup over scalar operations (see benchmark)\n", .{});
    std.debug.print("- Type-safe generic API: Vec3Batch(N, T), Mat3Batch(N, T), QuatBatch(N, T)\n", .{});
    std.debug.print("- Structure-of-Arrays layout for optimal cache performance\n", .{});
    std.debug.print("\n", .{});
}
