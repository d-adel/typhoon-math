//! Typhoon Math: vector, matrix, quaternion, and geometry utilities shared across crates.

pub const batch = @import("batch.zig");
pub const Vec3Batch = batch.Vec3Batch;
pub const Mat3Batch = batch.Mat3Batch;
pub const QuatBatch = batch.QuatBatch;
pub const geometry = @import("geometry.zig");
pub const matrix = @import("matrix.zig");
pub const Matrix = matrix.Matrix;
pub const quaternion = @import("quaternion.zig");
pub const Quaternion = quaternion.Quaternion;
pub const support = @import("support.zig");
pub const tests = @import("tests.zig");
pub const utils = @import("utils.zig");
pub const vector = @import("vector.zig");
pub const Vector = vector.Vector;

test {
    _ = @import("tests.zig");
}
