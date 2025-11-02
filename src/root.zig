//! Typhoon Math: vector, matrix, and quaternion utilities shared across crates.

pub const matrix = @import("matrix.zig");
pub const Matrix = matrix.Matrix;
pub const quaternion = @import("quaternion.zig");
pub const Quaternion = quaternion.Quaternion;
pub const tests = @import("tests.zig");
pub const utils = @import("utils.zig");
pub const vector = @import("vector.zig");
pub const Vector = vector.Vector;

test {
    _ = @import("tests.zig");
}
