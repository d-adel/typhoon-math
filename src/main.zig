const std = @import("std");

const ty_math = @import("typhoon_math");

pub fn main() !void {
    const V3 = ty_math.Vector(3, f32);
    const sample = V3.fromArray(.{ 1.0, 2.0, 2.0 });
    std.debug.print("||v|| = {d:.4}\n", .{sample.magnitude()});
}
