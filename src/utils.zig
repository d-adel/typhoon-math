const std = @import("std");

pub inline fn clamp(comptime T: type, value: T, min_val: T, max_val: T) T {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

pub inline fn saturate(comptime T: type, value: T) T {
    return clamp(T, value, @as(T, 0), @as(T, 1));
}

pub inline fn lerp(comptime T: type, a: T, b: T, t: T) T {
    return a + (b - a) * t;
}

pub inline fn smoothstep(comptime T: type, t: T) T {
    const clamped = saturate(T, t);
    return clamped * clamped * (@as(T, 3) - @as(T, 2) * clamped);
}

pub inline fn step(comptime T: type, edge: T, x: T) T {
    return if (x < edge) @as(T, 0) else @as(T, 1);
}

pub inline fn sign(comptime T: type, x: T) T {
    if (x > 0) return @as(T, 1);
    if (x < 0) return @as(T, -1);
    return @as(T, 0);
}

pub inline fn square(comptime T: type, x: T) T {
    return x * x;
}

pub inline fn cube(comptime T: type, x: T) T {
    return x * x * x;
}

pub inline fn invSqrt(comptime T: type, x: T) T {
    return @as(T, 1.0) / @sqrt(x);
}
