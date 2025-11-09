// High-level batching API for SIMD processing

const std = @import("std");

const batch = @import("batch.zig");

/// Generic batch processor that handles gathering, batching, processing, and remainder handling
pub fn BatchProcessor(comptime BatchSize: comptime_int, comptime T: type) type {
    return struct {
        const Self = @This();

        items: [BatchSize]T = undefined,
        count: usize = 0,

        pub fn init() Self {
            return .{ .items = undefined, .count = 0 };
        }

        /// Add item to batch. Returns true if batch is full.
        /// Auto-resets when full batch was previously processed.
        pub inline fn add(self: *Self, item: T) bool {
            if (self.count == BatchSize) {
                self.count = 0;
            }
            self.items[self.count] = item;
            self.count += 1;
            return self.count == BatchSize;
        }

        pub inline fn isFull(self: *const Self) bool {
            return self.count == BatchSize;
        }

        pub inline fn processFull(self: *Self, comptime func: anytype, args: anytype) void {
            if (self.isFull()) {
                @call(.auto, func, .{self.items} ++ args);
                self.count = 0;
            }
        }

        pub inline fn slice(self: *const Self) []const T {
            const actual_count = if (self.count == BatchSize) 0 else self.count;
            return self.items[0..actual_count];
        }

        pub inline fn array(self: *const Self) [BatchSize]T {
            return self.items;
        }

        pub inline fn reset(self: *Self) void {
            self.count = 0;
        }

        pub inline fn hasItems(self: *const Self) bool {
            return self.count > 0;
        }
    };
}

/// Specialized batch processor for Vector operations with SIMD support
pub fn VecBatchProcessor(comptime N: comptime_int, comptime BatchSize: comptime_int, comptime T: type) type {
    const VecType = batch.Vector(N, T);

    const use_vec3_batch = (N == 3);
    const VecBatchType = if (use_vec3_batch) batch.Vec3Batch(BatchSize, T) else struct {
        vectors: [BatchSize]@Vector(N, T),
        pub fn fromVectors(vecs: [BatchSize]VecType) @This() {
            var result: @This() = undefined;
            inline for (0..BatchSize) |i| {
                result.vectors[i] = vecs[i].data;
            }
            return result;
        }
        pub fn toVectors(self: @This()) [BatchSize]VecType {
            var result: [BatchSize]VecType = undefined;
            inline for (0..BatchSize) |i| {
                result[i] = VecType.init(self.vectors[i]);
            }
            return result;
        }
    };

    return struct {
        const Self = @This();

        vectors: [BatchSize]VecType = undefined,
        count: usize = 0,

        pub fn init() Self {
            return .{ .vectors = undefined, .count = 0 };
        }

        pub inline fn push(self: *Self, vec: VecType) bool {
            if (self.count == BatchSize) {
                self.count = 0;
            }
            self.vectors[self.count] = vec;
            self.count += 1;
            return self.count == BatchSize;
        }

        pub inline fn toBatch(self: *const Self) VecBatchType {
            return VecBatchType.fromVectors(self.vectors);
        }

        pub inline fn array(self: *const Self) [BatchSize]VecType {
            return self.vectors;
        }

        pub inline fn slice(self: *const Self) []const VecType {
            const actual_count = if (self.count == BatchSize) 0 else self.count;
            return self.vectors[0..actual_count];
        }

        pub inline fn applyBinary(
            self: *Self,
            other: [BatchSize]VecType,
            comptime op: anytype,
        ) [BatchSize]VecType {
            const batch_a = self.toBatch();
            const batch_b = VecBatchType.fromVectors(other);
            const result = op(batch_a, batch_b);
            return result.toVectors();
        }

        pub inline fn applyUnary(
            self: *Self,
            comptime op: anytype,
        ) [BatchSize]VecType {
            const batch_val = self.toBatch();
            const result = op(batch_val);
            return result.toVectors();
        }

        pub inline fn mulScalar(self: *Self, scalars: @Vector(BatchSize, T)) [BatchSize]VecType {
            const batch_val = self.toBatch();
            const result = VecBatchType.mulScalar(batch_val, scalars);
            return result.toVectors();
        }

        pub inline fn addVecs(self: *Self, other: [BatchSize]VecType) [BatchSize]VecType {
            return self.applyBinary(other, VecBatchType.add);
        }

        pub inline fn subVecs(self: *Self, other: [BatchSize]VecType) [BatchSize]VecType {
            return self.applyBinary(other, VecBatchType.sub);
        }

        pub inline fn reset(self: *Self) void {
            self.count = 0;
        }
    };
}

pub fn Vec3BatchProcessor(comptime BatchSize: comptime_int, comptime T: type) type {
    return VecBatchProcessor(3, BatchSize, T);
}

pub fn IndexedBatchProcessor(comptime BatchSize: comptime_int, comptime T: type) type {
    return struct {
        const Self = @This();

        pub const Entry = struct {
            item: T,
            index: usize,
        };

        items: [BatchSize]T = undefined,
        indices: [BatchSize]usize = undefined,
        count: usize = 0,

        pub fn init() Self {
            return .{
                .items = undefined,
                .indices = undefined,
                .count = 0,
            };
        }

        pub inline fn add(self: *Self, item: T, index: usize) bool {
            if (self.count == BatchSize) {
                self.count = 0;
            }
            self.items[self.count] = item;
            self.indices[self.count] = index;
            self.count += 1;
            return self.count == BatchSize;
        }

        pub inline fn reset(self: *Self) void {
            self.count = 0;
        }

        pub inline fn slice(self: *const Self) []const Entry {
            const actual_count = if (self.count == BatchSize) 0 else self.count;
            var result: [BatchSize]Entry = undefined;
            for (0..actual_count) |i| {
                result[i] = .{ .item = self.items[i], .index = self.indices[i] };
            }
            return result[0..actual_count];
        }

        pub inline fn array(self: *const Self) [BatchSize]Entry {
            var result: [BatchSize]Entry = undefined;
            inline for (0..BatchSize) |i| {
                result[i] = .{ .item = self.items[i], .index = self.indices[i] };
            }
            return result;
        }

        pub inline fn itemsSlice(self: *const Self) []const T {
            const actual_count = if (self.count == BatchSize) 0 else self.count;
            return self.items[0..actual_count];
        }

        pub inline fn indicesSlice(self: *const Self) []const usize {
            const actual_count = if (self.count == BatchSize) 0 else self.count;
            return self.indices[0..actual_count];
        }

        pub inline fn itemsArray(self: *const Self) [BatchSize]T {
            return self.items;
        }

        pub inline fn indicesArray(self: *const Self) [BatchSize]usize {
            return self.indices;
        }
    };
}

/// Batch iterator for functional-style processing
pub fn BatchIterator(comptime BatchSize: comptime_int) type {
    return struct {
        const Self = @This();

        start: usize,
        end: usize,

        pub fn init(len: usize) Self {
            return .{ .start = 0, .end = len };
        }

        pub fn forEach(self: Self, comptime T: type, items: []const T, comptime func: anytype, args: anytype) void {
            const full_batches = (self.end - self.start) / BatchSize;

            var batch_idx: usize = 0;
            while (batch_idx < full_batches) : (batch_idx += 1) {
                const base = self.start + batch_idx * BatchSize;
                var batch_items: [BatchSize]T = undefined;
                inline for (0..BatchSize) |i| {
                    batch_items[i] = items[base + i];
                }
                @call(.auto, func, .{batch_items} ++ args);
            }

            const remainder_start = self.start + full_batches * BatchSize;
            for (items[remainder_start..self.end]) |item| {
                @call(.auto, func, .{[1]T{item}} ++ args);
            }
        }

        pub fn forEachFiltered(
            self: Self,
            comptime T: type,
            items: []const T,
            comptime predicate: anytype,
            comptime func: anytype,
            args: anytype,
        ) void {
            var proc = BatchProcessor(BatchSize, T).init();
            for (items[self.start..self.end]) |item| {
                if (@call(.auto, predicate, .{item})) {
                    if (proc.add(item)) {
                        @call(.auto, func, .{proc.array()} ++ args);
                    }
                }
            }
            for (proc.slice()) |item| {
                @call(.auto, func, .{[1]T{item}} ++ args);
            }
        }
    };
}

/// One-shot batch operations namespace
pub const batchOps = struct {
    pub fn vec3Add(comptime BatchSize: comptime_int, comptime T: type, a: [BatchSize]batch.Vector(3, T), b: [BatchSize]batch.Vector(3, T)) [BatchSize]batch.Vector(3, T) {
        const batch_a = batch.Vec3Batch(BatchSize, T).fromVectors(a);
        const batch_b = batch.Vec3Batch(BatchSize, T).fromVectors(b);
        return batch_a.add(batch_b).toVectors();
    }

    pub fn vec3Sub(comptime BatchSize: comptime_int, comptime T: type, a: [BatchSize]batch.Vector(3, T), b: [BatchSize]batch.Vector(3, T)) [BatchSize]batch.Vector(3, T) {
        const batch_a = batch.Vec3Batch(BatchSize, T).fromVectors(a);
        const batch_b = batch.Vec3Batch(BatchSize, T).fromVectors(b);
        return batch_a.sub(batch_b).toVectors();
    }

    pub fn vec3MulScalar(comptime BatchSize: comptime_int, comptime T: type, a: [BatchSize]batch.Vector(3, T), scalars: @Vector(BatchSize, T)) [BatchSize]batch.Vector(3, T) {
        const batch_a = batch.Vec3Batch(BatchSize, T).fromVectors(a);
        return batch_a.mulScalar(scalars).toVectors();
    }
};
