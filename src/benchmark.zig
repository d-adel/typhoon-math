// RUN THE COMMAND ZIG BUILD BENCHMARK
const std = @import("std");
const math = std.math;
const Random = std.Random;
const sort = std.sort;

const typhoon_math = @import("root.zig");

const Vec3 = typhoon_math.Vector(3, f32);
const Mat3 = typhoon_math.Matrix(3, 3, f32);
const Quat = typhoon_math.Quaternion(f32);

const SampleCount = 6;
const WarmupDivisor = 16;
const NanosecondsPerMillisecond = 1_000_000.0;
const Batch = 4;

const VectorPairCount = 256;
const VectorEntryCount = 256;
const MatrixPairCount = 64;
const MatrixVecPairCount = 64;
const QuaternionPairCount = 128;
const QuaternionVecPairCount = 128;
const VectorPairBatchCount = VectorPairCount / Batch;
const VectorEntryBatchCount = VectorEntryCount / Batch;
const MatrixPairBatchCount = MatrixPairCount / Batch;
const MatrixVecPairBatchCount = MatrixVecPairCount / Batch;
const QuaternionPairBatchCount = QuaternionPairCount / Batch;
const QuaternionVecPairBatchCount = QuaternionVecPairCount / Batch;

const VectorPair = struct {
    simd_a: Vec3,
    simd_b: Vec3,
    scalar_a: ScalarVec3,
    scalar_b: ScalarVec3,
};

const VectorEntry = struct {
    simd: Vec3,
    scalar: ScalarVec3,
};

const MatrixPair = struct {
    simd_a: Mat3,
    simd_b: Mat3,
    scalar_a: ScalarMat3,
    scalar_b: ScalarMat3,
};

const MatrixVecPair = struct {
    simd_m: Mat3,
    simd_v: Vec3,
    scalar_m: ScalarMat3,
    scalar_v: ScalarVec3,
};

const QuaternionPair = struct {
    simd_a: Quat,
    simd_b: Quat,
    scalar_a: ScalarQuat,
    scalar_b: ScalarQuat,
};

const QuaternionVecPair = struct {
    simd_q: Quat,
    simd_v: Vec3,
    scalar_q: ScalarQuat,
    scalar_v: ScalarVec3,
};

const Vec3Batch = struct {
    x: @Vector(Batch, f32),
    y: @Vector(Batch, f32),
    z: @Vector(Batch, f32),
};

const Vec3PairBatch = struct {
    a: Vec3Batch,
    b: Vec3Batch,
};

const Vec3EntryBatch = struct {
    value: Vec3Batch,
};

const Mat3Batch = struct {
    m00: @Vector(Batch, f32),
    m01: @Vector(Batch, f32),
    m02: @Vector(Batch, f32),
    m10: @Vector(Batch, f32),
    m11: @Vector(Batch, f32),
    m12: @Vector(Batch, f32),
    m20: @Vector(Batch, f32),
    m21: @Vector(Batch, f32),
    m22: @Vector(Batch, f32),
};

const Mat3PairBatch = struct {
    a: Mat3Batch,
    b: Mat3Batch,
};

const MatVecBatch = struct {
    m: Mat3Batch,
    v: Vec3Batch,
};

const QuatBatch = struct {
    w: @Vector(Batch, f32),
    x: @Vector(Batch, f32),
    y: @Vector(Batch, f32),
    z: @Vector(Batch, f32),
};

const QuatPairBatch = struct {
    a: QuatBatch,
    b: QuatBatch,
};

const QuatVecBatch = struct {
    q: QuatBatch,
    v: Vec3Batch,
};

var vector_pairs: [VectorPairCount]VectorPair = undefined;
var vector_entries: [VectorEntryCount]VectorEntry = undefined;
var matrix_pairs: [MatrixPairCount]MatrixPair = undefined;
var matrix_vec_pairs: [MatrixVecPairCount]MatrixVecPair = undefined;
var quaternion_pairs: [QuaternionPairCount]QuaternionPair = undefined;
var quaternion_vec_pairs: [QuaternionVecPairCount]QuaternionVecPair = undefined;

var vector_pair_batches: [VectorPairBatchCount]Vec3PairBatch = undefined;
var vector_entry_batches: [VectorEntryBatchCount]Vec3EntryBatch = undefined;
var matrix_pair_batches: [MatrixPairBatchCount]Mat3PairBatch = undefined;
var matrix_vec_batches: [MatrixVecPairBatchCount]MatVecBatch = undefined;
var quaternion_pair_batches: [QuaternionPairBatchCount]QuatPairBatch = undefined;
var quaternion_vec_batches: [QuaternionVecPairBatchCount]QuatVecBatch = undefined;

const ScalarVec3 = struct {
    x: f32,
    y: f32,
    z: f32,

    fn init(x: f32, y: f32, z: f32) ScalarVec3 {
        return .{ .x = x, .y = y, .z = z };
    }

    fn add(a: ScalarVec3, b: ScalarVec3) ScalarVec3 {
        return .{ .x = a.x + b.x, .y = a.y + b.y, .z = a.z + b.z };
    }

    fn dot(a: ScalarVec3, b: ScalarVec3) f32 {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    fn cross(a: ScalarVec3, b: ScalarVec3) ScalarVec3 {
        return .{
            .x = a.y * b.z - a.z * b.y,
            .y = a.z * b.x - a.x * b.z,
            .z = a.x * b.y - a.y * b.x,
        };
    }

    fn normalize(v: ScalarVec3) ScalarVec3 {
        const len = @sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        const inv_len = 1.0 / len;
        return .{ .x = v.x * inv_len, .y = v.y * inv_len, .z = v.z * inv_len };
    }
};

const ScalarMat3 = struct {
    data: [9]f32,

    fn init(values: [9]f32) ScalarMat3 {
        return .{ .data = values };
    }

    fn mulMat(a: ScalarMat3, b: ScalarMat3) ScalarMat3 {
        var result: ScalarMat3 = undefined;
        inline for (0..3) |i| {
            inline for (0..3) |j| {
                var sum: f32 = 0;
                inline for (0..3) |k| {
                    sum += a.data[i * 3 + k] * b.data[k * 3 + j];
                }
                result.data[i * 3 + j] = sum;
            }
        }
        return result;
    }

    fn mulVec(m: ScalarMat3, v: ScalarVec3) ScalarVec3 {
        return .{
            .x = m.data[0] * v.x + m.data[1] * v.y + m.data[2] * v.z,
            .y = m.data[3] * v.x + m.data[4] * v.y + m.data[5] * v.z,
            .z = m.data[6] * v.x + m.data[7] * v.y + m.data[8] * v.z,
        };
    }
};

const ScalarQuat = struct {
    w: f32,
    x: f32,
    y: f32,
    z: f32,

    fn init(w: f32, x: f32, y: f32, z: f32) ScalarQuat {
        return .{ .w = w, .x = x, .y = y, .z = z };
    }

    fn mul(a: ScalarQuat, b: ScalarQuat) ScalarQuat {
        return .{
            .w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
            .x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            .y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            .z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        };
    }

    fn rotate(q: ScalarQuat, v: ScalarVec3) ScalarVec3 {
        const qv = ScalarVec3.init(q.x, q.y, q.z);
        const uv = ScalarVec3.cross(qv, v);
        const uuv = ScalarVec3.cross(qv, uv);
        const uv_scaled = ScalarVec3.init(uv.x * q.w * 2.0, uv.y * q.w * 2.0, uv.z * q.w * 2.0);
        const uuv_scaled = ScalarVec3.init(uuv.x * 2.0, uuv.y * 2.0, uuv.z * 2.0);
        return ScalarVec3.add(ScalarVec3.add(v, uv_scaled), uuv_scaled);
    }
};

const BenchmarkCase = struct {
    name: []const u8,
    iterations: usize,
    warmup_override: ?usize = null,
    simd_work: WorkFn,
    scalar_work: WorkFn,
};

const BenchmarkSample = struct {
    ns: u64,
    checksum: u32,
};

const BenchmarkStats = struct {
    min_ns: u64,
    max_ns: u64,
    median_ns: u64,
    mean_ns: f64,
    stddev_ns: f64,
};

const WorkFn = *const fn (iterations: usize) u32;

fn initDatasets() void {
    var prng = Random.DefaultPrng.init(0x5182_b7d6_ecec_9c45);
    var random = prng.random();

    for (&vector_pairs) |*entry| {
        const lhs = randomVec(&random);
        const rhs = randomVec(&random);
        entry.simd_a = Vec3.fromArray(lhs);
        entry.simd_b = Vec3.fromArray(rhs);
        entry.scalar_a = ScalarVec3.init(lhs[0], lhs[1], lhs[2]);
        entry.scalar_b = ScalarVec3.init(rhs[0], rhs[1], rhs[2]);
    }

    for (&vector_entries) |*entry| {
        const values = randomVec(&random);
        entry.simd = Vec3.fromArray(values);
        entry.scalar = ScalarVec3.init(values[0], values[1], values[2]);
    }

    for (&matrix_pairs) |*entry| {
        const a = randomMatrix(&random);
        const b = randomMatrix(&random);
        entry.simd_a = Mat3.fromArray(a);
        entry.simd_b = Mat3.fromArray(b);
        entry.scalar_a = ScalarMat3.init(a);
        entry.scalar_b = ScalarMat3.init(b);
    }

    for (&matrix_vec_pairs) |*entry| {
        const mat = randomMatrix(&random);
        const vec = randomVec(&random);
        entry.simd_m = Mat3.fromArray(mat);
        entry.simd_v = Vec3.fromArray(vec);
        entry.scalar_m = ScalarMat3.init(mat);
        entry.scalar_v = ScalarVec3.init(vec[0], vec[1], vec[2]);
    }

    for (&quaternion_pairs) |*entry| {
        const axis_a = randomVec(&random);
        const angle_a = randomAngle(&random);
        const axis_b = randomVec(&random);
        const angle_b = randomAngle(&random);
        entry.simd_a = axisAngleToQuat(axis_a, angle_a);
        entry.scalar_a = axisAngleToScalarQuat(axis_a, angle_a);
        entry.simd_b = axisAngleToQuat(axis_b, angle_b);
        entry.scalar_b = axisAngleToScalarQuat(axis_b, angle_b);
    }

    for (&quaternion_vec_pairs) |*entry| {
        const axis = randomVec(&random);
        const angle = randomAngle(&random);
        const vec = randomVec(&random);
        entry.simd_q = axisAngleToQuat(axis, angle);
        entry.scalar_q = axisAngleToScalarQuat(axis, angle);
        entry.simd_v = Vec3.fromArray(vec);
        entry.scalar_v = ScalarVec3.init(vec[0], vec[1], vec[2]);
    }

    for (0..VectorPairBatchCount) |batch_idx| {
        const base = batch_idx * Batch;
        vector_pair_batches[batch_idx] = makeVec3PairBatch(base);
    }

    for (0..VectorEntryBatchCount) |batch_idx| {
        const base = batch_idx * Batch;
        vector_entry_batches[batch_idx] = makeVec3EntryBatch(base);
    }

    for (0..MatrixPairBatchCount) |batch_idx| {
        const base = batch_idx * Batch;
        matrix_pair_batches[batch_idx] = makeMat3PairBatch(base);
    }

    for (0..MatrixVecPairBatchCount) |batch_idx| {
        const base = batch_idx * Batch;
        matrix_vec_batches[batch_idx] = makeMatVecBatch(base);
    }

    for (0..QuaternionPairBatchCount) |batch_idx| {
        const base = batch_idx * Batch;
        quaternion_pair_batches[batch_idx] = makeQuatPairBatch(base);
    }

    for (0..QuaternionVecPairBatchCount) |batch_idx| {
        const base = batch_idx * Batch;
        quaternion_vec_batches[batch_idx] = makeQuatVecBatch(base);
    }
}

fn randomVec(random: *Random) [3]f32 {
    var out = [_]f32{ 0, 0, 0 };
    for (&out) |*component| {
        const value = random.float(f32) * 10.0 - 5.0;
        component.* = value;
    }
    const len_sq = out[0] * out[0] + out[1] * out[1] + out[2] * out[2];
    if (len_sq < 0.0625) {
        out[0] += 0.5;
    }
    return out;
}

fn randomMatrix(random: *Random) [9]f32 {
    var out: [9]f32 = undefined;
    for (0..out.len) |i| {
        out[i] = random.float(f32) * 6.0 - 3.0;
    }
    return out;
}

fn randomAngle(random: *Random) f32 {
    const two_pi = @as(f32, 2.0 * math.pi);
    return random.float(f32) * two_pi;
}

fn axisAngleToQuat(axis_vals: [3]f32, angle: f32) Quat {
    var axis = Vec3.fromArray(axis_vals);
    axis = Vec3.normalized(axis);
    return Quat.fromAxisAngle(axis, angle);
}

fn axisAngleToScalarQuat(axis_vals: [3]f32, angle: f32) ScalarQuat {
    var axis = ScalarVec3.init(axis_vals[0], axis_vals[1], axis_vals[2]);
    axis = ScalarVec3.normalize(axis);
    const half_angle = angle * 0.5;
    const sin_half = math.sin(half_angle);
    return ScalarQuat.init(math.cos(half_angle), axis.x * sin_half, axis.y * sin_half, axis.z * sin_half);
}

fn makeVec3BatchFromArray(items: [Batch]Vec3) Vec3Batch {
    var xs: [Batch]f32 = undefined;
    var ys: [Batch]f32 = undefined;
    var zs: [Batch]f32 = undefined;
    inline for (0..Batch) |lane| {
        const v = items[lane];
        xs[lane] = v.getUnchecked(0);
        ys[lane] = v.getUnchecked(1);
        zs[lane] = v.getUnchecked(2);
    }
    return .{
        .x = @bitCast(xs),
        .y = @bitCast(ys),
        .z = @bitCast(zs),
    };
}

fn makeMat3BatchFromArray(items: [Batch]Mat3) Mat3Batch {
    var columns: [9][Batch]f32 = undefined;
    inline for (0..Batch) |lane| {
        const mat = items[lane];
        columns[0][lane] = mat.get1D(0);
        columns[1][lane] = mat.get1D(1);
        columns[2][lane] = mat.get1D(2);
        columns[3][lane] = mat.get1D(3);
        columns[4][lane] = mat.get1D(4);
        columns[5][lane] = mat.get1D(5);
        columns[6][lane] = mat.get1D(6);
        columns[7][lane] = mat.get1D(7);
        columns[8][lane] = mat.get1D(8);
    }
    return .{
        .m00 = @bitCast(columns[0]),
        .m01 = @bitCast(columns[1]),
        .m02 = @bitCast(columns[2]),
        .m10 = @bitCast(columns[3]),
        .m11 = @bitCast(columns[4]),
        .m12 = @bitCast(columns[5]),
        .m20 = @bitCast(columns[6]),
        .m21 = @bitCast(columns[7]),
        .m22 = @bitCast(columns[8]),
    };
}

fn makeQuatBatchFromArray(items: [Batch]Quat) QuatBatch {
    var ws: [Batch]f32 = undefined;
    var xs: [Batch]f32 = undefined;
    var ys: [Batch]f32 = undefined;
    var zs: [Batch]f32 = undefined;
    inline for (0..Batch) |lane| {
        const q = items[lane];
        ws[lane] = q.w();
        xs[lane] = q.x();
        ys[lane] = q.y();
        zs[lane] = q.z();
    }
    return .{
        .w = @bitCast(ws),
        .x = @bitCast(xs),
        .y = @bitCast(ys),
        .z = @bitCast(zs),
    };
}

fn makeVec3PairBatch(start: usize) Vec3PairBatch {
    var a_array: [Batch]Vec3 = undefined;
    var b_array: [Batch]Vec3 = undefined;
    inline for (0..Batch) |lane| {
        const pair = vector_pairs[start + lane];
        a_array[lane] = pair.simd_a;
        b_array[lane] = pair.simd_b;
    }
    return .{
        .a = makeVec3BatchFromArray(a_array),
        .b = makeVec3BatchFromArray(b_array),
    };
}

fn makeVec3EntryBatch(start: usize) Vec3EntryBatch {
    var array: [Batch]Vec3 = undefined;
    inline for (0..Batch) |lane| {
        const entry = vector_entries[start + lane].simd;
        array[lane] = entry;
    }
    return .{ .value = makeVec3BatchFromArray(array) };
}

fn makeMat3PairBatch(start: usize) Mat3PairBatch {
    var a_array: [Batch]Mat3 = undefined;
    var b_array: [Batch]Mat3 = undefined;
    inline for (0..Batch) |lane| {
        const pair = matrix_pairs[start + lane];
        a_array[lane] = pair.simd_a;
        b_array[lane] = pair.simd_b;
    }
    return .{ .a = makeMat3BatchFromArray(a_array), .b = makeMat3BatchFromArray(b_array) };
}

fn makeMatVecBatch(start: usize) MatVecBatch {
    var m_array: [Batch]Mat3 = undefined;
    var v_array: [Batch]Vec3 = undefined;
    inline for (0..Batch) |lane| {
        const entry = matrix_vec_pairs[start + lane];
        m_array[lane] = entry.simd_m;
        v_array[lane] = entry.simd_v;
    }
    return .{ .m = makeMat3BatchFromArray(m_array), .v = makeVec3BatchFromArray(v_array) };
}

fn makeQuatPairBatch(start: usize) QuatPairBatch {
    var a_array: [Batch]Quat = undefined;
    var b_array: [Batch]Quat = undefined;
    inline for (0..Batch) |lane| {
        const pair = quaternion_pairs[start + lane];
        a_array[lane] = pair.simd_a;
        b_array[lane] = pair.simd_b;
    }
    return .{ .a = makeQuatBatchFromArray(a_array), .b = makeQuatBatchFromArray(b_array) };
}

fn makeQuatVecBatch(start: usize) QuatVecBatch {
    var q_array: [Batch]Quat = undefined;
    var v_array: [Batch]Vec3 = undefined;
    inline for (0..Batch) |lane| {
        const entry = quaternion_vec_pairs[start + lane];
        q_array[lane] = entry.simd_q;
        v_array[lane] = entry.simd_v;
    }
    return .{ .q = makeQuatBatchFromArray(q_array), .v = makeVec3BatchFromArray(v_array) };
}

inline fn vec3BatchAdd(a: Vec3Batch, b: Vec3Batch) Vec3Batch {
    return .{ .x = a.x + b.x, .y = a.y + b.y, .z = a.z + b.z };
}

inline fn vec3BatchSub(a: Vec3Batch, b: Vec3Batch) Vec3Batch {
    return .{ .x = a.x - b.x, .y = a.y - b.y, .z = a.z - b.z };
}

inline fn vec3BatchDot(a: Vec3Batch, b: Vec3Batch) @Vector(Batch, f32) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline fn vec3BatchCross(a: Vec3Batch, b: Vec3Batch) Vec3Batch {
    return .{
        .x = a.y * b.z - a.z * b.y,
        .y = a.z * b.x - a.x * b.z,
        .z = a.x * b.y - a.y * b.x,
    };
}

inline fn vec3BatchNormalize(v: Vec3Batch) Vec3Batch {
    const epsilon: @Vector(Batch, f32) = @splat(@as(f32, 1e-12));
    const zero: @Vector(Batch, f32) = @splat(@as(f32, 0.0));
    const one: @Vector(Batch, f32) = @splat(@as(f32, 1.0));
    const len_sq = vec3BatchDot(v, v);
    const safe_len_sq = @max(len_sq, epsilon);
    const inv_len = one / @sqrt(safe_len_sq);
    const mask = len_sq > epsilon;
    const selected = @select(f32, mask, inv_len, zero);
    return .{ .x = v.x * selected, .y = v.y * selected, .z = v.z * selected };
}

inline fn mat3BatchMul(a: Mat3Batch, b: Mat3Batch) Mat3Batch {
    return .{
        .m00 = a.m00 * b.m00 + a.m01 * b.m10 + a.m02 * b.m20,
        .m01 = a.m00 * b.m01 + a.m01 * b.m11 + a.m02 * b.m21,
        .m02 = a.m00 * b.m02 + a.m01 * b.m12 + a.m02 * b.m22,
        .m10 = a.m10 * b.m00 + a.m11 * b.m10 + a.m12 * b.m20,
        .m11 = a.m10 * b.m01 + a.m11 * b.m11 + a.m12 * b.m21,
        .m12 = a.m10 * b.m02 + a.m11 * b.m12 + a.m12 * b.m22,
        .m20 = a.m20 * b.m00 + a.m21 * b.m10 + a.m22 * b.m20,
        .m21 = a.m20 * b.m01 + a.m21 * b.m11 + a.m22 * b.m21,
        .m22 = a.m20 * b.m02 + a.m21 * b.m12 + a.m22 * b.m22,
    };
}

inline fn mat3BatchMulVec(m: Mat3Batch, v: Vec3Batch) Vec3Batch {
    return .{
        .x = m.m00 * v.x + m.m01 * v.y + m.m02 * v.z,
        .y = m.m10 * v.x + m.m11 * v.y + m.m12 * v.z,
        .z = m.m20 * v.x + m.m21 * v.y + m.m22 * v.z,
    };
}

inline fn quatBatchMul(a: QuatBatch, b: QuatBatch) QuatBatch {
    return .{
        .w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        .x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        .y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        .z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    };
}

inline fn quatBatchRotate(q: QuatBatch, v: Vec3Batch) Vec3Batch {
    const q_vec = Vec3Batch{ .x = q.x, .y = q.y, .z = q.z };
    const uv = vec3BatchCross(q_vec, v);
    const uuv = vec3BatchCross(q_vec, uv);
    const two: @Vector(Batch, f32) = @splat(@as(f32, 2.0));
    const w_scale = q.w * two;
    return vec3BatchAdd(vec3BatchAdd(v, .{
        .x = uv.x * w_scale,
        .y = uv.y * w_scale,
        .z = uv.z * w_scale,
    }), .{
        .x = uuv.x * two,
        .y = uuv.y * two,
        .z = uuv.z * two,
    });
}

inline fn quatBatchNormalize(q: QuatBatch) QuatBatch {
    const epsilon: @Vector(Batch, f32) = @splat(@as(f32, 1e-12));
    const zero: @Vector(Batch, f32) = @splat(@as(f32, 0.0));
    const one: @Vector(Batch, f32) = @splat(@as(f32, 1.0));
    const len_sq = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    const safe_len_sq = @max(len_sq, epsilon);
    const inv_len = one / @sqrt(safe_len_sq);
    const mask = len_sq > epsilon;
    const selected = @select(f32, mask, inv_len, zero);
    return .{ .w = q.w * selected, .x = q.x * selected, .y = q.y * selected, .z = q.z * selected };
}

inline fn reduceBits(v: @Vector(Batch, f32)) u32 {
    const bits: @Vector(Batch, u32) = @bitCast(v);
    return @reduce(.Add, bits);
}

inline fn reduceVec3Components(v: Vec3Batch, lane_index: usize) u32 {
    const component = switch (lane_index % 3) {
        0 => v.x,
        1 => v.y,
        else => v.z,
    };
    return reduceBits(component);
}

fn runBenchmarkCase(case: BenchmarkCase) !void {
    const warmup_iters = computeWarmupIterations(case);

    var simd_samples: [SampleCount]u64 = undefined;
    var scalar_samples: [SampleCount]u64 = undefined;
    var simd_sorted: [SampleCount]u64 = undefined;
    var scalar_sorted: [SampleCount]u64 = undefined;

    var expected_simd_checksum: ?u32 = null;
    var expected_scalar_checksum: ?u32 = null;

    for (0..SampleCount) |i| {
        const simd_sample = try measure(case.simd_work, warmup_iters, case.iterations);
        const scalar_sample = try measure(case.scalar_work, warmup_iters, case.iterations);

        if (expected_simd_checksum) |expected| {
            if (expected != simd_sample.checksum) {
                std.debug.panic("SIMD checksum mismatch for {s}: expected {d}, got {d}", .{ case.name, expected, simd_sample.checksum });
            }
        } else {
            expected_simd_checksum = simd_sample.checksum;
        }

        if (expected_scalar_checksum) |expected| {
            if (expected != scalar_sample.checksum) {
                std.debug.panic("Scalar checksum mismatch for {s}: expected {d}, got {d}", .{ case.name, expected, scalar_sample.checksum });
            }
        } else {
            expected_scalar_checksum = scalar_sample.checksum;
        }

        simd_samples[i] = simd_sample.ns;
        scalar_samples[i] = scalar_sample.ns;
    }

    const simd_stats = computeStats(SampleCount, &simd_samples, &simd_sorted);
    const scalar_stats = computeStats(SampleCount, &scalar_samples, &scalar_sorted);

    const median_speedup = @as(f64, @floatFromInt(scalar_stats.median_ns)) / @as(f64, @floatFromInt(simd_stats.median_ns));

    std.debug.print(
        "{s:<20} {d:7.3} ±{d:5.3} ms   {d:7.3} ±{d:5.3} ms   {d:6.2}x   iters: {d}\n",
        .{
            case.name,
            nsToMsInt(simd_stats.median_ns),
            nsToMsFloat(simd_stats.stddev_ns),
            nsToMsInt(scalar_stats.median_ns),
            nsToMsFloat(scalar_stats.stddev_ns),
            median_speedup,
            case.iterations,
        },
    );
}

fn measure(work: WorkFn, warmup_iters: usize, iterations: usize) !BenchmarkSample {
    if (warmup_iters > 0) {
        _ = work(warmup_iters);
    }

    var timer = try std.time.Timer.start();
    const checksum = work(iterations);
    const elapsed = timer.read();
    return .{ .ns = elapsed, .checksum = checksum };
}

fn computeWarmupIterations(case: BenchmarkCase) usize {
    if (case.warmup_override) |override| {
        return override;
    }
    const provisional = case.iterations / WarmupDivisor;
    return if (provisional == 0) 1 else provisional;
}

fn computeStats(comptime N: usize, samples: *const [N]u64, sorted_out: *[N]u64) BenchmarkStats {
    sorted_out.* = samples.*;
    sort.insertion(u64, sorted_out.*[0..], {}, sort.asc(u64));

    var total: f64 = 0;
    for (samples.*) |value| {
        total += @as(f64, @floatFromInt(value));
    }
    const mean_ns = total / @as(f64, @floatFromInt(N));

    var variance_sum: f64 = 0;
    for (samples.*) |value| {
        const diff = @as(f64, @floatFromInt(value)) - mean_ns;
        variance_sum += diff * diff;
    }
    const denominator = if (N > 1) @as(f64, @floatFromInt(N - 1)) else 1.0;
    const stddev_ns = math.sqrt(variance_sum / denominator);

    return .{
        .min_ns = sorted_out.*[0],
        .max_ns = sorted_out.*[N - 1],
        .median_ns = sorted_out.*[N / 2],
        .mean_ns = mean_ns,
        .stddev_ns = stddev_ns,
    };
}

inline fn nsToMsInt(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / NanosecondsPerMillisecond;
}

inline fn nsToMsFloat(ns: f64) f64 {
    return ns / NanosecondsPerMillisecond;
}

inline fn nextIndex(idx: usize, len: usize) usize {
    const next = idx + 1;
    return if (next == len) 0 else next;
}

fn vectorAddSimd(iterations: usize) u32 {
    std.debug.assert(iterations % Batch == 0);
    const passes = iterations / Batch;
    var acc = vector_pair_batches[0].a;
    var checksum: u32 = 0;
    var idx: usize = 0;
    var lane: usize = 0;
    for (0..passes) |_| {
        const entry = vector_pair_batches[idx];
        acc = vec3BatchAdd(acc, entry.b);
        checksum +%= reduceVec3Components(acc, lane);
        idx = nextIndex(idx, vector_pair_batches.len);
        lane = nextLane(lane, 3);
    }
    std.mem.doNotOptimizeAway(&acc);
    return checksum;
}

fn vectorAddScalar(iterations: usize) u32 {
    var acc = vector_pairs[0].scalar_a;
    var checksum: u32 = 0;
    var idx: usize = 0;
    var lane: usize = 0;
    for (0..iterations) |_| {
        const entry = vector_pairs[idx];
        acc = ScalarVec3.add(acc, entry.scalar_b);
        checksum +%= @as(u32, @bitCast(selectScalarComponent(acc, lane)));
        idx = nextIndex(idx, vector_pairs.len);
        lane = nextLane(lane, 3);
    }
    std.mem.doNotOptimizeAway(&acc);
    return checksum;
}

fn vectorDotSimd(iterations: usize) u32 {
    std.debug.assert(iterations % Batch == 0);
    const passes = iterations / Batch;
    var result_vec: @Vector(Batch, f32) = @splat(@as(f32, 0.0));
    var idx: usize = 0;
    for (0..passes) |_| {
        const entry = vector_pair_batches[idx];
        result_vec += vec3BatchDot(entry.a, entry.b);
        idx = nextIndex(idx, vector_pair_batches.len);
    }
    std.mem.doNotOptimizeAway(&result_vec);
    return reduceBits(result_vec);
}

fn vectorDotScalar(iterations: usize) u32 {
    var result: f32 = 0;
    var idx: usize = 0;
    for (0..iterations) |_| {
        const entry = vector_pairs[idx];
        result += ScalarVec3.dot(entry.scalar_a, entry.scalar_b);
        idx = nextIndex(idx, vector_pairs.len);
    }
    std.mem.doNotOptimizeAway(&result);
    return @as(u32, @bitCast(result));
}

fn vectorCrossSimd(iterations: usize) u32 {
    std.debug.assert(iterations % Batch == 0);
    const passes = iterations / Batch;
    var current = vector_pair_batches[0].a;
    var checksum: u32 = 0;
    var idx: usize = 0;
    var lane: usize = 0;
    for (0..passes) |_| {
        const entry = vector_pair_batches[idx];
        current = vec3BatchAdd(vec3BatchCross(current, entry.b), entry.a);
        checksum ^= reduceVec3Components(current, lane);
        idx = nextIndex(idx, vector_pair_batches.len);
        lane = nextLane(lane, 3);
    }
    std.mem.doNotOptimizeAway(&current);
    return checksum;
}

fn vectorCrossScalar(iterations: usize) u32 {
    var current = vector_pairs[0].scalar_a;
    var checksum: u32 = 0;
    var idx: usize = 0;
    var lane: usize = 0;
    for (0..iterations) |_| {
        const entry = vector_pairs[idx];
        current = ScalarVec3.cross(current, entry.scalar_b);
        checksum ^= @as(u32, @bitCast(selectScalarComponent(current, lane)));
        current = ScalarVec3.add(current, entry.scalar_a);
        idx = nextIndex(idx, vector_pairs.len);
        lane = nextLane(lane, 3);
    }
    std.mem.doNotOptimizeAway(&current);
    return checksum;
}

fn vectorNormalizeSimd(iterations: usize) u32 {
    std.debug.assert(iterations % Batch == 0);
    const passes = iterations / Batch;
    var checksum: u32 = 0;
    var idx: usize = 0;
    var lane: usize = 0;
    for (0..passes) |_| {
        const batch = vector_entry_batches[idx].value;
        const normalized = vec3BatchNormalize(batch);
        checksum +%= reduceVec3Components(normalized, lane);
        idx = nextIndex(idx, vector_entry_batches.len);
        lane = nextLane(lane, 3);
    }
    return checksum;
}

fn vectorNormalizeScalar(iterations: usize) u32 {
    var checksum: u32 = 0;
    var idx: usize = 0;
    var lane: usize = 0;
    for (0..iterations) |_| {
        var value = vector_entries[idx].scalar;
        value = ScalarVec3.normalize(value);
        checksum +%= @as(u32, @bitCast(selectScalarComponent(value, lane)));
        idx = nextIndex(idx, vector_entries.len);
        lane = nextLane(lane, 3);
    }
    return checksum;
}

fn matrixMulSimd(iterations: usize) u32 {
    std.debug.assert(iterations % Batch == 0);
    const passes = iterations / Batch;
    var acc = matrix_pair_batches[0].a;
    var checksum: u32 = 0;
    var idx: usize = 0;
    var lane: usize = 0;
    for (0..passes) |_| {
        const entry = matrix_pair_batches[idx];
        acc = mat3BatchMul(acc, entry.b);
        const components = switch (lane % 9) {
            0 => acc.m00,
            1 => acc.m01,
            2 => acc.m02,
            3 => acc.m10,
            4 => acc.m11,
            5 => acc.m12,
            6 => acc.m20,
            7 => acc.m21,
            else => acc.m22,
        };
        checksum ^= reduceBits(components);
        idx = nextIndex(idx, matrix_pair_batches.len);
        lane = nextLane(lane, 9);
    }
    std.mem.doNotOptimizeAway(&acc);
    return checksum;
}

fn matrixMulScalar(iterations: usize) u32 {
    var acc = matrix_pairs[0].scalar_a;
    var checksum: u32 = 0;
    var idx: usize = 0;
    var lane: usize = 0;
    for (0..iterations) |_| {
        const entry = matrix_pairs[idx];
        acc = ScalarMat3.mulMat(acc, entry.scalar_b);
        checksum ^= @as(u32, @bitCast(acc.data[lane]));
        idx = nextIndex(idx, matrix_pairs.len);
        lane = nextLane(lane, 9);
    }
    std.mem.doNotOptimizeAway(&acc);
    return checksum;
}

fn matrixVecSimd(iterations: usize) u32 {
    std.debug.assert(iterations % Batch == 0);
    const passes = iterations / Batch;
    var vec = matrix_vec_batches[0].v;
    var checksum: u32 = 0;
    var idx: usize = 0;
    var lane: usize = 0;
    for (0..passes) |_| {
        const entry = matrix_vec_batches[idx];
        vec = mat3BatchMulVec(entry.m, vec);
        checksum +%= reduceVec3Components(vec, lane);
        idx = nextIndex(idx, matrix_vec_batches.len);
        lane = nextLane(lane, 3);
    }
    std.mem.doNotOptimizeAway(&vec);
    return checksum;
}

fn matrixVecScalar(iterations: usize) u32 {
    var vec = matrix_vec_pairs[0].scalar_v;
    var checksum: u32 = 0;
    var idx: usize = 0;
    var lane: usize = 0;
    for (0..iterations) |_| {
        const entry = matrix_vec_pairs[idx];
        vec = ScalarMat3.mulVec(entry.scalar_m, vec);
        checksum +%= @as(u32, @bitCast(selectScalarComponent(vec, lane)));
        idx = nextIndex(idx, matrix_vec_pairs.len);
        lane = nextLane(lane, 3);
    }
    std.mem.doNotOptimizeAway(&vec);
    return checksum;
}

fn quaternionMulSimd(iterations: usize) u32 {
    std.debug.assert(iterations % Batch == 0);
    const passes = iterations / Batch;
    var quat = quaternion_pair_batches[0].a;
    var checksum: u32 = 0;
    var idx: usize = 0;
    for (0..passes) |_| {
        const entry = quaternion_pair_batches[idx];
        quat = quatBatchMul(quat, entry.b);
        checksum ^= reduceBits(quat.w);
        idx = nextIndex(idx, quaternion_pair_batches.len);
    }
    std.mem.doNotOptimizeAway(&quat);
    return checksum;
}

fn quaternionMulScalar(iterations: usize) u32 {
    var quat = quaternion_pairs[0].scalar_a;
    var checksum: u32 = 0;
    var idx: usize = 0;
    for (0..iterations) |_| {
        const entry = quaternion_pairs[idx];
        quat = ScalarQuat.mul(quat, entry.scalar_b);
        checksum ^= @as(u32, @bitCast(quat.w));
        idx = nextIndex(idx, quaternion_pairs.len);
    }
    std.mem.doNotOptimizeAway(&quat);
    return checksum;
}

fn quaternionRotateSimd(iterations: usize) u32 {
    std.debug.assert(iterations % Batch == 0);
    const passes = iterations / Batch;
    var vec = quaternion_vec_batches[0].v;
    var checksum: u32 = 0;
    var idx: usize = 0;
    var lane: usize = 0;
    for (0..passes) |_| {
        const entry = quaternion_vec_batches[idx];
        const normalized_q = quatBatchNormalize(entry.q);
        vec = quatBatchRotate(normalized_q, vec);
        checksum ^= reduceVec3Components(vec, lane);
        idx = nextIndex(idx, quaternion_vec_batches.len);
        lane = nextLane(lane, 3);
    }
    std.mem.doNotOptimizeAway(&vec);
    return checksum;
}

fn quaternionRotateScalar(iterations: usize) u32 {
    var vec = quaternion_vec_pairs[0].scalar_v;
    var checksum: u32 = 0;
    var idx: usize = 0;
    var lane: usize = 0;
    for (0..iterations) |_| {
        const entry = quaternion_vec_pairs[idx];
        vec = ScalarQuat.rotate(entry.scalar_q, vec);
        checksum ^= @as(u32, @bitCast(selectScalarComponent(vec, lane)));
        idx = nextIndex(idx, quaternion_vec_pairs.len);
        lane = nextLane(lane, 3);
    }
    std.mem.doNotOptimizeAway(&vec);
    return checksum;
}

inline fn selectScalarComponent(v: ScalarVec3, lane: usize) f32 {
    return switch (lane) {
        0 => v.x,
        1 => v.y,
        else => v.z,
    };
}

inline fn nextLane(current: usize, count: usize) usize {
    const next = current + 1;
    return if (next == count) 0 else next;
}

pub fn main() !void {
    initDatasets();

    const cases = [_]BenchmarkCase{
        .{ .name = "Vec3 Add", .iterations = 4_000_000, .simd_work = vectorAddSimd, .scalar_work = vectorAddScalar },
        .{ .name = "Vec3 Dot", .iterations = 20_000_000, .simd_work = vectorDotSimd, .scalar_work = vectorDotScalar },
        .{ .name = "Vec3 Cross", .iterations = 8_000_000, .simd_work = vectorCrossSimd, .scalar_work = vectorCrossScalar },
        .{ .name = "Vec3 Normalize", .iterations = 6_000_000, .simd_work = vectorNormalizeSimd, .scalar_work = vectorNormalizeScalar },
        .{ .name = "Mat3 × Mat3", .iterations = 1_000_000, .simd_work = matrixMulSimd, .scalar_work = matrixMulScalar },
        .{ .name = "Mat3 × Vec3", .iterations = 4_000_000, .simd_work = matrixVecSimd, .scalar_work = matrixVecScalar },
        .{ .name = "Quat Mul", .iterations = 4_000_000, .simd_work = quaternionMulSimd, .scalar_work = quaternionMulScalar },
        .{ .name = "Quat Rotate", .iterations = 10_000_000, .simd_work = quaternionRotateSimd, .scalar_work = quaternionRotateScalar },
    };

    std.debug.print("\n=== Typhoon Math SIMD vs Scalar Benchmarks ===\n", .{});
    std.debug.print("Samples per case: {d}, warmup divisor: {d}\n\n", .{ SampleCount, WarmupDivisor });
    std.debug.print("{s:<20} {s:>16}   {s:>18}   {s:>8}   {s:>8}\n", .{
        "Benchmark",
        "SIMD median",
        "Scalar median",
        "Speedup",
        "iters",
    });
    std.debug.print("{s:-<20} {s:-<16}   {s:-<18}   {s:-<8}   {s:-<8}\n", .{ "", "", "", "", "" });

    for (cases) |case| {
        try runBenchmarkCase(case);
    }

    std.debug.print("\n=== Benchmark Complete ===\n", .{});
}
