# Typhoon Math

A high-performance, SIMD-optimized mathematical library for 3D graphics, physics simulation, and computational geometry. Written in Zig and designed for both f32 and f64 precision.

## Features

- **SIMD-Optimized Operations**: All vector and matrix operations use native SIMD instructions (`@Vector`, `@reduce`, `@splat`, `@shuffle`)
- **Generic Precision**: Supports both f32 and f64 floating-point types
- **Zero Dependencies**: Pure Zig implementation with no external dependencies
- **Comprehensive**: Vectors, matrices, quaternions, geometry primitives, and collision support functions
- **Type-Safe**: Compile-time generic programming ensures type safety and zero-cost abstractions
- **Well-Tested**: Extensive test coverage including edge cases and numerical stability
- **Dedicated Test Suites**: Geometry, SIMD batch helpers, and support mappings each have their own unit suites so regressions surface immediately

## Modules

### Vector (`vector.zig`)
Generic N-dimensional vectors with SIMD operations:
```zig
const Vec3 = Vector(3, f64);

var a = Vec3.fromArray(.{ 1.0, 2.0, 3.0 });
const b = Vec3.fromArray(.{ 4.0, 5.0, 6.0 });

a.add(b);                    // Mutating addition
const dot = a.dot(b);        // Dot product
const cross = a.cross(b);    // Cross product (3D only)
a.normalize();               // Normalize in-place
const len = a.magnitude();   // Vector length
```

**Key Operations:**
- Construction: `fromArray`, `fromSlice`, `zero`, `splat`
- Arithmetic: `add`, `sub`, `mulScalar`, `divScalar`, `negate`
- Products: `dot`, `cross` (3D)
- Magnitude: `magnitude`, `magnitudeSq`, `normalize`
- Comparison: `equals`, `approxEquals`
- Non-mutating variants: `added`, `subbed`, `scaled`, `normalized`

### Matrix (`matrix.zig`)
Generic NxM matrices with specialized 3x3 and 3x4 transform support:
```zig
const Mat3 = Matrix(3, 3, f64);
const Mat4 = Matrix(3, 4, f64);

const identity = Mat3.identity();
var m = Mat3.fromRows(.{
    .{ 1.0, 0.0, 0.0 },
    .{ 0.0, 2.0, 0.0 },
    .{ 0.0, 0.0, 3.0 },
});

const inv = m.inverse3();           // Invert 3x3 matrix
const det = m.determinant3();       // Determinant
const transposed = m.transpose();   // Transpose

// 3x4 transform matrices (rotation + translation)
const pose = Mat4.identity();
const point = pose.transform(vec);        // Transform point
const dir = pose.transformDirection(vec); // Transform direction (no translation)
```

**Key Operations:**
- Construction: `identity`, `zero`, `fromRows`, `fromColumns`
- Arithmetic: `add`, `sub`, `mulScalar`
- Multiplication: `mul` (matrix-matrix), `mulVec` (matrix-vector)
- Transform: `transpose`, `inverse3`, `determinant3`
- 3D Transforms: `transform`, `transformDirection`, `transformInverse`

### Quaternion (`quaternion.zig`)
Unit quaternions for 3D rotations:
```zig
const Quat = Quaternion(f64);

const identity = Quat.identity();
const q = Quat.fromAxisAngle(axis, angle);

var rotation = Quat.fromEuler(roll, pitch, yaw);
rotation.normalize();

const result = q1.mul(q2);          // Concatenate rotations
const rotated = q.rotateVector(v);  // Rotate vector
const inv = q.inverse();            // Inverse rotation
const conj = q.conjugate();         // Conjugate (for unit quats, same as inverse)
```

**Key Operations:**
- Construction: `identity`, `fromAxisAngle`, `fromEuler`
- Arithmetic: `mul` (quaternion multiplication)
- Rotation: `rotateVector`, `rotateVectorAssign`
- Transform: `conjugate`, `inverse`, `normalize`
- Interpolation: `addScaledVector` (for integration)

### Geometry (`geometry.zig`)
Pure geometry primitives for collision detection and spatial queries:
```zig
const Sphere = geometry.Sphere(f64);
const Box = geometry.Box(f64);
const Capsule = geometry.Capsule(f64);
const Shape = geometry.Shape(f64);
const Pose = geometry.Pose(f64);

const sphere = Shape{ .sphere = .{ .radius = 1.0 } };
const box = Shape{ .box = .{ .half_extents = Vec3.fromArray(.{ 0.5, 1.0, 0.5 }) } };

const pose = Pose.from(position, rotation);
const world_point = pose.localToWorldPoint(local_point);
const local_point = pose.worldToLocalPoint(world_point);
```

**Supported Shapes:**
- `Sphere`: Radius-based
- `Box`: Axis-aligned or oriented, defined by half-extents
- `Capsule`: Cylinder with hemispherical caps
- `Cylinder`: Radius and half-height
- `Hull`: Convex hull from vertex list
- `Scaled`: Wrapped shape with non-uniform scaling

**Pose Transform:**
- Position + rotation (quaternion)
- Cached 3x4 transform matrices (forward and inverse)
- Point and direction transformations

### Support (`support.zig`)
Support mapping functions for collision detection (GJK/EPA):
```zig
const support_point = support.supportPoint(shape, pose, direction);
const minkowski_diff = support.supportDifference(shapeA, poseA, shapeB, poseB, direction);
```

Support functions return the farthest point on a shape in a given direction, fundamental for convex collision detection algorithms.

### Utilities (`utils.zig`)
Scalar utility functions:
```zig
const clamped = utils.clamp(f64, value, min, max);
const saturated = utils.saturate(f64, value);        // Clamp to [0, 1]
const interpolated = utils.lerp(f64, a, b, t);
const smooth = utils.smoothstep(f64, t);
const magnitude = utils.sign(f64, x);
const squared = utils.square(f64, x);
const inv = utils.invSqrt(f64, x);
```

## Installation

### As a Zig Module

Add to your `build.zig.zon`:
```zig
.dependencies = .{
    .typhoon_math = .{
        .path = "libs/typhoon-math",
    },
},
```

In your `build.zig`:
```zig
const typhoon_math = b.dependency("typhoon_math", .{
    .target = target,
    .optimize = optimize,
});

exe.root_module.addImport("typhoon_math", typhoon_math.module("typhoon_math"));
```

### Usage in Code
```zig
const tm = @import("typhoon_math");

const Vec3 = tm.Vector(3, f64);
const Quat = tm.Quaternion(f64);
const Mat3 = tm.Matrix(3, 3, f64);
```

## Building

```bash
# Run tests
zig build test

# Run the example executable
zig build run

# Build for specific target
zig build -Dtarget=wasm32-freestanding -Doptimize=ReleaseFast
```

### Testing & Benchmarks

- **Unit tests** – Run `zig test src/root.zig` (or `zig build test` from the repo root) to execute every math suite. All test cases live under `src/test/` (split into vector/matrix, geometry, batch, and support files) so additions stay organized by domain.
- **Microbenchmarks** – `zig build benchmark` compares the SIMD batch helpers against scalar reference implementations on the current machine. Treat these numbers as kernel-level signals: they run single-threaded on cache-friendly workloads and don’t account for batching/marshaling overhead in a full physics pipeline.

## Design Philosophy

### SIMD-First
All vector and matrix operations are built on Zig's `@Vector` type, ensuring optimal SIMD code generation on all platforms. No scalar fallbacks.

### Generic Precision
The library is generic over floating-point types (f32/f64), allowing users to choose precision based on their needs. Physics simulations typically use f64 for stability, while rendering uses f32.

### Minimal Comments
Code is kept clean with minimal comments. Only non-obvious algorithms or constraints are documented. The API is self-explanatory through clear naming.

### Zero-Cost Abstractions
All generic functions are resolved at compile-time. Inline functions ensure no overhead. The library compiles to optimal machine code equivalent to hand-written SIMD intrinsics.

## Performance

- Vector operations compile to single SIMD instructions
- Matrix multiplication uses SIMD shuffles and reduces
- Quaternion rotations optimized for minimal operations
- All hot paths use `inline` for guaranteed inlining
- No allocations in core operations (stack-only)

## Compatibility

- **Zig Version**: 0.15.0 or later
- **Platforms**: Any platform with SIMD support (x86_64, ARM64, WASM with simd128)
- **Precision**: f32, f64 (generic over float types)
- **Freestanding**: Compatible with freestanding targets (WASM, embedded)

## Testing

Run the comprehensive test suite:
```bash
zig build test
```

Tests cover:
- Basic operations (construction, arithmetic)
- Edge cases (zero vectors, singular matrices, degenerate quaternions)
- Numerical stability
- SIMD verification
- 3D transform chains
- Rotation composition
- Inverse/determinant accuracy

## License

See LICENSE file for details.

## Contributing

This library is part of the Typhoon physics engine. Contributions should:
- Maintain SIMD-first approach (no scalar loops)
- Follow existing naming conventions (camelCase for functions, PascalCase for types)
- Include tests for new functionality
- Keep comments minimal (only document non-obvious behavior)
- Preserve zero-cost abstraction guarantees

## Related Projects

- **Typhoon**: Rigid body physics engine using this math library
- Used for game physics, robotics simulation, and real-time collision detection
