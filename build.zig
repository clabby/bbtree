const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const module = b.addModule("bbtree", .{
        .root_source_file = b.path("src/bbtree.zig"),
        .target = target,
        .optimize = optimize,
    });
    build_tests(b, target, optimize);
    build_bench(b, target, optimize, module);
}

fn build_tests(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) void {
    // Scales the iterations of the randomized and concurrency tests, for long runs.
    const stress_help = "Multiply randomized test iterations (default 1)";
    const stress = b.option(u32, "stress", stress_help) orelse 1;
    const test_options = b.addOptions();
    test_options.addOption(u32, "stress", stress);
    const test_module = b.createModule(.{
        .root_source_file = b.path("src/bbtree.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "test_options", .module = test_options.createModule() }},
    });
    const tests = b.addTest(.{ .root_module = test_module });
    const tests_run = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run the unit, differential and concurrency tests");
    test_step.dependOn(&tests_run.step);
}

fn build_bench(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    module: *std.Build.Module,
) void {
    const bench = b.addExecutable(.{
        .name = "bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/bench.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "bbtree", .module = module }},
        }),
    });
    // Installed with everything else, so that `zig build` compiles it on every target.
    b.installArtifact(bench);
    const bench_run = b.addRunArtifact(bench);
    bench_run.addPassthruArgs();
    const bench_step = b.step("bench", "Run the microbenchmarks (use -Doptimize=ReleaseFast)");
    bench_step.dependOn(&bench_run.step);
}
