//! h0279's build: gate.zig against the worktree's real idx_pick_wide.zig / idx_fused.zig / fused_trace.zig (and the
//! closure fused_trace pulls). The worktree root comes from -Dwt (build.sh passes it); gate.zig is local. The cuda
//! runtime module is built host-only exactly as zig/build/cuda.zig's runtime() does (no fatbins).
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const wt = b.option([]const u8, "wt", "the worktree root") orelse "";
    const dsv41 = b.pathJoin(&.{ wt, "zig", "src", "families", "dsv41" });

    const options = b.addOptions();
    options.addOption(bool, "with_kernels", false);
    options.addOption(bool, "has_dsv41_port", false);
    const cuda = b.createModule(.{ .root_source_file = .{ .cwd_relative = b.pathJoin(&.{ wt, "zig", "src", "cuda", "root.zig" }) }, .target = target, .optimize = optimize, .link_libc = true });
    cuda.addOptions("kernel_options", options);
    cuda.addImport("stagger", b.createModule(.{ .root_source_file = .{ .cwd_relative = b.pathJoin(&.{ wt, "zig", "src", "core", "stagger.zig" }) }, .target = target, .optimize = optimize }));

    // one module owns the whole worktree closure (a file may belong to only one module): fused_trace's root
    // pulls idx_pick_wide/idx_fused and their helpers as relatives
    const ft = b.createModule(.{ .root_source_file = .{ .cwd_relative = b.pathJoin(&.{ dsv41, "fused_trace.zig" }) }, .target = target, .optimize = optimize, .link_libc = true });
    ft.addImport("cuda", cuda);

    const gate = b.createModule(.{ .root_source_file = b.path("gate.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{
        .{ .name = "fused_trace", .module = ft },
    } });
    const exe = b.addExecutable(.{ .name = "h0279-gate", .root_module = gate });
    b.installArtifact(exe);
    const run = b.addRunArtifact(exe);
    run.step.dependOn(b.getInstallStep());
    b.step("run", "run the h0279 host gate against the worktree modules").dependOn(&run.step);
}
