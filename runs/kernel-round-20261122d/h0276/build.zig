//! h0276's build: the harness against the worktree's core/prompt_cache.zig. The worktree path comes from -Dpc
//! (build.sh passes it); the harness and the model are local.
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const pc_path = b.option([]const u8, "pc", "the worktree's zig/src/core/prompt_cache.zig") orelse "";
    const pc = b.createModule(.{ .root_source_file = .{ .cwd_relative = pc_path }, .target = target, .optimize = optimize, .link_libc = true });
    const harness = b.createModule(.{ .root_source_file = b.path("harness.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "prompt_cache", .module = pc }} });
    const exe = b.addExecutable(.{ .name = "h0276", .root_module = harness });
    b.installArtifact(exe);
    const run = b.addRunArtifact(exe);
    run.step.dependOn(b.getInstallStep());
    b.step("run", "run the h0276 keep-side harness").dependOn(&run.step);
}
