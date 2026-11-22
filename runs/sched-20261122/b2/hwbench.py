"""Round B2: the same GPU microbench on each Spark with the engine stopped. Is rank 0's box slower on identical work?
Decode at small rows is weight-read bound, so time: (1) device-to-device copy bandwidth (large), (2) bf16 GEMV-shaped
matmuls at decode shapes (rows 1/7/28 x 5120 against 5120x2304 and 2304x5120 weights, many distinct weight tensors so
nothing stays in L2), (3) a launch-latency loop of tiny kernels. CUDA events, median of repeats. Prints one JSON line."""
import json, socket, statistics as st, torch
torch.backends.cuda.matmul.allow_tf32 = False
dev = torch.device("cuda")
def timeit(fn, reps=30, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return st.median(ts), min(ts), max(ts)
out = {"host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0)}
n = 1 << 30
x = torch.empty(n, dtype=torch.uint8, device=dev); y = torch.empty_like(x)
m, lo, hi = timeit(lambda: y.copy_(x))
out["d2d_copy_GBps"] = round(2 * n / (m / 1e3) / 1e9, 1); out["d2d_copy_spread_ms"] = [round(lo, 3), round(hi, 3)]
del x, y
W = [torch.randn(5120, 2304, dtype=torch.bfloat16, device=dev) for _ in range(48)]    # 48 x 23.6 MB, beyond L2
W2 = [torch.randn(2304, 5120, dtype=torch.bfloat16, device=dev) for _ in range(48)]
for rows in (1, 7, 28):
    a = torch.randn(rows, 5120, dtype=torch.bfloat16, device=dev); b = torch.randn(rows, 2304, dtype=torch.bfloat16, device=dev)
    def chain():
        for w, w2 in zip(W, W2):
            torch.mm(a, w); torch.mm(b, w2)
    m, lo, hi = timeit(chain, reps=20)
    out[f"gemv_rows{rows}_us_per_pair"] = round(1000 * m / len(W), 2)
    out[f"gemv_rows{rows}_eff_GBps"] = round(2 * 5120 * 2304 * 2 * len(W) / (m / 1e3) / 1e9, 1)
z = torch.zeros(1, device=dev)
def tiny():
    for _ in range(1000): z.add_(1)
m, lo, hi = timeit(tiny, reps=10)
out["tiny_kernel_us"] = round(m, 3)
print(json.dumps(out))
