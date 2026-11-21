#!/usr/bin/env python3
"""Occupancy of the fused 0197 kernels at the served launch shapes (driver API), plus the analytic limits that bind."""
import ctypes as ct, json, sys
cuda = ct.CDLL("libcuda.so.1")
I, P = ct.c_int, ct.c_void_p
def cu(n,*a):
    e=getattr(cuda,n)(*a)
    if e: raise RuntimeError(f"{n}: {e}")
cu("cuInit",0); ctx=P(); cu("cuDevicePrimaryCtxRetain",ct.byref(ctx),0); cu("cuCtxSetCurrent",ctx)
def fn(cubin,name):
    m=P(); cu("cuModuleLoad",ct.byref(m),str(cubin).encode())
    f=P(); cu("cuModuleGetFunction",ct.byref(f),m,name.encode()); return f
def occ(f,threads):
    n=I(); cu("cuOccupancyMaxActiveBlocksPerMultiprocessor",ct.byref(n),f,I(threads),ct.c_size_t(0)); return n.value
def attr(f,a):
    v=I(); cu("cuFuncGetAttribute",ct.byref(v),I(a),f); return v.value
# CU_FUNC_ATTRIBUTE: 0 MAX_THREADS_PER_BLOCK, 2 NUM_REGS, 3 SHARED_SIZE_BYTES, 4 LOCAL_SIZE_BYTES, 8 MAX_DYNAMIC_SHARED_SIZE
bin_=sys.argv[1]; out=sys.argv[2]
rows=[]
def add(label,cubin,name,threads):
    f=fn(cubin,name)
    rows.append({"kernel":label,"threads":threads,"max_threads":attr(f,0),"regs":attr(f,4),
                 "shared":attr(f,1),"local":attr(f,3),"active_blocks":occ(f,threads)})
    print(rows[-1], flush=True)
b=sys.argv[1]
add("compress2_norm d=512 R=128 (32,8)",f"{b}/new-torch.cubin","tf_ds_compress2_norm_kernel",256)
add("compress2_norm d=512 R=1024 (32,4)",f"{b}/new-torch.cubin","tf_ds_compress2_norm_kernel",128)
add("compress2_norm d=2048 R=128 (32,8)",f"{b}/new-torch.cubin","tf_ds_compress2_norm_kernel",256)
add("compress2_norm d=2048 R=1024 (32,4)",f"{b}/new-torch.cubin","tf_ds_compress2_norm_kernel",128)
add("compress2x_norm128 R=1..48",f"{b}/new-torch.cubin","tf_ds_compress2x_norm128_kernel",128)
add("compress2x_norm512 R=1..48",f"{b}/new-torch.cubin","tf_ds_compress2x_norm512_kernel",512)
add("old compress2x_norm (both widths) 128",f"{b}/old-torch.cubin","tf_ds_compress2x_norm_kernel",128)
add("old compress2x_norm (both widths) 512",f"{b}/old-torch.cubin","tf_ds_compress2x_norm_kernel",512)
open(out,"w").write(json.dumps(rows,indent=1))
