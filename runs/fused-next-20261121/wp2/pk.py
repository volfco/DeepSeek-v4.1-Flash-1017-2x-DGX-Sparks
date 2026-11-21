import sys, ctypes as ct, numpy as np
from pathlib import Path
cuda = ct.CDLL("libcuda.so.1"); P=ct.c_void_p; I=ct.c_int; L=ct.c_int64; F=ct.c_float
def cu(n,*a):
    e=getattr(cuda,n)(*a)
    if e: raise RuntimeError(f"{n}: {e}")
cu("cuInit",0); ctx=P(); cu("cuDevicePrimaryCtxRetain",ct.byref(ctx),0); cu("cuCtxSetCurrent",ctx)
s=P(); cu("cuStreamCreate",ct.byref(s),1)
bin=Path.home()/"tf-bench/fused-next/wp2/bin"
m=P(); cu("cuModuleLoad",ct.byref(m),str(bin/"new-torch.cubin").encode())
f=P(); cu("cuModuleGetFunction",ct.byref(f),m,b"tf_ds_fused_prompt_idx_key_kernel")
def buf(a):
    a=np.ascontiguousarray(a); p=P(); cu("cuMemAlloc_v2",ct.byref(p),ct.c_size_t(a.nbytes)); cu("cuMemcpyHtoD_v2",p,P(a.ctypes.data),ct.c_size_t(a.nbytes)); return p
rows=128; d=128; base=115; rd=64
x=buf(np.random.randint(0,65535,(rows,d),"u2")); w=buf(np.random.randint(0,65535,(d,),"u2"))
cos=buf(np.cos(np.random.rand(base+rows,32)).astype("f4")); sin=buf(np.sin(np.random.rand(base+rows,32)).astype("f4"))
codes=buf(np.zeros((base+rows,d//2),"u1")); scales=buf(np.zeros((base+rows,d//32),"u1"))
lanes=32; y=2
args=[x,L(d),w,F(1e-20),F(1/d),cos,sin,L(base),I(1),codes,scales,I(d),I(rd),I(0)]
argv=(P*len(args))(*(ct.addressof(a) for a in args))
cu("cuLaunchKernel",f,rows,1,1,lanes,y,1,0,s,argv,None)
print("sync:", cuda.cuStreamSynchronize(s))
