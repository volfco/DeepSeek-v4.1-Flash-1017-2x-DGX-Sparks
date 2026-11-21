import sys, ctypes as ct, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path.home()/"tf-bench/fused-next/tools"))
cuda = ct.CDLL("libcuda.so.1"); P=ct.c_void_p; I=ct.c_int; L=ct.c_int64
def cu(n,*a):
    e=getattr(cuda,n)(*a)
    if e: raise RuntimeError(f"{n}: {e}")
cu("cuInit",0); ctx=P(); cu("cuDevicePrimaryCtxRetain",ct.byref(ctx),0); cu("cuCtxSetCurrent",ctx)
s=P(); cu("cuStreamCreate",ct.byref(s),1)
bin=Path.home()/"tf-bench/fused-next/wp2/bin"
m=P(); cu("cuModuleLoad",ct.byref(m),str(bin/"old-torch.cubin").encode())
def fn(name):
    f=P(); cu("cuModuleGetFunction",ct.byref(f),m,name.encode()); return f
def buf(a):
    a=np.ascontiguousarray(a); p=P(); cu("cuMemAlloc_v2",ct.byref(p),ct.c_size_t(a.nbytes)); cu("cuMemcpyHtoD_v2",p,P(a.ctypes.data),ct.c_size_t(a.nbytes)); return p
rows=128; d=128; base=115; half=32
x=buf((np.random.randn(rows,1,d)*1000).astype(np.float32).view(np.uint16))
cos=buf(np.cos(np.random.rand(base+rows,32)).astype("f4")); sin=buf(np.sin(np.random.rand(base+rows,32)).astype("f4"))
pos=buf(np.arange(base,base+rows,dtype="i8"))
args=[x,L(d),I(d-64),cos,sin,pos,I(rows),I(half),I(0),I(0)]
argv=(P*len(args))(*(ct.addressof(a) for a in args))
cu("cuLaunchKernel",fn("tf_ds_rope_kernel"),(rows*32+255)//256,1,1,256,1,1,0,s,argv,None)
print("sync:", cuda.cuStreamSynchronize(s))
