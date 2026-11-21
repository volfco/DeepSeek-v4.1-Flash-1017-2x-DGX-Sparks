"""Package only the benchmark head and worker executables on their pinned runtime base."""
import hashlib
import io
from pathlib import Path
import sys
import tarfile

def pack(root):
    files=("native/bin/tensorfold-native","bin/tf-dsv41-lanes")
    digest=hashlib.sha256(b"".join(hashlib.sha256((root/n).read_bytes()).digest() for n in files)).hexdigest()[:12]
    path=Path("/tmp")/f"gb10-opt-{digest}.tar.gz"
    recipe=("FROM tensorfold-dsv41:exl3-tp2-234f9080b326\n"+"".join(f"COPY {n} /opt/tensorfold/{n}\n" for n in files)).encode()
    with tarfile.open(path,"w:gz",compresslevel=2) as t:
        info=tarfile.TarInfo("Dockerfile");info.size=len(recipe)
        t.addfile(info,io.BytesIO(recipe))
        for n in files:t.add(root/n,arcname=n,recursive=False)
    print(digest,path)
if __name__=="__main__":pack(Path(sys.argv[1]))
