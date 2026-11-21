FROM tensorfold-dsv41:exl3-tp2-234f9080b326 AS builder
ADD source.tar.gz /tensorfold/
WORKDIR /tensorfold
RUN for patch in /opt/tf-patches/*.patch; do git apply "$patch" || exit 1; done
COPY finish-grace.patch /tmp/finish-grace.patch
RUN git apply --check /tmp/finish-grace.patch && git apply /tmp/finish-grace.patch && \
    /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=ReleaseFast --prefix /opt/bench -j4 native
FROM scratch AS binary
COPY --from=builder /opt/bench/native/bin/tensorfold-native /tensorfold-native
