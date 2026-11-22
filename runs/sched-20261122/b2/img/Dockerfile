FROM tensorfold-dsv41:bench-fused-next-10c9866
COPY nsys-wrap.sh /nsys-wrap.sh
RUN chmod +x /nsys-wrap.sh
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "/nsys-wrap.sh"]
