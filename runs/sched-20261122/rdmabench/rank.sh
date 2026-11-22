#!/bin/sh
# Inside the bench container: run tf-dsv41-rdma-bench, pin only its tf-rdma-proxy thread to CPU 15 (as production's
# TF_DS_RDMA_CPUS=15 does from inside the engine; the bench never passes .cpu), record the proxy's scheduling counters.
# args: RANK DEVICES ROWS WIDTH REPS
R=$1; DEV=$2; ROWS=$3; W=$4; REPS=$5
/opt/tensorfold/bin/tf-dsv41-rdma-bench $R 2 192.168.100.1 /b/rdma.fatbin $DEV --rows $ROWS --width $W --reps $REPS --pdl 1 --gid 3 --port 29700 > /b/out.$R.json 2> /b/err.$R.log &
P=$!
for i in $(seq 600); do
  T=$(grep -l tf-rdma-proxy /proc/$P/task/*/comm 2>/dev/null | head -1 | cut -d/ -f5)
  [ -n "$T" ] && break; sleep 0.05
done
if [ -n "$T" ]; then taskset -pc 15 $T > /b/pin.$R.txt 2>&1; else echo "no proxy thread seen" > /b/pin.$R.txt; fi
grep -E 'Cpus_allowed_list|ctxt' /proc/$P/task/$T/status > /b/proxy0.$R.txt 2>/dev/null; cat /proc/$P/task/$T/schedstat >> /b/proxy0.$R.txt 2>/dev/null
wait_and_snap() { while kill -0 $P 2>/dev/null; do grep -E 'Cpus_allowed_list|ctxt' /proc/$P/task/$T/status > /b/proxy1.$R.txt 2>/dev/null; cat /proc/$P/task/$T/schedstat >> /b/proxy1.$R.txt 2>/dev/null; sleep 0.2; done; }
wait_and_snap; wait $P; echo "rc=$?" >> /b/err.$R.log
