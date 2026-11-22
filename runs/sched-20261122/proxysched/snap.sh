#!/bin/sh
# One rank's snapshot (run on the host): the engine container's tf-rdma-proxy thread's CPU mask, context switches,
# migrations, schedstat (run ns, wait-on-runqueue ns, timeslices); and every other engine thread allowed on CPU 15.
P=$(docker inspect -f '{{.State.Pid}}' dsv41-1017-exl3-tf)
for t in /proc/$P/task/*; do
  n=$(cat $t/comm 2>/dev/null)
  if [ "$n" = tf-rdma-proxy ]; then
    echo "proxy tid=${t##*/} $(grep -E 'Cpus_allowed_list' $t/status | tr -s '\t ' ' ') $(grep -E 'ctxt' $t/status | tr -s '\t\n ' ' ') migrations=$(grep -E '^se.nr_migrations' $t/sched 2>/dev/null | awk '{print $3}') schedstat=$(cat $t/schedstat)"
  else
    echo "other tid=${t##*/} comm=$n allowed=$(grep Cpus_allowed_list $t/status | awk '{print $2}') run_ns=$(cut -d' ' -f1 $t/schedstat)"
  fi
done
