#!/usr/bin/env bash
# One-shot waiter: run a3's queue once the a5 m21 sampler has released the GPU.
#
# a3/joint_complete_s100 was appended to a3's queue after the single-pass runner
# had already cleared a3, so it never ran. run_campaign.sh's skip logic means
# this pass touches nothing else: the other four rungs have their JSON already.
#
# It waits rather than starting now because m21 (a5's last rung) is mid-flight on
# the same H100 and contention would slow a run that is already 7h+ deep.
set -uo pipefail
REDO=/media/volume/darksirens-data/gws-agn-js2-data/analyses/selection_redo
stamp() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }

echo "$(stamp) [wait] holding for the a5 m21 sampler to exit"
while pgrep -f 'sample_4d\.py' > /dev/null; do sleep 300; done
echo "$(stamp) [wait] GPU released; settling 60s"
sleep 60

echo "$(stamp) [run] run_campaign.sh a3"
"$REDO/scripts/run_campaign.sh" a3
echo "$(stamp) [done] exit=$?"
