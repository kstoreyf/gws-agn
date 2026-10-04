#!/bin/bash
# Run ON THE VM: one-line-per-field status of the newest A13 run (used by hilda-side monitors).
L=/media/volume/tbs/gws-agn-data-js2/analysis_13_joint_shared_widths/logs
T=$(ls -t $L/*.out 2>/dev/null | head -1); T=${T%.out}
pgrep -f "bin/python a13_core_sampler.py" >/dev/null && echo "STATE RUNNING" || echo "STATE STOPPED"
echo "RUN $(basename $T)"
grep -hE '\[preflight\]|\[fatal\]|exit=' $T.out | sed 's/^/OUT /'
grep -hE 'Traceback|RESOURCE_EXHAUSTED|Error:|Killed' $T.err 2>/dev/null | grep -v Warning | tail -3 | cut -c1-200 | sed 's/^/ERR /'
echo "MEM $(tail -1 $T.mem 2>/dev/null)"
echo "IT $(tail -c 2000 $T.err 2>/dev/null | tr '\r' '\n' | grep -E 'it \[' | tail -1 | cut -c1-170)"
