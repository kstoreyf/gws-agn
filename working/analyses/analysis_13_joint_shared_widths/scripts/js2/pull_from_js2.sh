#!/bin/bash
# Run on hilda. Copies the VM's A13 results, checkpoints and logs back to
# $HILDA_RESULTS (phy230054p), and the small result files (json/npz) into the repo's analysis dir.
set -euo pipefail
source "$(dirname "$0")/config.sh"
DEST=$HILDA_RESULTS/analysis_13_joint_shared_widths
mkdir -p $DEST
rsync -a --partial $JS2:$JS2_A13/{results,queue,logs} $DEST/
rsync -a $DEST/results/ $A13/results/
echo "pulled to $DEST: $(ls $DEST/results | wc -l) results, $(ls $DEST/queue | wc -l) queue files"
