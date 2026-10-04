# A13 on the Jetstream2 A100 VM (ssh alias js2a100 = exouser@149.165.174.18).
# Sourced by the hilda-side scripts. J2 cannot reach hilda, so every transfer runs FROM hilda.
JS2=js2a100
JS2_ROOT=/media/volume/tbs/gws-agn-data-js2          # everything on the VM lives here
JS2_ALT=/media/volume/h0tbs/gws-agn-data-js2         # same layout, if tbs runs short of space
JS2_ENV=$JS2_ROOT/env/core-bf58aa6                   # venv: core's exact pins, cloned from ~/qchieffenv's set
JS2_CORE=$JS2_ROOT/src/darksirens-core-bf58aa6       # git archive of core bf58aa6 (path name checked by the driver)
JS2_A13=$JS2_ROOT/analysis_13_joint_shared_widths    # scripts/ results/ queue/ logs/ ref/
JS2_DATA=$JS2_ROOT/data/seed100                      # only the four A13 inputs
REPO=/hildafs/projects/phy230014p/magana/gws-agn
A13=$REPO/working/analyses/analysis_13_joint_shared_widths
CORE_SRC=/hildafs/projects/phy230014p/magana/src/darksirens-core-bf58aa6
CORE_SHA=bf58aa64982864eff40392f545182745db02ece0
SEED100=$REPO/working/data/seed100
REF_CELLS=$REPO/working/analyses/experiments/experiment_darksirens_core_equivalence/results/core_bf58aa6_a11_cells.json
# results live on hilda under phy230054p, mirroring the VM tree
HILDA_RESULTS=/hildafs/projects/phy230054p/magana/gws-agn-data-js2
A13_INPUTS="surveys/survey_gal_complete_ns32.h5 surveys/survey_agn_complete_ns32.h5 events/events_marked_dmu0p10_dmuG5.h5 injections/injections_targeted.h5"
