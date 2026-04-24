#!/bin/bash
ENV="/data/lmcat/Computer_vision/ai_project/ai_project_env/bin/python"

SCRIPT="/data/lmcat/Computer_vision/automated_graphene_growth/action_vs_frame_eval_hyperpars_on_sequence.py"

$ENV $SCRIPT --hist 1 > verb_hist1.txt 2>&1 &
$ENV $SCRIPT --hist 2 > verb_hist2.txt 2>&1 &
$ENV $SCRIPT --hist 5 > verb_hist5.txt 2>&1 &

wait

echo "All grid searches completed"