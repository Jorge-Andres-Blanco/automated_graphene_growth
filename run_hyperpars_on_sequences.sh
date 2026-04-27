#!/bin/bash
ENV="/data/lmcat/Computer_vision/ai_project/ai_project_env/bin/python"

SCRIPT="/data/lmcat/Computer_vision/automated_graphene_growth/action_vs_frame_eval_hyperpars_on_sequence.py"

$ENV $SCRIPT --hiddim 128 > verb_hiddim128.txt 2>&1 &
$ENV $SCRIPT --hiddim 256 > verb_hiddim256.txt 2>&1 &
$ENV $SCRIPT --hiddim 512 > verb_hiddim512.txt 2>&1 &

wait

echo "All grid searches completed"