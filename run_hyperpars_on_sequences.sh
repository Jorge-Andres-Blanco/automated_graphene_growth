#!/bin/bash
ENV="/data/lmcat/Computer_vision/ai_project/ai_project_env/bin/python"

SCRIPT="/data/lmcat/Computer_vision/automated_graphene_growth/action_vs_frame_eval_hyperpars_on_sequence.py"

$ENV $SCRIPT --hiddim 128 --activation "relu" > verb_hiddim128_relu.txt 2>&1 &
$ENV $SCRIPT --hiddim 512 --activation "relu" > verb_hiddim512_relu.txt 2>&1 &
$ENV $SCRIPT --hiddim 128 --activation "leaky_relu" > verb_hiddim512_leakyRelu.txt 2>&1 &
$ENV $SCRIPT --hiddim 512 --activation "leaky_relu" > verb_hiddim512_leakyRelu.txt 2>&1 &

wait

echo "All grid searches completed"