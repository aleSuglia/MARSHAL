#!/bin/bash
set +x
ray stop

CONFIG_PATH=$(basename $(dirname $0))
echo "CONFIG_PATH: $CONFIG_PATH"

ROLL_PATH=${PWD}
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"

ROLL_OUTPUT_DIR="./runs/playpen_wordle_withcritic_eval/$(date +%Y%m%d-%H%M%S)"
ROLL_LOG_DIR=$ROLL_OUTPUT_DIR/logs
ROLL_RENDER_DIR=$ROLL_OUTPUT_DIR/render
export ROLL_OUTPUT_DIR=$ROLL_OUTPUT_DIR
export ROLL_LOG_DIR=$ROLL_LOG_DIR
export ROLL_RENDER_DIR=$ROLL_RENDER_DIR
mkdir -p $ROLL_LOG_DIR


python examples/start_agentic_rollout_pipeline.py --config_path $CONFIG_PATH  --config_name agentic_rollout_playpen_wordle_withcritic_eval | tee $ROLL_LOG_DIR/custom_logs.log
