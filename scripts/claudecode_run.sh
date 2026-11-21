#!/bin/bash

export CLAUDECODE_CONFIG="/workspace/scripts/claudecode_config.yaml"
export ARGS="--agents claudecode --model-names claude-haiku-5.5 --tag claude-haiku"

uv run patchbench infer    $ARGS
uv run patchbench poc      $ARGS --rerun
uv run patchbench unittest $ARGS --rerun
uv run patchbench replay   $ARGS --rerun
uv run patchbench verify   $ARGS --rerun
uv run patchbench analyze