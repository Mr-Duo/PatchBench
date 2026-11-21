#!/bin/bash

export OPENHANDS_CONFIG="/workspace/scripts/openhands_config.yaml"
export ARGS="--agents openhands --model-names claude-haiku-5.5 --tag openhands-haiku"

uv run patchbench infer    $ARGS
uv run patchbench poc      $ARGS --rerun
uv run patchbench unittest $ARGS --rerun
uv run patchbench replay   $ARGS --rerun
uv run patchbench verify   $ARGS --rerun
uv run patchbench analyze