#!/bin/bash
# mini-swe-agent 1.x, pinned to the release the harness was written against.
set -euo pipefail

MINI_SWE_AGENT_VERSION="1.17.4"

uv pip install 'litellm==1.93.0' "mini-swe-agent==${MINI_SWE_AGENT_VERSION}"
