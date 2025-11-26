#!/bin/bash
# Wrapper script to run generate_gwsamples.py with proper flags
python -u "$(dirname "$0")/generate_gwsamples.py" --undefok=coordination_agent_recoverable "$@"

