#!/usr/bin/env bash
set -euo pipefail

exec /usr/bin/gcc -std=gnu99 "$@"
