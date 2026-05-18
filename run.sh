#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -x ".venv/bin/python" ]; then
  echo "ERROR: Virtual environment .venv not found. Create it with: python3.13 -m venv .venv"
  exit 1
fi
.venv/bin/python run.py
