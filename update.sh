#!/usr/bin/env bash


cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


cd "fedlag"

find . -type d -name "__pycache__" -exec rm -r {} +
git add .
git commit -m "update"
git push -u origin main