#!/usr/bin/env bash

set -e

echo "⬇️ Downloading docking_vina..."

FILE_ID="1B7270Q4yG00TuJztwveHcLo9Mr6WaLU5"
URL="https://drive.google.com/uc?export=download&id=${FILE_ID}"

wget --no-check-certificate "$URL" -O docking_vina.zip

echo "📦 Unzipping..."
unzip docking_vina.zip
rm docking_vina.zip

echo "✅ docking_vina ready"

echo ""
echo "⚠️ IMPORTANT: Install MGLTools manually:"
echo "https://ccsb.scripps.edu/mgltools/"