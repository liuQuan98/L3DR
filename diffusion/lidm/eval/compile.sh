#!/bin/sh
# Build the CUDA extensions required by the CD / EMD metrics.
# NOTE: upstream shipped `cd modules/chamfer`, but the directories are
# `chamfer2D` and `chamfer3D`. Fixed here.
set -e
cd "$(dirname "$0")/modules"

for d in chamfer2D chamfer3D emd; do
    echo ">>> building $d"
    (cd "$d" && python setup.py build_ext --inplace)
done

echo "done."
