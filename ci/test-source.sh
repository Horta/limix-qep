#!/bin/bash

set -e -x

pushd limix-qep
pip install build_capi -q
pip install ncephes -q
pip install limix_math -q
python setup.py test
popd
