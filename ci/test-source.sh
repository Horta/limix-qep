#!/bin/bash

set -e -x

pushd limix-qep
pip install build_capi -q
pip install ncephes -q
python setup.py test
popd
