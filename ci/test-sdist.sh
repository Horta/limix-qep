#!/bin/bash

set -e -x

pushd limix-qep
pip install build_capi -q
pip install ncephes -q
pip install limix_math -q
python setup.py sdist
FILENAME=`ls dist/ | head -1`
pip install dist/$FILENAME
python -c "import limix_qep; limix_qep.test()"
popd
