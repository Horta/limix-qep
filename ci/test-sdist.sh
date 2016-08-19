#!/bin/bash

set -e -x

pushd limix-qep
pip install build_capi -q
pip install ncephes -q
python setup.py sdist
FILENAME=`ls sdist/ | head -1`
pip install sdist/$FILENAME
python -c "import limix_qep; limix_qep.test()"
popd
