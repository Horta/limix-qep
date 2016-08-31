#!/bin/bash

set -e -x

pushd limix-qep
python setup.py sdist
FILENAME=`ls dist/ | head -1`
pip install dist/$FILENAME
cd /
python -c "import limix_qep; limix_qep.test()"
popd
