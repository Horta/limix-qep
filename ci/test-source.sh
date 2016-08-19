#!/bin/bash

set -e -x

pushd limix-qep
python setup.py test
popd
