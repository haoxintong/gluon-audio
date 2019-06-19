#!/bin/bash

DOC_ROOT=$(dirname "${BASH_SOURCE[0]}")
echo $DOC_ROOT && cd $DOC_ROOT

# remove history build
make clean

# generate module docs
# sphinx-apidoc -f -e -o ./source/api ../gluonar/

# build doc htmls
make html

