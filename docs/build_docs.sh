#!/bin/bash

cd ~/uqp/core/docs/sphinx
make html
cp ~/uqp/core/docs/sphinx/build/html /collab/usr/gapps/uq/UQPipeline/doc/tmp/ -rf
chmod 777 /collab/usr/gapps/uq/UQPipeline/doc/tmp/html -R
xsu vnvadm -c '/collab/usr/gapps/uq/UQPipeline/doc/tmp/deploy.sh'

