#!/bin/bash

set -eux

sh ./tools/install-uv.sh
export PATH=$PATH:$HOME/.local/bin
uv tool install dvc[s3]
dvc pull

for i in $(find . -name \*.tar.bz); do
    (
	cd $(dirname $i)
	tar -xjf $(basename $i)
    )
done
