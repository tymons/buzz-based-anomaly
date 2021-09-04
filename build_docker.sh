#!/bin/bash

echo "-> Running build docker image script..."
CUDA_VERSION=$(nvcc --version | sed -n 's/^.*release \([0-9\.]\+\).*$/\1/p')
echo "-> Your device has cuda:$CUDA_VERSION"

PYTORCH_CUDA_VERSION=$CUDA_VERSION
DOCKER_CUDA_VERSION=$CUDA_VERSION

IFS='.'
read -a strarr <<< "$CUDA_VERSION"
if [ $((strarr[0])) -gt 10 ]
then
  PYTORCH_CUDA_VERSION=11.1
  DOCKER_CUDA_VERSION=$CUDA_VERSION.0
else
  PYTORCH_CUDA_VERSION=10.2
fi

echo "-> Using pytorch cudatoolkit of version $PYTORCH_CUDA_VERSION"
if [ "$PYTORCH_CUDA_VERSION" != "$CUDA_VERSION" ]
then
  echo "-> ! WARNING ! possible mismatch for local cuda version and pytorch installer!"
fi

cp environment.yml environment_backup.yml
awk -v cuda_ver="$PYTORCH_CUDA_VERSION" '{sub(/\${CUDA_TOOLKIT_VERSION}/, cuda_ver)}1' environment_backup.yml > environment.yml
docker build -t buzz-based-anomaly . --build-arg CUDA_VERSION="$DOCKER_CUDA_VERSION"
mv environment_backup.yml environment.yml
