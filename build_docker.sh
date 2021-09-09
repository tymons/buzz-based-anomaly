#!/bin/bash

echo "-> Running build docker image script..."
HOST_CUDA_VERSION=$(nvcc --version | sed -n 's/^.*release \([0-9\.]\+\).*$/\1/p')
echo "-> Running on host with cuda:$HOST_CUDA_VERSION"

PYTORCH_CUDA_VERSION=HOST_CUDA_VERSION
DOCKER_CUDA_VERSION=HOST_CUDA_VERSION

IFS='.'
read -a strarr <<< "$HOST_CUDA_VERSION"
if [ $((strarr[0])) -gt 10 ]
  then
  PYTORCH_CUDA_VERSION=11.1
  if [ $((strarr[0])) -gt 11 ]
    then
    DOCKER_CUDA_VERSION=$HOST_CUDA_VERSION.0
  else
    DOCKER_CUDA_VERSION=$HOST_CUDA_VERSION.1
  fi
else
  PYTORCH_CUDA_VERSION=10.2
fi

while getopts ":d:p:" opt; do
  case $opt in
    d) DOCKER_CUDA_VERSION="$OPTARG"
       echo "-> overwriting host DOCKER cuda version with value $OPTARG"
    ;;
    p) PYTORCH_CUDA_VERSION="$OPTARG"
      echo "-> overwriting host PYTORCH cuda version with value $OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

echo "-> Using pytorch cudatoolkit of version $PYTORCH_CUDA_VERSION"
if [ "$PYTORCH_CUDA_VERSION" != "$HOST_CUDA_VERSION" ] || [ "$PYTORCH_CUDA_VERSION" != "$DOCKER_CUDA_VERSION" ]
then
  echo "-> ! WARNING ! possible mismatch for local cuda version and pytorch installer!"
  echo "     pytorch_v($PYTORCH_CUDA_VERSION) / docker_cuda_v($DOCKER_CUDA_VERSION) / cuda_host_v($HOST_CUDA_VERSION)"
fi

cp environment.yml environment_backup.yml
awk -v cuda_ver="$PYTORCH_CUDA_VERSION" '{sub(/\${CUDA_TOOLKIT_VERSION}/, cuda_ver)}1' environment_backup.yml > environment.yml
docker build -t "tymons/buzz-based-anomaly:cuda-$PYTORCH_CUDA_VERSION" . --build-arg CUDA_VERSION="$DOCKER_CUDA_VERSION"
mv environment_backup.yml environment.yml
