#!/bin/bash

echo "-> Running build docker image script..."
HOST_CUDA_VERSION=$(nvcc --version | sed -n 's/^.*release \([0-9\.]\+\).*$/\1/p')
echo "-> Running on host with cuda:$HOST_CUDA_VERSION"

PYTORCH_CUDA_VERSION=$HOST_CUDA_VERSION
DOCKER_CUDA_VERSION=$DOCKER_CUDA_VERSION

while getopts ":d:p:" opt; do
  case $opt in
    d) DOCKER_CUDA_VERSION="$OPTARG"
       echo "-> using nvidia/cuda docker image with version $OPTARG"
    ;;
    p) PYTORCH_CUDA_VERSION="$OPTARG"
      echo "-> using pytorch with cuda toolkit with version $OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

if [ "$PYTORCH_CUDA_VERSION" != "$DOCKER_CUDA_VERSION" ]
then
  echo "-> ! WARNING ! possible mismatch for local cuda version, docker image and pytorch cuda toolkit installer!"
  echo "     pytorch_v($PYTORCH_CUDA_VERSION) / docker_cuda_v($DOCKER_CUDA_VERSION)"
fi

cp environment.yml environment_backup.yml
awk -v cuda_ver="$PYTORCH_CUDA_VERSION" '{sub(/\${CUDA_TOOLKIT_VERSION}/, cuda_ver)}1' environment_backup.yml > environment.yml
docker build -t "buzz-based-anomaly:cuda-$DOCKER_CUDA_VERSION" . --build-arg CUDA_VERSION="$DOCKER_CUDA_VERSION"
mv environment_backup.yml environment.yml
