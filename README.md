# Buzz-based anomaly

This repo contains scripts and utils for buzz-based bee anomaly detection model. We utilize multiple different methods for e.g. swarming, pest attact, queenless detection tasks.

## Data preparation

In order to build working dataset one should use ```data_utils.py``` script. In order to download data from smartula server `SMARTULA_API` and `SMARTULA_TOKEN` environemnts should be set. 
By default script will scan for nu-hive data and extract bees sound from available data. It can be used as anomaly or for trained models.

### Model train

TBA

## Docker support 

Repo has ready to use docker images at [dockerhub/tymonzz](https://hub.docker.com/repository/docker/tymonzz/buzz-based-anomaly)
with entrypoint set to `train.py` script. Example script: 

```shell
docker run  --gpus all \
              -v /c/your/path/to/data/folder:/data \
              -v /c/your/path/to/output/folder:/output \
              tymonzz/buzz-based-anomaly:cuda-11.1 \
              conv1d_autoencoder periodogram /data \
              --filter_hives DEADBEEF93 DEADBEEF94 DEADBEEF95 \
              --filter_dates 2020-08-10 2020-09-30 \
              --log_folder /output/logs \
              --model_output /output/models \
              --find_best 4
```

_Note that if you are using git-for-windows or other emulators MSYS_NO_PATHCPMV=1 environment variable should be set 
in order to correctly parse paths for data and output volumes._
### Building own docker image

To build cuda-capable docker image simply use `build_docker.sh` script. By default, your 
host cuda version will be checked and corresponding image from [nvidia/cuda](https://hub.docker.com/r/nvidia/cuda) 
docker hub repository will be downloaded. Similarly, [PyTorch Cuda Toolkit](https://pytorch.org/) will be downloaded with
version matching your cuda installation.

To overwrite default cuda versions and use options: `-p` for _PyTorch_ and `-d` for _nvidia cuda_.
For example, pytorch cuda toolkit _v10.2_ and nvidia/cuda_ docker image with _v11.2.0_ could be overwritten with:

```shell
./build_docker.sh -p 10.2 -d 11.2.0
```

