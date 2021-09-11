# Buzz-based anomaly

![unittest](https://github.com/tymons/buzz-based-anomaly/actions/workflows/code-check-anaconda.yml/badge.svg)

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
docker run -d --name buzz-based-anomaly \
              -v /ssd_local/142847ct/buzz-based-anomaly-dataset/smartula:/data \
              -v /home/macierz/142847ct/142847ct/research/buzz-based-anomaly:/io \
              --gpus all \
              -t tymonzz/buzz-based-anomaly:cuda-11.1 \
              conv1d_autoencoder periodogram /data \
              --filter_hives DEADBEEF93 DEADBEEF94 DEADBEEF95 \
              --log_folder /io/output/logs \
              --model_output /io/output/models \
              --model_config /io/input/model_config.yml \
              --feature_config /io/input/feature_config.yml \
              --learning_config /io/input/learning_config.yml \
              --comet_config /io/input/comet.config \
              --find_best 4
```

_Note that if you are using git-for-windows or other emulators MSYS_NO_PATHCPMV=1 environment variable should be set 
in order to correctly parse paths for data and output volumes._
### Building own docker image

To build cuda-capable docker image simply use `build_docker.sh` script. By default, your 
host cuda version will be parsed and image from [nvidia/cuda](https://hub.docker.com/r/nvidia/cuda) tagged with same version
will be downloaded. Similarly, [PyTorch Cuda Toolkit](https://pytorch.org/) should be downloaded with
version matching your cuda installation. **Note that your host cuda version has format MAJOR.MINOR - probably there is no nvidia docker image/anaconda pytorch cuda toolkit which match that format.**. 

To overwrite default cuda versions just use options: `-p` for _PyTorch_ and `-d` for _nvidia cuda_.
For example, pytorch cuda toolkit _v10.2_ and nvidia/cuda_ docker image with _v11.2.0_ could be overwritten with:

```shell
./build_docker.sh -p 10.2 -d 11.2.0
```

