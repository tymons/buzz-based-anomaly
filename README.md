# Buzz-Based Anomaly

![unittest](https://github.com/tymons/buzz-based-anomaly/actions/workflows/code-check-anaconda.yml/badge.svg)

This repo contains scripts and utils for buzz-based bee anomaly detection models. We utilize multiple different methods for e.g. swarming, pest attact, queenless detection tasks.
Models are tested against anomalies which are (for now) just different colonies sounds. In presented paper [NU-HIVE](https://zenodo.org/record/1321278) project data are labeled as anomalies for ML models. This reflects real world scenario
as the data comes from different devices and bees subspecies. 

## Data preprocess

In order to build working dataset one should use ```data_prepare.py``` script. Script will scan for nu-hive data in **dataset/nu-hive/** folder and extract only bees sound from available data.
It can be used as anomaly or for trained models.

One could download data from smartula server to train own models. Mind that this requires `SMARTULA_API` and `SMARTULA_TOKEN` environments to be set.
Smartula raw data should be preprocessed in order to reject samples which are too silent or distorted sounds.

Example for preparing only NU-HIVE data:
```shell
python data_prepare.py
```
extending date prepare with smartula data involves additional arguments about dates (start/end) and hive sns:
```shell
python data_prepare.py --start YYYY-MM-DD --end YYYY-MM-DD --hives DEADBEEF99
```
## Model train
Model training entrypoint is based on `train.py` script. Currently, only few models and sound features are supported. Mind that this list will change.

- **Models**
  - Vanilla Autoencoder (_autoencoder_)
  - Convolutional 1D Autoencoder (_conv1d_autoencoder_)


- **Features**
  - Periodogram (_periodogram_) for models: autoencoder, conv1d_autoencoder
  

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

