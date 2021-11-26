# Buzz-Based Anomaly

![unittest](https://github.com/tymons/buzz-based-anomaly/actions/workflows/code-check-anaconda.yml/badge.svg)

This repo contains scripts and utils for buzz-based bee anomaly detection models. We utilize different methods for e.g. swarming, pest attact, queenless detection tasks.
Models are tested against anomalies which are (for now) just different colonies sounds. The [NU-HIVE](https://zenodo.org/record/1321278) project data are labeled as anomalies for ML models. This reflects real world scenario as the data comes from different devices and bees subspecies. 

# Data preprocess

In order to build working dataset one should use ```data_prepare.py``` script along with specified option.
There are three different options to perform data processing: 
 
* **NU-HIVE BEES' DATA EXTRACTION OPTION**


This step extracts only bees' sound data from nuhive dataset. New WAV sound files
are saved within created `nu-hive-processed/` folder. 
To execute such option run:

```shell
$ data-prepare.py extract-nuhive-bees --data_folder C:\NUHIVE_FOLDER
```

* **FRAGMENT SOUNDS**  

In order to use some fancy methods with `train.py` script one should have coherent dataset. To fragment
all wav sounds to files with equal length `fragment-sound-bees` option should be used. Additional options are
`--duration` which is length in seconds for every fragment. Example:

```shell
$ data-prepare.py fragment-sound --data_folder C:\NUHIVE_FOLDER --duration 2
```

* **UPSAMPLED SOUNDS**  

Machine Learning models require unified input where e.g. spectrogram should exhibit unified shape for the 
full dataset. In order to handle such scenario one could use `upsample-sound` option. Such utility upsample 
all audio files within a folder to given frequency. Audio data which already satisfy sampling rate requirement 
will be preserved. The rest of data will be upsampled and saved as new file where parent audio will be deleted.  
```shell
$ data-prepare.py upsample-sound --data_folder C:\NUHIVE_FOLDER --sampling_rate 44100
```


* **DOWNLOAD SMARTULA DATA**

One could download data from smartula server to train own models. Mind that this requires `SMARTULA_API` and `SMARTULA_TOKEN` environments to be set.
Smartula raw data will be preprocessed in order to reject samples which are too silent or distorted sounds.

```shell
$ python data_prepare.py --start YYYY-MM-DD --end YYYY-MM-DD --smartula_hives DEADBEEF99
```

# Model train
Model training entrypoint is based on `train.py` script. Currently, only listed models and sound features are supported.
Names in italics are direct arguments to the `train.py` script.

#### Vanilla Autoencoders
- :white_check_mark: Autoencoder (`autoencoder`)
  - periodogram (`periodogram`)
- :white_check_mark: Convolutional 1D Autoencoder (`conv1d_autoencoder`)
  - Periodogram (`periodogram`)
- :white_check_mark: Convolutional 2D Autoencoder (`conv2d_autoencoder`)
  - Spectrogram (`spectrogram`)
  - MelSpectrogram (`melspectrogram`)
  - MFCC (`mfcc`)

#### Variational Autoencoders
- :white_check_mark: Variational Autoencoder (`vae`)
  - periodogram (`periodogram`)
- :white_check_mark: Convolutional 1D Variational Autoencoder (`conv1d_vae`)
  - periodogram (`periodogram`)
- :white_check_mark: Convolutional 2D Variational Autoencoder (`conv2d_vae`)
  - Spectrogram (`spectrogram`)
  - MelSpectrogram (`melspectrogram`)
  - MFCC (`mfcc`)
  
#### Contrastive Autoencoders

>Mind that for contrastive learning `--contrastive_data_folder` argument should be passed. 
All data from contrastive data folder will be transformed to feature passed as 1st argument of train script.
Contrastive data will be shuffled and truncated to the length of original dataset. 

>**For now, contrastive data
should originate from the same source and has same parameters (eg. sampling frequency, sound length) as 
target data**.

- :white_check_mark: Contrastive Autoencoder (`contrastive_autoencoder`)
  - Periodogram (`periodogram`)
- :white_check_mark: Contrastive Convolutional 1D Autoencoder (`contrastive_conv1d_autoencoder`)
  - Periodogram (`periodogram`)
- :white_check_mark: Contrastive Convolutional 2D Autoencoder (`contrastive_conv2d_autoencoder`)
  - Spectrogram (`spectrogram`)
  - MelSpectrogram (`melspectrogram`)
  - MFCC (`mfcc`)
- :white_check_mark: Contrastive Variational Autoencoder (`contrastive_vae`)
  - Periodogram (`periodogram`)
- :white_check_mark: Contrastive Convolutional Variational 1D Autoencoder (`contrastive_conv1d_vae`)
  - Periodogram (`periodogram`)
- :white_check_mark: Contrastive Convolutional Variational 2D Autoencoder (`contrastive_conv2d_vae`)
  - Spectrogram (`spectrogram`)
  - MelSpectrogram (`melspectrogram`)
  - MFCC (`mfcc`)


## Docker support 

Repo has ready to use docker images at [dockerhub/tymonzz](https://hub.docker.com/repository/docker/tymonzz/buzz-based-anomaly)
with entrypoint set to `train.py` script. Example script: 

```shell
$ docker run -d --name buzz-based-anomaly \
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

_Note that if you are using git-for-windows or other shell emulators MSYS_NO_PATHCPMV=1 environment variable should be set 
in order to correctly parse paths for data and output volumes._
### Building own docker image

To build cuda-capable docker image simply use `build_docker.sh` script. By default, your 
host cuda version will be parsed and image from [nvidia/cuda](https://hub.docker.com/r/nvidia/cuda) tagged with same version
will be downloaded. Similarly, [PyTorch Cuda Toolkit](https://pytorch.org/) should be downloaded with
version matching your cuda installation. **Note that your host cuda version has format MAJOR.MINOR - probably there is no nvidia docker image/anaconda pytorch cuda toolkit which match that format.**. 

To overwrite default cuda versions just use options: `-p` for _PyTorch_ and `-d` for _nvidia cuda_.
For example, pytorch cuda toolkit _v10.2_ and nvidia/cuda_ docker image with _v11.2.0_ could be overwritten with:

```shell
$ ./build_docker.sh -p 10.2 -d 11.2.0
```

