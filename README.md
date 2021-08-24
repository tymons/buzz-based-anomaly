# Buzz-based anomaly

This repo contains scripts and utils for buzz-based bee anomaly detection model. We utilize multiple different methods for e.g. swarming, pest attact, queenless detection tasks.

### Data preparation

In order to build working dataset one should use ```data_utils.py``` script. In order to download data from smartula server `SMARTULA_API` and `SMARTULA_TOKEN` environemnts should be set. 
By default script will scan for nu-hive data and extract bees sound from available data. It can be used as anomaly or for trained models.

### Model train

TBA