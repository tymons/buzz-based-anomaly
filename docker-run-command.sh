#!/bin/sh

docker run -d --name buzz-based-anomaly \
-e CUDA_VISIBLE_DEVICES=4,5,6,7 \
-v /home/tcejrowski/buzz-based-anomaly-dataset/smartula:/data \
-v /home/macierz/142847ct/142847ct/research/buzz-based-anomaly:/io \
--gpus all \
-t tymonzz/buzz-based-anomaly:cuda-11.1.1 \
autoencoder periodogram /data \
--hives DEADBEEF93 \
--filter_dates 2020-08-10 2021-08-10 \
--log_folder /io/output/logs \
--model_output /io/output/models \
--model_config /io/input/model_config.yml \
--feature_config /io/input/feature_config.yml \
--learning_config /io/input/learning_config.yml \
--comet_config /io/input/comet.config \
--gpu_ids 0 1 2 3 