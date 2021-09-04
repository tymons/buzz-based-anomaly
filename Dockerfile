ARG CUDA_VERSION=10.2
FROM nvidia/cuda:$CUDA_VERSION-base

ARG CUDA_VERSION
ENV CUDA_VERSION=$CUDA_VERSION

CMD nvidia-smi

WORKDIR .

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
RUN conda --version

COPY ./environment.yml /tmp/environment.yml
RUN conda env create --name pytorch-cuda-env -f /tmp/environment.yml

# manually install pytorch because environmet.yml is not working (reported bug)
# RUN conda install -n pytorch-cuda-env pytorch cudatoolkit=10.2 -c pytorch
# RUN conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge

RUN echo "conda activate pytorch-cuda-env" >> ~/.bashrc

ENV PATH /opt/conda/envs/pytorch-cuda-env/bin:$PATH
ENV CONDA_DEFAULT_ENV $pytorch-cuda-env

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "pytorch-cuda-env", "python", "train.py"]
