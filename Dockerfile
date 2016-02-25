FROM ubuntu:latest
RUN apt-get update
RUN apt-get install --yes wget build-essential
RUN wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
RUN chmod +x miniconda.sh
RUN ./miniconda.sh -b
ENV PATH /root/miniconda2/bin:$PATH
RUN conda update --yes conda
RUN conda update --yes pip
RUN conda install --yes python=2.7 numpy cython scipy numba
RUN pip install limix_build
RUN pip install limix_util
#RUN mkdir /limix-qep
#WORKDIR /limix-qep
#ADD . /limix-qep
