
# get tensorflow image
FROM tensorflow/tensorflow:latest-py3
COPY . /repos/prevision-quantum-nn
WORKDIR /repos

MAINTAINER Michel Nowak "michel.nowak@prevision.io"

# install dependencies
RUN pip3 install --upgrade pip

# install prevision-quantum-nn
RUN cd /repos/prevision-quantum-nn/ && \
    pip3 install --no-cache-dir -r requirements.txt --use-feature=2020-resolver && \
    pip3 install .
