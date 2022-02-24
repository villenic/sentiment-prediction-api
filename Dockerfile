FROM ubuntu:20.04

RUN mkdir -p /data

ADD requirements.txt /
ADD main.py /
ADD data/DisneylandReviews.csv /data

RUN apt update \
&& apt-get install python3-pip -y \
#&& apt-get install -y apt-utils \
&& pip install -r requirements.txt \
&& pip3 install --user -U nltk \
&& pip3 install --user -U numpy \
#&& python3 -m nltk.downloader all \
&& python3 -m nltk.downloader stopwords

EXPOSE 8000

CMD uvicorn main:server --host 0.0.0.0
