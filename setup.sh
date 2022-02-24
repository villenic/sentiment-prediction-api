#!/bin/bash

mkdir -p ~/projet2/
cd ~/projet2/

# sudo apt update && apt install python3-pip -y && pip install -r requirements.txt

# docker-compose down
# docker volume rm shared_volume
# docker image rm server

# du -sh /var/lib/docker
# docker system prune --all --force
# docker system prune --all --force --volumes

docker image build . -t villenic/sentiment-prediction-api:1.0.0

# docker image pull villenic/sentiment-prediction-api:1.0.0

docker login -u "villenic" -p "Ulysse31#" docker.io
docker image push villenic/sentiment-prediction-api:1.0.0

docker-compose up

