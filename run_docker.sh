#!/bin/bash
docker build -t avatart .

#TEST ENV
docker run --name=avatart_test -v $PWD:/workspace/ avatart:latest ping www.google.es
docker exec -it avatart_test /bin/bash

#WORKING:
docker run --name=avatart_processor -v $PWD:/workspace/ avatart:latest python3 processor.py tasks/

# NO TESTED YET
docker run --name=avatart_flask -v $PWD:/workspace/ -p 80:80 -p 443:443 avatart:latest uwsgi --ini halloween_nossl.ini


## GUNICORN WORKS WITH ERRORS
docker run --name=halloween_flask -v $PWD:/workspace/ -p 443:443 halloween:latest gunicorn --certfile ssl/certificate.crt --keyfile ssl/certificate.key --ca-certs ssl/certificate.ca.crt -b :443 server:app

## STANDALONE TO WORK ON IT
docker run --name=halloween -ti -v $PWD:/workspace/ -p 80:80 -p 443:443 --gpus all pytorch/pytorch
