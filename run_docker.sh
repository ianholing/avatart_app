#!/bin/bash
docker build -t avatart .
docker run --name=avatart_processor -v $PWD:/workspace/ avatart:latest
docker run --name=avatart_processor -v $PWD:/workspace/ avatart:latest ping www.google.es
docker exec -it avatart_processor /bin/bash

## WORKING UWSGI VERSION WORKS
docker run --name=halloween_flask -v $PWD:/workspace/ -v /media/data/tensorplayground/playground/faceEditor/MEGATRONIC\ TETRAMACRO\ K-28/web/uploads:/workspace/uploads/ -v /media/data/tensorplayground/playground/faceEditor/MEGATRONIC\ TETRAMACRO\ K-28/web/processed:/workspace/static/processed/ -v /media/data/tensorplayground/playground/faceEditor/MEGATRONIC\ TETRAMACRO\ K-28/web/tasks:/workspace/tasks/ -v /media/data/tensorplayground/playground/faceEditor/MEGATRONIC\ TETRAMACRO\ K-28/web/errors/:/workspace/errors/ -p 443:443 halloween:latest uwsgi --ini halloween.ini

## WORKING PROCESSOR
docker run --name=halloween_processor -v $PWD:/workspace/ -v /media/data/tensorplayground/playground/faceEditor/MEGATRONIC\ TETRAMACRO\ K-28/web/uploads:/workspace/uploads/ -v /media/data/tensorplayground/playground/faceEditor/MEGATRONIC\ TETRAMACRO\ K-28/web/processed:/workspace/static/processed/ -v /media/data/tensorplayground/playground/faceEditor/MEGATRONIC\ TETRAMACRO\ K-28/web/tasks:/workspace/tasks/ -v /media/data/tensorplayground/playground/faceEditor/MEGATRONIC\ TETRAMACRO\ K-28/web/errors/:/workspace/errors/ --gpus all halloween:latest python processor.py tasks/

## GUNICORN WORKS WITH ERRORS
docker run --name=halloween_flask -v $PWD:/workspace/ -v /media/data/tensorplayground/playground/faceEditor/MEGATRONIC\ TETRAMACRO\ K-28/web/uploads:/workspace/uploads/ -v /media/data/tensorplayground/playground/faceEditor/MEGATRONIC\ TETRAMACRO\ K-28/web/processed:/workspace/static/processed/ -v /media/data/tensorplayground/playground/faceEditor/MEGATRONIC\ TETRAMACRO\ K-28/web/tasks:/workspace/tasks/ -v /media/data/tensorplayground/playground/faceEditor/MEGATRONIC\ TETRAMACRO\ K-28/web/errors/:/workspace/errors/ -p 443:443 halloween:latest gunicorn --certfile ssl/certificate.crt --keyfile ssl/certificate.key --ca-certs ssl/certificate.ca.crt -b :443 server:app

## STANDALONE TO WORK ON IT
docker run --name=halloween -ti -v $PWD:/workspace/ -v /media/data/tensorplayground/playground/faceEditor/MEGATRONIC\ TETRAMACRO\ K-28/web/uploads:/workspace/uploads/ -v /media/data/tensorplayground/playground/faceEditor/MEGATRONIC\ TETRAMACRO\ K-28/web/processed:/workspace/static/processed/ -v /media/data/tensorplayground/playground/faceEditor/MEGATRONIC\ TETRAMACRO\ K-28/web/tasks:/workspace/tasks/ -v /media/data/tensorplayground/playground/faceEditor/MEGATRONIC\ TETRAMACRO\ K-28/web/errors/:/workspace/errors/ -p 80:80 -p 443:443 --gpus all pytorch/pytorch
