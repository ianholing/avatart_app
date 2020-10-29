#!/bin/bash

apt update
apt install -y build-essential cmake
apt install -y libmagickwand-dev
apt install -y libgl1-mesa-glx
#apt install -y libglib2.0-0
apt install -y ffmpeg
#apt install -y uwsgi

pip install flask
pip install wand
pip install flask-socketio
pip install imutils
pip install opencv-python
pip install dlib
pip install av
pip install watchdog
pip install uwsgi
#pip install gunicorn
