FROM python:3.6

MAINTAINER Santi Iglesias "siglesias@metodica.es"

ADD ./prepare.sh /workspace/
RUN /workspace/prepare.sh

WORKDIR /workspace/

# Set the default command to python3
CMD ["/bin/bash"]
