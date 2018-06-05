#!/bin/bash
# May need to uncomment and update to find current packages
apt-get update
apt-get install -y cuda
apt-get install -y cuda-libraries-9-2

wget -nc https://s3-us-west-1.amazonaws.com/ymlai87416.lyft/libcudnn7_7.1.4.18-1%2Bcuda9.2_amd64.deb
wget -nc https://s3-us-west-1.amazonaws.com/ymlai87416.lyft/libcudnn7-dev_7.1.4.18-1%2Bcuda9.2_amd64.deb
wget -nc https://s3-us-west-1.amazonaws.com/ymlai87416.lyft/libcudnn7-doc_7.1.4.18-1%2Bcuda9.2_amd64.deb

dpkg -i libcudnn7_7.1.4.18-1+cuda9.2_amd64.deb
dpkg -i libcudnn7-dev_7.1.4.18-1+cuda9.2_amd64.deb
dpkg -i libcudnn7-doc_7.1.4.18-1+cuda9.2_amd64.deb

# Required for demo script! #
pip install scikit-video

# Add your desired packages for each workspace initialization
#          Add here!          #
pip install https://s3-us-west-1.amazonaws.com/ymlai87416.lyft/tensorflow-1.8.0-cp36-cp36m-linux_x86_64.whl
pip install numpy
pip install PILLOW
pip install scipy
pip install sk-video
pip install opencv-python
