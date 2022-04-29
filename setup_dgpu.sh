#!/bin/bash

# # Progress Bar
# function ProgressBar {
#     let _progress=(${1}*100/${2}*100)/100
#     let _done=(${_progress}*4)/10
#     let _left=40-$_done

#     _fill=$(printf "%${_done}s")
#     _empty=$(printf "%${_left}s")

# printf "\rProgress : [${_fill// /#}${_empty// /-}] ${_progress}%%"
# }

# Enter AssetID or MachineID
# read -p "Enter Asset ID / Machine ID: "  assetID

username=$USER
read -p "Enter Machine Password again please: " password 
cd

# # Log file
# exec 3>&1 4>&2
# trap 'exec 2>&4 1>&3' 0 1 2 3
# exec 1>iris_installation_log.out 2>&1
# set +e

sudo sed -i '5 i 3.109.91.77 iwizregistry.iwizardsolutions.com' /etc/hosts

# Check OS Version
. /etc/os-release
if [[ $VERSION_ID == "18.04" ]];
then

    # # Install Cuda 11.1
    # wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    # echo $password | sudo -kS mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    # wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda-repo-ubuntu2004-11-1-local_11.1.1-455.32.00-1_amd64.deb
    # sudo dpkg -i cuda-repo-ubuntu2004-11-1-local_11.1.1-455.32.00-1_amd64.deb
    # sudo apt-key add /var/cuda-repo-ubuntu2004-11-1-local/7fa2af80.pub
    # sudo apt-get update
    # echo $password | sudo -kS apt-get -y install cuda-11.1

    # Purge Existing Dependencies
    sudo apt -y remove --purge "^libcuda.*"
    sudo apt -y remove --purge "^cuda.*"
    sudo apt -y remove --purge "^libnvidia.*"
    sudo apt -y remove --purge "^nvidia.*"
    sudo apt -y remove --purge "^tensorrt.*" 

    sudo apt-get -y --purge remove "*cublas*" "cuda*" "nsight*" 
    sudo apt-get -y --purge remove "*nvidia*"
    sudo apt remove -y --autoremove nvidia-cuda-toolkit

    # Install Cuda 10.2
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
    echo $password | sudo -kS mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
    sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
    sudo apt-get -y update
    sudo apt -y install cuda-10-2
    sleep 5

    # Install Tensorrt 7.0
    wget https://irisretailstoragebox.blob.core.windows.net/irismodels/nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.0.0.11-ga-20191216_1-1_amd64.deb
    echo $password | sudo -kS dpkg -i nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.0.0.11-ga-20191216_1-1_amd64.deb
    sudo apt-get -y update
    sudo apt install -y tensorrt
    sudo apt -y --fix-broken install
    sudo ldconfig
    sleep 5

else 

    # Purge Existing Dependencies
    sudo apt -y remove --purge "^libcuda.*"
    sudo apt -y remove --purge "^cuda.*"
    sudo apt -y remove --purge "^libnvidia.*"
    sudo apt -y remove --purge "^nvidia.*"
    sudo apt -y remove --purge "^tensorrt.*" 

    # Purge Existing Dependencies
    sudo apt-get -y --purge remove "*cublas*" "cuda*" "nsight*" 
    sudo apt-get -y --purge remove "*nvidia*"
    sudo apt remove -y --autoremove nvidia-cuda-toolkit

	# Install Cuda 11.4
	wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
	sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
	wget https://developer.download.nvidia.com/compute/cuda/11.4.1/local_installers/cuda-repo-ubuntu2004-11-4-local_11.4.1-470.57.02-1_amd64.deb
	sudo dpkg -i cuda-repo-ubuntu2004-11-4-local_11.4.1-470.57.02-1_amd64.deb
	sudo apt-key add /var/cuda-repo-ubuntu2004-11-4-local/7fa2af80.pub
	sudo apt-get update
	sudo apt-get -y install cuda

    # Install Cuda 11.2
    #wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    #sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    #wget https://developer.download.nvidia.com/compute/cuda/11.2.1/local_installers/cuda-repo-ubuntu2004-11-2-local_11.2.1-460.32.03-1_amd64.deb
    #sudo dpkg -i cuda-repo-ubuntu2004-11-2-local_11.2.1-460.32.03-1_amd64.deb
    #sudo apt-key add /var/cuda-repo-ubuntu2004-11-2-local/7fa2af80.pub
    #sudo apt-get -y update
    #sudo apt-get -y install cuda
    #sleep 5

    # Install Cuda 11.1
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda-repo-ubuntu2004-11-1-local_11.1.1-455.32.00-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu2004-11-1-local_11.1.1-455.32.00-1_amd64.deb
    sudo apt-key add /var/cuda-repo-ubuntu2004-11-1-local/7fa2af80.pub
    sudo apt-get -y update
    sudo apt-get -y install cuda-11.1
    sleep 5

    # Install Tensorrt 7.2.2
    wget https://irisretailstoragebox.blob.core.windows.net/irismodels/nv-tensorrt-repo-ubuntu1804-cuda11.1-trt7.2.2.3-ga-20201211_1-1_amd64.deb
    sudo dpkg -i nv-tensorrt-repo-ubuntu1804-cuda11.1-trt7.2.2.3-ga-20201211_1-1_amd64.deb
    sudo apt -y update
    sudo apt install -y tensorrt
    sudo apt -y --fix-broken install
    sudo ldconfig
    sleep 5 

fi

# Install Docker
cd
echo $password | sudo -kS apt-get -y remove docker docker-engine docker.io containerd runc
echo $password | sudo -kS apt-get -y update
echo $password | sudo -kS apt install -y curl
echo $password | sudo -kS apt install -y openssh-server
echo $password | sudo -kS apt-get install -y \
apt-transport-https \
ca-certificates \
curl \
gnupg-agent \
software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
echo $password | sudo -kS apt-key fingerprint 0EBFCD8
echo $password | sudo -kS add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
echo $password | sudo -kS apt-get -y update
echo $password | sudo -kS apt-get install -y docker-ce docker-ce-cli containerd.io
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
echo $password | sudo -kS apt-get -y update && echo $password | sudo -kS apt-get install -y nvidia-container-toolkit
echo $password | sudo -kS systemctl restart docker
echo $password | sudo -kS apt-get install -y nvidia-container-runtime
echo $password | sudo -kS apt install -y libsqlite3-dev
echo $password | sudo -kS mkdir -p /etc/systemd/system/docker.service.d
echo $password | sudo -kS systemctl daemon-reload && echo $password | sudo -kS systemctl restart docker
cd
echo $password | sudo -kS apt -y update && echo $password | sudo -kS apt -y upgrade
echo $password | sudo -kS apt -y install curl
curl -sSL https://get.docker.com/ | sh
sudo tee /etc/docker/daemon.json <<EOF
    {
            "default-runtime" : "nvidia",
         "runtimes": {
                  "nvidia": {
                    "path": "nvidia-container-runtime",
                  "runtimeArgs": []
              }
     },
            "insecure-registries" : ["iwizregistry.iwizardsolutions.com:5000"]
    }
EOF
echo $password | sudo -kS systemctl daemon-reload && echo $password | sudo -kS systemctl restart docker

# Install Resolv Conf
echo $password | sudo -kS apt -y install resolvconf
sudo tee /etc/resolvconf/resolv.conf.d/head <<EOF 
nameserver 8.8.8.8 
nameserver 8.8.8.4 
EOF
sudo resolvconf --enable-updates
sudo resolvconf -u
cd

# Install Boost
cd
echo $password | sudo -kS apt install -y libboost-all-dev
# wget -O boost_1_71_0.tar.gz https://dl.bintray.com/boostorg/release/1.71.0/source/boost_1_71_0.tar.gz
# tar xvzf boost_1_71_0.tar.gz
# cd boost_1_71_0/
# echo $password | sudo -kS apt -y update
# echo $password | sudo -kS apt -y install build-essential g++ python-dev autotools-dev libicu-dev build-essential libbz2-dev libboost-all-dev ffmpeg
# ./bootstrap.sh --prefix=/usr/
# ./b2
# echo $password | sudo -kS -S ./b2 install

# Pull Docker Base Images
#echo $password | sudo -kS docker pull iwizregistry.iwizardsolutions.com:5000/dgpu_ds5_base:v3.0
echo $password | sudo -kS docker pull iwizregistry.iwizardsolutions.com:5000/iris_dgpu_3xxx_base_image:v3.0
echo $password | sudo -kS docker pull iwizregistry.iwizardsolutions.com:5000/iris_dgpu_3xxx_base_image:v4.0
#echo $password | sudo -kS docker pull iwizregistry.iwizardsolutions.com:5000/iris-alpr-service:1.1

# Install DS and its Dependencies
echo $password | sudo -kS apt -y install libjson-glib-dev
echo $password | sudo -kS apt -y install libcrypto++-dev libcrypto++-doc libcrypto++-utils openssh-server
echo $password | sudo -kS apt -y install libgstrtspserver-1.0-dev uuid uuid-dev libconfig++-dev libboost-all-dev libopencv-dev libcurl4-gnutls-dev gnutls-dev libsqlite3-dev glances git-lfs gdm3 vino nload net-tools 
echo $password | sudo -kS apt -y install gcc-7 g++-7
echo $password | sudo -kS update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 7
echo $password | sudo -kS update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 7
cd
wget https://irisretailstoragebox.blob.core.windows.net/irismodels/deepstream-5.1_5.1.0-1_amd64.deb
echo $password | sudo -kS dpkg -i deepstream-5.1_5.1.0-1_amd64.deb
echo $password | sudo -kS apt -y --fix-broken install
echo $password | sudo -kS apt -y install libgstrtspserver-1.0-dev ffmpeg
echo $password | sudo -kS apt -y install build-essential gcc make cmake cmake-gui cmake-curses-gui libssl-dev doxygen graphviz libcppunit-dev

# Install MQTT 
cd ~
git clone https://github.com/eclipse/paho.mqtt.c.git
cd paho.mqtt.c
git checkout v1.3.7
cmake -Bbuild -H. -DPAHO_WITH_SSL=ON -DPAHO_ENABLE_TESTING=OFF
echo $password | sudo -kS cmake --build build/ --target install
echo $password | sudo -kS ldconfig
cd ~
git clone https://github.com/eclipse/paho.mqtt.cpp
cd paho.mqtt.cpp
cmake -Bbuild -H. -DPAHO_BUILD_DOCUMENTATION=TRUE -DPAHO_BUILD_SAMPLES=TRUE
echo $password | sudo -kS cmake --build build/ --target install
echo $password | sudo -kS ldconfig
cd

# Log file
# exec 3>&1 4>&2
# trap 'exec 2>&4 1>&3' 0 1 2 3
# exec 1>iris_send_log.out 2>&1

# Install OPENVPN
# cd
# wget https://irisretailstoragebox.blob.core.windows.net/vpn-files/2.ovpn
# echo $password | sudo -kS apt -y install openvpn
# echo $password | sudo -kS cp 2.ovpn /etc/openvpn
# echo $password | sudo -kS cp /etc/openvpn/2.ovpn /etc/openvpn/clientvm.conf
cd /etc/openvpn
sudo sed -i '5,6 s/#/ /' update-resolv-conf
sudo sed -i '5 i script-security 2' update-resolv-conf
cd
# echo $password | sudo -kS systemctl enable openvpn@clientvm
# echo $password | sudo -kS systemctl restart openvpn@clientvm
# cd

# # Send Logs to ELK
# mkdir edge_info && cd edge_info
# wget https://irisretailstoragebox.blob.core.windows.net/irismodels/Dockerfile_edge
# mv Dockerfile_edge Dockerfile
# wget https://irisretailstoragebox.blob.core.windows.net/irismodels/edge_info.py
# ASSET_ID="$assetID"
# sed -i 's/asset_id/'${ASSET_ID}'/g' Dockerfile
# echo $password | sudo -kS docker build -t irisregistry.southindia.cloudapp.azure.com/edge_info:v1.0 .
# echo $password | sudo -kS docker create -it --net=host --restart=always --name edge_info irisregistry.southindia.cloudapp.azure.com/edge_info:v1.0
# echo $password | sudo -kS docker start edge_info

# Make LAN Driver Persistent
cd ~/Downloads
wget https://irisretailstoragebox.blob.core.windows.net/irismodels/r8125-9.004.01.tar.bz2
echo $password | sudo -kS chmod 0777 *
tar -xvf r8125-9.004.01.tar.bz2
cd r8125-9.004.01/
echo $password | sudo -kS ./autorun.sh
sudo tee /etc/modules-load.d/r8125.conf << EOF
r8125
EOF
sudo tee /etc/modules-load.d/r8169.conf << EOF
r8169
EOF
echo $password | sudo -kS modprobe r8169
echo $password | sudo -kS modprobe r8125

# Disable Auto Update
sudo tee /etc/apt/apt.conf.d/20auto-upgrades <<EOF
APT::Periodic::Unattended-Upgrade "0";
EOF

# Enable Auto Login
sudo sed -i 's/#  AutomaticLoginEnable = true/AutomaticLoginEnable = true/g' /etc/gdm3/custom.conf
sudo sed -i 's/#  AutomaticLogin = user1/AutomaticLogin = '${username}'/g' /etc/gdm3/custom.conf
sudo sed -i 's/#WaylandEnable=false/WaylandEnable=false/g' /etc/gdm3/custom.conf

# Enable Docker API
sudo tee /etc/systemd/system/docker.service.d/override.conf <<EOF
[Service]
ExecStart=
ExecStart=/usr/bin/dockerd --host=fd:// --add-runtime=nvidia=/usr/bin/nvidia-container-runtime -H fd:// -H tcp://0.0.0.0:2375
EOF

echo $password | sudo -kS apt install -y libopencv-dev

echo $password | sudo -kS apt install -y openssh-server gunicorn

cd ~ && wget https://irisretailstoragebox.blob.core.windows.net/irismodelss/alpr.zip
unzip alpr.zip
cd alpr-service
git checkout version/onnx
echo $password | sudo -kS apt install -y python3-pip git
pip3 install -r requirements.txt

cd ~ && wget https://irisretailstoragebox.blob.core.windows.net/irismodelss/yolo_engine_path.patch

cd ~ && wget https://irisretailstoragebox.blob.core.windows.net/irismodelss/build.patch

cd ~ && wget https://irisretailstoragebox.blob.core.windows.net/irismodelss/libnvinfer_plugin.so.7.2.2

echo $password | sudo -kS patch -R /opt/nvidia/deepstream/deepstream/sources/libs/nvdsinfer/nvdsinfer_model_builder.cpp yolo_engine_path.patch

echo $password | sudo -kS patch -R /opt/nvidia/deepstream/deepstream/sources/libs/nvdsinfer/Makefile build.patch

cd /opt/nvidia/deepstream/deepstream/sources/libs/nvdsinfer

echo $password | sudo -kS make install -j$(nproc) CUDA_VER=11.1

cd 

sudo ldconfig


cd ~ && echo $password | sudo -kS cp libnvinfer_plugin.so.7.2.2 /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.7.2.2

sudo ldconfig

# Details to Log
ifconfig
cat /etc/hosts
printf '\nFinished!\n'
printf '\nRebboting the machine in 60 secs\n'
sleep 5
# echo $password | sudo -kS systemctl restart anydesk.service
sleep 60
echo $password | sudo -kS reboot
