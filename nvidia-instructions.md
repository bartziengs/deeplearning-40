# Evaluation
## Installation instructions
### Get CUDA installed
This is a bit tricky and will depend on your situation. The following two guides are a useful starting point.

* [Community CUDA installation guide](https://www.pugetsystems.com/labs/hpc/How-To-Install-CUDA-10-1-on-Ubuntu-19-04-1405/)
* [Official CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

The [Nvidia proprietary drivers ppa](https://launchpad.net/~graphics-drivers/+archive/ubuntu/ppa) provides details about the available drivers.
The community guide recommends to install these dependencies. Not sure if needed.
```bash
sudo apt-get install -y freeglut3 freeglut3-dev libxi-dev libxmu-dev
```
The guide installs the driver before CUDA, but this results in a warning from the CUDA install script.
Download the latest [CUDA driver](http://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64) (local runfile).
Then do what it says on the download page:
```bash
sudo sh cuda_10.***_linux.run (use just downloaded newest version)
```
While the script says it will install a driver, this did not happen for me, so I installed it manually. (TODO: add ppa-add command)
```bash
sudo apt-get install -y nvidia-driver-440  # Get newest version I guess
sudo shutdown -r Now  # reboot to load driver(s)
```

Try if everything is working by running
```bash
nvidia-smi
```
The CUDA version in the top left of the table should be the same as the one you installed.

### Nvidia-docker

We found it to be easiest to run [caffe](https://caffe.berkeleyvision.org/) in a docker container.

In order to run docker images with CUDA gpu support, you need [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).
Go ahead and install it for your distro. If you do not have docker itself, install it using [these instructions](https://docs.docker.com/engine/install/).

To successfully run a docker container with GPU passthrough, be sure to supply the `--gpu` argument, as explained
in the [usage section](https://github.com/NVIDIA/nvidia-docker#usage) of nvidia-docker.

### Caffe container

There are two flavours of the [docker container](https://github.com/BVLC/caffe/tree/master/docker),
 cpu only and gpu support. The evaluate script seems to _require_ gpu support, as the `--gpu_ids -1` trick results in an error.

Go ahead and run
```bash
docker pull bvlc/caffe:gpu
```
to acquire the container image. Of course, this will be done automatically when a command is run with this container name.
