# Install

JAX-AM supports Linux and macOS. Create a new conda environment and run

```bash
git clone https://github.com/tianjuxue/jax-am.git
cd jax-am
pip install .
```

>**Note**: JAX-AM depends on [petsc4py](https://www.mcs.anl.gov/petsc/petsc4py-current/docs/usrman/index.html). We have found difficulty installing `petsc4py` with `pip` on certain platforms. Installing `petsc4py` with `conda` is therefore recommended (see [here](https://anaconda.org/conda-forge/petsc4py)).


Several remarks:

* If you are not familiar with conda, check [how set up a conda environment](conda).


* JAX-AM depends on JAX. If you want to use GPU, you need to [install the GPU version of JAX](https://github.com/google/jax#installation) properly. Before that, make sure [CUDA](cuda) and [cuDNN](cudnn) are properly configured on your machine.


* If you want to use the phase-field package of JAX-AM, [Neper](neper) is required for polycrystal generation.

(conda)=
## Conda

[Conda](https://docs.conda.io/en/latest/?ref=learnubuntu.com) is an open source package management system and environment management system.The installer is offered as a shell script on [here](https://repo.anaconda.com/archive/). Find and download the version that matches your system and CPU architecture. Then make the script you download executable.

```shell
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
chmod -v +x Anaconda3-2022.05-Linux-x86_64.sh
```

Once done, execute the script to run the installer for Anaconda.

```shell
./Anaconda3-2023.03-1-Linux-x86_64.sh
```

Then you will see that the installation program has an EULA, and you need to press "Enter" to agree. This will install Anaconda in the default location, which is .`/~/anaconda3`. Finally, to complete the installation of Anaconda, update the PATH variable. Open your `.bashrc` file using a text editor (e.g. Vim, Nano, VSCode, etc.), add the following three lines, and save the file.

```
if ! [[ $PATH =~ "$HOME/anaconda3/bin" ]]; then
    PATH="$HOME/anaconda3/bin:$PATH"
fi
```

Restart the terminal and run the following command to check whether Anaconda has been successfully installed and set in PATH.

```shell
conda list
```

Anaconda is successfully installed if you get a list of packages. To create and open an environment, run


```shell
conda create -n myenv python=3.9
conda activate myenv
```

(cuda)=
## CUDA 

[Compute Unified Device Architecture (CUDA)](https://developer.nvidia.com/cuda-toolkit) is a parallel computing platform and application programming interface developed by Nvidia, which enables software developers to perform general-purpose computing using GPUs that support CUDA software. Before using JAX, you need to install CUDA first.

In general, the process of installing and using CUDA under Linux is as follows:

1. Preparation for installation of CUDA Toolkit
2. Install the CUDA Toolkit.

To verify that your GPU is CUDA-capable, run

```shell
lspci | grep -i nvidia
```

You will get a prompt similar to the following if your GPU supports CUDA programming.

```shell
3b:00.0 VGA compatible controller: NVIDIA Corporation Device 2230 (rev a1)
3b:00.1 Audio device: NVIDIA Corporation Device 1aef (rev a1)
```

Make sure that `gcc`, `g++` and `make` are installed on your system. If you want to run CUDA examples, you need to install the appropriate dependency libraries.

```shell
sudo apt update # update apt
sudo apt install gcc g++ make # install gcc, g++, make
sudo apt install libglu1-mesa libxi-dev libxmu-dev libglu1-mesa-dev freeglut3-dev # install dependency libraries
```

Select the installation method from the CUDA toolbox that matches your system version and architecture [download page](https://developer.nvidia.com/cuda-downloads?target_os=Linux), download and run the runfile. 

```shell
# Download CUDA runfile
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run 
# Install CUDA Toolkit
sudo sh cuda_12.1.1_530.30.02_linux.run
```

Note: You can get information about your system's architecture, version, etc. by run

```shell
uname -m && cat /etc/*release
```

After the CUDA toolkit has been installed, the screen will output the following message:

```shell
Driver:   Installed
Toolkit:  Installed in /usr/local/cuda-10.1/
Samples:  Installed in /home/abneryepku/

Please make sure that
 -   PATH includes /usr/local/cuda-10.1/
 -   LD_LIBRARY_PATH includes /usr/local/cuda-10.1/lib64, or, add /usr/local/cuda-10.1/lib64 to /etc/ld.so.conf and run ldconfig as root
```

Write  the following text in the `~/.bashrc` file to complete the CUDA configuration.

```shell
export PATH=$PATH:/usr/local/cuda-10.1/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.1/lib64:/usr/lib/x86_64-linux-gnu
```

You can check if CUDA is successfully installed by run

```shell
nvcc --version
```

If the installation is successful, you will receive a message similar to the following

```shell
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Feb__7_19:32:13_PST_2023
Cuda compilation tools, release 12.1, V12.1.66
Build cuda_12.1.r12.1/compiler.32415258_0
```

(cudnn)=
## cuDNN

[The NVIDIA CUDA Deep Neural Network library (cuDNN)](https://developer.nvidia.com/cudnn) is a GPU-accelerated library of primitives for deep neural networks. You need to install it after installing CUDA. Select the cuDNN version that matches your CUDA version from the [download page](https://developer.nvidia.com/rdp/cudnn-archive).

The installation process is actually copying the cuDNN header files into the CUDA header directory. Unzip the downloaded cuDNN, and then execute the following command

```shell
# Copy cuDNN files
sudo cp cuda/include/* /usr/local/cuda-12.0/include/
sudo cp cuda/lib64/* /usr/local/cuda-12.0/lib64/
# Add executable permissions
sudo chmod +x /usr/local/cuda-12.0/include/cudnn.h
sudo chmod +x /usr/local/cuda-12.0/lib64/libcudnn*
```

You can run the following command to check if cuDNN has been installed successfully.

```shell
cat /usr/local/cuda-12.0/include/cudnn.h | grep CUDNN_MAJOR -A 2
```

(neper)=
## Neper

[Neper](https://neper.info/) is a free / open source software package for polycrystal generation and meshing. It can be used to generate polycrystals with a wide variety of morphological properties. Install the following components before downloading Neper.

```shell
sudo apt install cmake
sudo apt-get install libnlopt.dev
sudo apt-get install povray
sudo apt-get install libgmsh-dev
sudo apt-get install imagemagick
sudo apt-get install libgsl0-dev
```

The latest version is available on the `main` branch of the GitHub repository. To get the latest version as a Git user, run:

```shell
git clone https://github.com/neperfepx/neper.git
```

Enter the `neper/src/` directory, create a folder named `build`, and execute the compilation.

```shell
mkdir build && cd buid && cmake ..
make -j
sudo make install
```

You can perform testing by running the following commands to check if Neper has been successfully installed.

```shell
neper -T -n 10 -o test
neper -V  test.tess -datacellcol id -print img_test
```

If Neper has been successfully installed, you should be able to find two files named `test.tess` and `img_test.png` in the current directory where you executed the above commands.
