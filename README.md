DLF: Deep Learning Foundation Library
==

DLFL is a high performance, high reliable software library aimed at deep learning applications.

**Low-Level:**

Provide a mechanism to perform efficient operations on low-level data structures such as vector, 
matrix and tensor. Most of these operations are related to Deep Learning algorithms such as convolution, 
matrix multiplication, activation functions and more. The algorithm make the best use of system 
resources to gain extremely high performance. These include CPU/GPU parallel computation and NPU
acceleration. The resources are managed by the library and transparent to end user.

**Middle-Level:**

Manipulate various neural network models and translate them into an efficient Intermediate 
Representation (IR). The IR can be executed Just In Time (JIT) or compiled into target machine 
code. We have a uniform platform to run models on various targets (CPU, GPU, NPU).

**High-Level:**

Provide a convenient Application Programming Interface (API). The API can be used by end user 
to develop AI applications in business domain. The API has integrated some common application 
scenarios such as face detection and recognition. The API can be extended to meet the special
requirements of user.

How to build this package?
--

DLF is a C++ template library. You need docker to be installed on your system to build the project.
Please refers to http://www.docker.com for more information.

You can also install nvidia-docker to use CUDA and/or OpenCL to accelerate DNN computation.
The installation instructions can be found [here](https://github.com/nvidia/nvidia-docker/wiki/Installation-%28version-2.0%29).

That's all the prerequisites to building the project. After you have installed and configured
docker properly, cd to the project root and run "make".