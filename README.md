# PyTorchTOP

## Download LibTorch

From [https://pytorch.org/](https://pytorch.org/) download, 1.7.1 (stable), Windows, LibTorch, C++/Java, CUDA 10.1.

## CUDA and cuDNN

From NVIDIA, install CUDA 10.1, which will create `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1`. Download cuDNN 7.6.5 for 10.1 and place the files into this folder too.

## [CMake 3.15.1 or higher](https://cmake.org/download/)

The following steps rely on CUDA and cuDNN being in your system path. If you have multiple versions of CUDA installed, you can temporarily modify your path to make sure the right one is on top. For example, since we're using CUDA 10.1, in a command window

    set CUDA_HOME=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1
    set CUDA_PATH=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1
    set PATH=%CUDA_HOME%;%CUDA_PATH%;%PATH%

This changes your system path but only in this command window. With the same window, inside the root of `PyTorchTOP` create a build folder:

    mkdir build
    cd build
    cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..

where `/path/to/libtorch` should be the full path to the unzipped LibTorch distribution. Expected output:

	-- Selecting Windows SDK version 10.0.18362.0 to target Windows 10.0.
	x64 architecture in use
	-- Caffe2: CUDA detected: 10.1
	-- Caffe2: CUDA nvcc is: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/bin/nvcc.exe
	-- Caffe2: CUDA toolkit directory: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1
	-- Caffe2: Header version is: 10.1
	-- Found cuDNN: v7.6.5  (include: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/include, library: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/lib/x64/cudnn.lib)
	-- Autodetected CUDA architecture(s):  7.5
	-- Added CUDA NVCC flags for: -gencode;arch=compute_75,code=sm_75
	-- Configuring done
	-- Generating done
	-- Build files have been written to: /path/to/PyTorchTOP/build
If it works, you should end up with a Visual Studio solution inside `build`. Open `PyTorchTOP.sln`, select the Release build and press F5 to build the DLL and launch TouchDesigner. When you build, the newly built plugin and the necessary LibTorch DLLs will be copied to your C++ TouchDesigner plugin folder at `C:\Users\admin\Documents\Derivative\Plugins`, replacing `admin` with your user name. If this folder doesn't already exist, make sure it does.

## Debugging

The steps to build a debug-mode Visual Studio solution are similar. Instead of `build`, make a folder `build_debug`.

    mkdir build_debug
    cd build_debug
    set DEBUG=1
    cmake -DCMAKE_PREFIX_PATH=/path/to/debug/libtorch ..

You should download the debug version from PyTorch and use its path. Now you can build `build_debug\PyTorchTOP.sln` in Debug mode. You can manually copy the `.pdb` files from the LibTorch folder to the `Plugins` folder in order to help with stack traces during debugging.

## Background Matte

As an example, this project uses models that have been exported from [Background Matting V2](https://github.com/PeterL1n/BackgroundMattingV2). Follow their links to download their "TorchScript" models and place them in this repo's `models` folder. More information on exporting models in this format is available [here](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html).

## Extra Notes

This example project has been tested with TouchDesigner 2020.28110 and libtorch with CUDA 10.1. We have no control over what CUDA version TouchDesigner uses, so it's important to pick the libtorch version that matches. If you're unsure which version to use, check TouchDesigner's [Release Notes](https://docs.derivative.ca/Release_Notes).
