# PyTorchTOP

This project demonstrates how to use OpenCV with CUDA modules and PyTorch/LibTorch in a TouchDesigner Custom Operator. Building this project is moderately complex and time-consuming, so read all the instructions to get a general overview before getting started.

## Background Matting Project

PyTorchTOP uses [Background Matting v2](https://github.com/PeterL1n/BackgroundMattingV2) as a foundation because their project is MIT-licensed (thank you!). Their research paper focuses on the speed and quality of the matting operation. For their quantitative results, they preprocess the images with OpenCV and don't add this computational cost to their numbers. It is *ok* to not preprocess the images but only if the frozen background image does not need to be transformed to match the live foreground images. This optional transformation is called [homography](https://docs.opencv.org/master/d7/dff/tutorial_feature_homography.html) in OpenCV. To see how the researchers do it, look at their [HomographicAlignment](https://github.com/PeterL1n/BackgroundMattingV2/blob/97e2df124d0fa96eb7f101961a2eb806cdd25049/inference_utils.py#L6) class. PyTorchTOP has a toggle for using CUDA contrib modules from OpenCV to do the same homography efficiently on the GPU.

PyTorchTOP uses TorchScript models that can be [downloaded](https://github.com/PeterL1n/BackgroundMattingV2#download) from [Background Matting V2](https://github.com/PeterL1n/BackgroundMattingV2). Download their "TorchScript" models and place them in this repo's `models` folder. More information on exporting models in this format is available [here](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html).

## Python

Install Python 3.7 or higher and make sure it's in your system path. This will help later when building OpenCV.

## Download LibTorch

From [https://pytorch.org/](https://pytorch.org/) download, 1.7.1 (stable), Windows, LibTorch, C++/Java, CUDA 11.0

## CUDA and cuDNN

From NVIDIA, install CUDA which will create `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0`. Check that this creates a system environment variable: `CUDA_TOOLKIT_ROOT_DIR` holding `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0`. Download the [cuDNN](https://developer.nvidia.com/cudnn) version that corresponds to the CUDA version and place the files in the CUDA folder too.

## OpenCV and the Contrib modules.

Use git to clone [OpenCV](https://github.com/opencv/opencv) and [OpenCV Contrib](https://github.com/opencv/opencv_contrib). It would be good to place them next to each other. Use [CMake-GUI](https://cmake.org/download/) to select the source folder of OpenCV. In CMake-GUI, for the box `Where is the source code`, select the path to OpenCV. For `Where to build the binaries`, one suggestion is to create a `build` folder inside the OpenCV repo. You should permanently edit your system environment variables so that `OpenCV_DIR` holds this path. In CMake-GUI, press `Configure`. In the pop-up, select the latest version of Visual Studio, possibly Visual Studio 16 2019, and `Finish`. It will take less than a minute, and a table with lots of options will appear. Make the following modifications:

* Enable the checkbox `BUILD_opencv_world`.
* For `OPENCV_EXTRA_MODULES_PATH`, enter the path to `modules` folder in the opencv_contrib repository which you cloned earlier.
* Enable the checkbox `WITH_CUDA`.

Click `Generate`.

Check the entries in the table again. You should see a new entry for `CUDA_TOOLKIT_ROOT_DIR`, holding `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0`. If you don't, you may add the entry yourself, but this isn't guaranteed to work. You should probably check your CUDA installation, delete the CMake-GUI cache and restart the CMake-GUI steps.

If `CUDA_TOOLKIT_ROOT_DIR` is valid, then click `Open Project`. Build in `Release` mode. This will take between 1 to 2 hours. When it's complete, look for the "world" dll (`opencv_world451.dll` or equivalent) in the build's `bin\Release` folder and move it to `%USERPROFILE%\Documents\Derivative\Plugins`. If this folder doesn't already exist, create it.

## Building the PyTorchTOP Project.

To get started, we must first make sure several environment variables are correct. The process of installing CUDA and cuDNN should have already put them in your system path. If you have multiple versions of CUDA installed, you can temporarily modify your path to make sure the right one is found first. For example, since we're using CUDA 11.0, in a command window, do the following:

    set CUDA_HOME=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0
    set CUDA_PATH=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0
    set PATH=%CUDA_HOME%;%CUDA_PATH%;%PATH%

This changes your system path but only in this command window. Furthermore, we need an environment variable `OpenCV_DIR` to be set to the path where OpenCV was built, such as `C:/opencv/build`. If you haven't set this variable, you can do it temporarily with the right path.

	set OpenCV_DIR=C:/opencv/build

 With the same window, inside the root of `PyTorchTOP` create a build folder:

    mkdir build
    cd build
    cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..

where `/path/to/libtorch` should be the full path to the unzipped LibTorch distribution. Expected output:

	-- Selecting Windows SDK version 10.0.18362.0 to target Windows 10.0.
	x64 architecture in use
	-- Caffe2: CUDA detected: 11.0
	-- Caffe2: CUDA nvcc is: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/bin/nvcc.exe
	-- Caffe2: CUDA toolkit directory: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0
	-- Caffe2: Header version is: 11.0
	-- Found cuDNN: v8.0.5  (include: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/include, library: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/lib/x64/cudnn.lib)
	-- Autodetected CUDA architecture(s):  7.5
	-- Added CUDA NVCC flags for: -gencode;arch=compute_75,code=sm_75
	-- Configuring done
	-- Generating done
	-- Build files have been written to: /path/to/PyTorchTOP/build

If it works, you should end up with a Visual Studio solution `build/PyTorchTOP.sln`. Open it, select the Release build and press F5 to build the DLL and launch TouchDesigner. When you build, the newly built plugin and the necessary LibTorch DLLs will be copied to your C++ TouchDesigner plugin folder at `%USERPROFILE%\Documents\Derivative\Plugins`.

## Debugging

The steps to build a debug-mode Visual Studio solution are similar. You should download the debug version from PyTorch and use its path for CMake. Instead of `build`, make a folder `build_debug`.

    mkdir build_debug
    cd build_debug
    set DEBUG=1
    cmake -DCMAKE_PREFIX_PATH=/path/to/debug/libtorch ..

 Now you can build `build_debug\PyTorchTOP.sln` in Debug mode. You can manually copy the `.pdb` files from the LibTorch folder to the `Plugins` folder in order to help with stack traces during debugging. You may also consider duplicating the steps for building OpenCV in debug mode.

## Using the TouchDesigner project

When the project opens, check the Model path parameter on the plugin. If you select a 32-bit model, then the custom parameter for Bytes Per Model Input should be 4 because 4 bytes is 32 bits. If the model is 16-bit, then select 2. Then toggle the "Unload Plugin" parameter to load the plugin. The first input to the plugin is the live video and the second input is the static background image.

## Extra Notes

This example project has been tested with TouchDesigner 2020.28110 and libtorch with CUDA 11.0. At the time of writing this, TouchDesigner is using 10.1, so it's risky to use 11.0. Luckily it works. Choose your versions of the various components in this project at your own discretion. TouchDesigner's [Release Notes](https://docs.derivative.ca/Release_Notes) often mention changes to their CUDA version.

## Style Transfer Branch

A [branch](https://github.com/DBraun/PyTorchTOP/tree/model/style-transfer) of this project demonstrates Style Transfer. You should use git to checkout the branch. Then follow the [steps](https://github.com/DBraun/PyTorchTOP-cpumem#neural-style-transfer) in the old PyTorchTOP repository to save "traced_model.pt" to the models folder. For example, you should save "rain_princess_1280x720.pt". Build the `PyTorchTOP.dll`, open `PyTorchTOP.toe` and toggle the button that had unloaded the C++ TOP. You should get about 11 FPS at 1280x720 on an RTX 2080.