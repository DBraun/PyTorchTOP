#pragma once
#include <torch/torch.h>
#include <torch/script.h>

// Define a new Module.
struct WrapperModel : torch::nn::Module {

	torch::Dtype myModelInputDtype;
	torch::jit::script::Module myModule;
	
	WrapperModel(torch::Dtype dtype, torch::jit::script::Module module) {
		myModelInputDtype = dtype;
		
		myModule = module;
	}

	torch::Tensor forward(torch::Tensor rgba_foreground) {
		// narrow RGBA to RGB, permute to (batch, channels, rows, cols), and convert to data type that the traced model takes.
		auto rgb_foreground = rgba_foreground.narrow(3, 0, 3).permute({ 0, 3, 1, 2 }).to(myModelInputDtype);

		if (rgba_foreground.dtype() == torch::kByte && myModelInputDtype != torch::kByte) {
			rgb_foreground /= 255.;
		}

		// NB: style transfer model outputs [0-255] scale for some reason?
		// so divide by 255.
		auto rgb = myModule.forward({ rgb_foreground }).toTensor() / 255.;

		// need to permute to (batch, rows, cols, channels)
		auto channels_last = rgb.permute({ 0, 2, 3, 1 });

		// add alpha channel
		auto sizes = channels_last.sizes();

		//std::cout << "sizes " << sizes << std::endl;

		const torch::TensorOptions tensorOptions = torch::TensorOptions(torch::kCUDA).dtype(channels_last.dtype());
		auto alpha = torch::ones({ sizes[0], sizes[1], sizes[2], 1 }, tensorOptions);

		auto rgba_out = torch::cat({ channels_last, alpha }, -1);
	
		return rgba_out;
	}
};