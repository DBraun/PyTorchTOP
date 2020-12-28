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

	torch::Tensor forward(torch::Tensor rgba_foreground, torch::Tensor rgba_background) {
		// narrow RGBA to RGB, permute to (batch, channels, rows, cols), and convert to data type that the traced model takes.
		auto rgb_foreground = rgba_foreground.narrow(3, 0, 3).permute({ 0, 3, 1, 2 }).to(myModelInputDtype);
		auto rgb_background = rgba_background.narrow(3, 0, 3).permute({ 0, 3, 1, 2 }).to(myModelInputDtype);

		if (rgba_foreground.dtype() == torch::kByte && myModelInputDtype != torch::kByte) {
			rgb_foreground /= 255.;
			rgb_background /= 255.;
		}

		auto outputs = myModule.forward({ rgb_foreground, rgb_background }).toTuple()->elements();
		auto pha = outputs[0].toTensor();

		// need to permute to (batch, rows, cols, channels)
		auto channels_last = pha.permute({ 0, 2, 3, 1 });
		
		return channels_last;
	}
};