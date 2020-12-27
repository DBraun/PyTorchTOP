/* Shared Use License: This file is owned by Derivative Inc. (Derivative)
* and can only be used, and/or modified for use, in conjunction with
* Derivative's TouchDesigner software, and only if you are a licensee who has
* accepted Derivative's TouchDesigner license or assignment agreement
* (which also govern the use of this file). You may share or redistribute
* a modified version of this file provided the following conditions are met:
*
* 1. The shared file or redistribution must retain the information set out
* above and this list of conditions.
* 2. Derivative's name (Derivative Inc.) or its trademarks may not be used
* to endorse or promote products derived from this file without specific
* prior written permission from Derivative.
*/

#include "TOP_CPlusPlusBase.h"
#include <torch/torch.h>
#include "WrapperModel.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudafeatures2d.hpp>

using namespace std::chrono;

class PyTorchTOP : public TOP_CPlusPlusBase
{
public:
	PyTorchTOP(const OP_NodeInfo *info, TOP_Context *context);
	virtual ~PyTorchTOP();

	virtual void		getGeneralInfo(TOP_GeneralInfo*, const OP_Inputs*, void* reserved1) override;
	virtual bool		getOutputFormat(TOP_OutputFormat*, const OP_Inputs*, void* reserved) override;


	virtual void		execute(TOP_OutputFormatSpecs*,
								const OP_Inputs*,
								TOP_Context *context,
								void* reserved) override;


	virtual int32_t		getNumInfoCHOPChans(void* reserved) override;
	virtual void		getInfoCHOPChan(int32_t index,
										OP_InfoCHOPChan *chan,
										void* reserved) override;

	virtual bool		getInfoDATSize(OP_InfoDATSize *infoSize, void* reserved) override;
	virtual void		getInfoDATEntries(int32_t index,
											int32_t nEntries,
											OP_InfoDATEntries *entries,
											void* reserved) override;

    virtual void		getErrorString(OP_String *error, void* reserved) override;

	virtual void		setupParameters(OP_ParameterManager *manager, void* reserved) override;
	virtual void		pulsePressed(const char *name, void* reserved) override;

private:
	// We don't need to store this pointer, but we do for the example.
	// The OP_NodeInfo class store information about the node that's using
	// this instance of the class (like its name).
	const OP_NodeInfo*	myNodeInfo;

	// In this example this value will be incremented each time the execute()
	// function is called, then passes back to the TOP 
	int32_t				myExecuteCount;

	// basic vars
	std::stringstream myError;
	milliseconds myDuration;
	std::string myLoadedModelFilePath;
	const std::string myEmptyString = "";

	// torch vars
	torch::Device myDevice = torch::kCUDA;
	torch::Tensor myInputTensorForeground;
	torch::Tensor myInputTensorBackground;
	torch::jit::script::Module myModule;
	std::shared_ptr<WrapperModel> myMainModel;
	torch::Dtype myIntermediateDtype;

	int myInputWidth = 1280;
	int myInputHeight = 720;
	int myInputNumChannels = 1;
	size_t myBytesPerInputChannel = 0; // initializing it to zero forces allocateGPU_textures to run once at startup.

	int myOutputWidth = 1280;
	int myOutputHeight = 720;
	int myOutputNumChannels = 1;
	size_t myBytesPerOutputChannel = 4;

	bool checkModelFile(const char* newModelFilePath);

	void setModelParameters(const OP_Inputs* inputs);

	bool checkInputTOP(const OP_TOPInput* inputTOP, int desiredWidth, int desiredHeight);

	bool prepareEverything(const OP_Inputs* inputs);
	bool allocateGPU_textures();

	void executeWithHomography(TOP_OutputFormatSpecs* outputFormat, const OP_Inputs* inputs);
	void executeWithoutHomography(TOP_OutputFormatSpecs* outputFormat, const OP_Inputs* inputs);

	int myRefineMode = 0; // TODO: make this an enum.
	int myBackboneScale = 0; // TODO: make this an enum.
	int myRefineSamplePixels = 80000;

	cv::cuda::GpuMat myGpuWarpedOutput;
	cv::cuda::GpuMat myGpuForegroundInput;
	cv::cuda::GpuMat myGpuBackgroundInput;

	cv::cuda::GpuMat myGpuForegroundInputGray;
	cv::cuda::GpuMat myGpuBackgroundInputGray;
	
	cv::cuda::GpuMat myKeypointsForeground_gpu;
	cv::cuda::GpuMat myKeypointsBackground_gpu;

	std::vector<cv::KeyPoint> myKeypointsForeground;
	std::vector<cv::KeyPoint> myKeypointsBackground;

	cv::cuda::GpuMat myDescriptorsForeground;
	cv::cuda::GpuMat myDescriptorsBackground;

	std::vector<cv::DMatch > myMatches;

	cv::Ptr<cv::cuda::DescriptorMatcher> myMatcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);  // NORM_HAMMING is for ORB.
};
