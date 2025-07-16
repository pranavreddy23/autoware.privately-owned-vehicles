

#include "scene_seg.hpp"
#include <vector>
#include <stdexcept>
#include "rclcpp/rclcpp.hpp"

// Use the C API for wider compatibility with pre-built binaries
#include "onnxruntime_c_api.h"
#include <onnxruntime_cxx_api.h>


namespace autoware_pov::AutoSeg::SceneSeg
{

SceneSeg::SceneSeg(const std::string & model_path, const std::string & precision, int gpu_id)
: env_(ORT_LOGGING_LEVEL_WARNING, "scene_seg"), memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
  initializeOrtSession(model_path, precision, gpu_id);
  createColorMap();
}

void SceneSeg::initializeOrtSession(const std::string& model_path, const std::string& precision, int gpu_id)
{
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  // Set up execution providers using the C API for better compatibility
  if (precision == "cuda") {
     try {
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options_, gpu_id));
        RCLCPP_INFO(rclcpp::get_logger("scene_seg"), "Using CUDA Execution Provider on GPU %d.", gpu_id);
     } catch (const Ort::Exception& e) {
        RCLCPP_WARN(rclcpp::get_logger("scene_seg"), "CUDA Execution Provider is not available. Falling back to CPU. Error: %s", e.what());
     }
  } else {
    RCLCPP_INFO(rclcpp::get_logger("scene_seg"), "Using default CPU Execution Provider.");
  }

  ort_session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);

  Ort::AllocatorWithDefaultOptions allocator;
  input_names_.push_back(strdup(ort_session_->GetInputNameAllocated(0, allocator).get()));
  output_names_.push_back(strdup(ort_session_->GetOutputNameAllocated(0, allocator).get()));
  
  auto input_shape = ort_session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  model_input_height_ = static_cast<int>(input_shape[2]);
  model_input_width_ = static_cast<int>(input_shape[3]);
}

void SceneSeg::createColorMap()
{
  // Set all non-foreground classes to black to only highlight the foreground.
  color_map_.push_back(cv::Vec3b(0, 0, 0));       // BG: Black
  color_map_.push_back(cv::Vec3b(0, 0, 255));     // FG: Red
  color_map_.push_back(cv::Vec3b(0, 0, 0));       // Road: Black (was Blue)
}

void SceneSeg::preprocess(const cv::Mat & input_image, std::vector<float> & output_tensor, std::vector<int64_t>& input_dims)
{
    cv::Mat resized_image, float_image;
    cv::resize(input_image, resized_image, cv::Size(model_input_width_, model_input_height_));
    resized_image.convertTo(float_image, CV_32FC3, 1.0 / 255.0);

    cv::subtract(float_image, cv::Scalar(0.406, 0.456, 0.485), float_image);
    cv::divide(float_image, cv::Scalar(0.225, 0.224, 0.229), float_image);

    output_tensor.resize(model_input_height_ * model_input_width_ * 3);
    std::vector<cv::Mat> channels(3);
    cv::split(float_image, channels);
    // HWC to CHW
    memcpy(output_tensor.data(), channels[0].data, model_input_height_ * model_input_width_ * sizeof(float));
    memcpy(output_tensor.data() + model_input_height_ * model_input_width_, channels[1].data, model_input_height_ * model_input_width_ * sizeof(float));
    memcpy(output_tensor.data() + 2 * model_input_height_ * model_input_width_, channels[2].data, model_input_height_ * model_input_width_ * sizeof(float));
    
    input_dims = {1, 3, model_input_height_, model_input_width_};
}

bool SceneSeg::doInference(const cv::Mat & input_image)
{
    std::vector<float> preprocessed_data;
    std::vector<int64_t> input_dims;
    preprocess(input_image, preprocessed_data, input_dims);

    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, preprocessed_data.data(), preprocessed_data.size(),
        input_dims.data(), input_dims.size());

    try {
        last_output_tensors_ = ort_session_->Run(
            Ort::RunOptions{nullptr}, const_cast<const char* const*>(input_names_.data()), &input_tensor, 1,
            const_cast<const char* const*>(output_names_.data()), output_names_.size());
    } catch (const Ort::Exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("scene_seg"), "ONNX Runtime inference failed: %s", e.what());
        return false;
    }
    
    return true;
}

void SceneSeg::getRawMask(cv::Mat & raw_mask, const cv::Size & output_size) const
{
    if (last_output_tensors_.empty()) {
        throw std::runtime_error("Inference has not been run yet. Call doInference() first.");
    }
    const float* output_data = last_output_tensors_.front().GetTensorData<float>();
    auto output_dims = last_output_tensors_.front().GetTensorTypeAndShapeInfo().GetShape();
    const int num_classes = static_cast<int>(output_dims[1]);
    const int height = static_cast<int>(output_dims[2]);
    const int width = static_cast<int>(output_dims[3]);

    // Create argmax mask manually since cv::reduce doesn't work with 3D tensors
    cv::Mat argmax_mask(height, width, CV_8UC1);
    
    // Process each spatial location to find the class with maximum score
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            float max_score = -std::numeric_limits<float>::infinity();
            uint8_t best_class = 0;
            
            for (int c = 0; c < num_classes; ++c) {
                // NCHW format: output_data[c * height * width + h * width + w]
                float score = output_data[c * height * width + h * width + w];
                if (score > max_score) {
                    max_score = score;
                    best_class = static_cast<uint8_t>(c);
                }
            }
            
            argmax_mask.at<uint8_t>(h, w) = best_class;
        }
    }
    
    cv::resize(argmax_mask, raw_mask, output_size, 0, 0, cv::INTER_NEAREST);
}

void SceneSeg::colorizeMask(const cv::Mat & raw_mask, const cv::Mat & original_image, cv::Mat & blended_image) const
{
    cv::Mat color_mask(raw_mask.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    
    // Simple and fast colorization - only color foreground pixels
    for (int y = 0; y < raw_mask.rows; ++y) {
        const uint8_t* mask_row = raw_mask.ptr<uint8_t>(y);
        cv::Vec3b* color_row = color_mask.ptr<cv::Vec3b>(y);
        
        for (int x = 0; x < raw_mask.cols; ++x) {
            uint8_t class_id = mask_row[x];
            if (class_id == 1) {  // Only color foreground class
                color_row[x] = cv::Vec3b(0, 0, 255);  // Red for foreground
            }
            // All other classes remain black (already initialized to 0,0,0)
        }
    }

    // Alpha-blend the color mask with the original image
    cv::addWeighted(color_mask, 0.5, original_image, 0.5, 0.0, blended_image);
}

}  // namespace autoware_pov::AutoSeg::SceneSeg 