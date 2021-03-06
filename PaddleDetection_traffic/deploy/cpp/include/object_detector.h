//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <utility>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "paddle_inference_api.h" // NOLINT

#include "include/preprocess_op.h"
#include "include/config_parser.h"


namespace PaddleDetection {
// Object Detection Result
struct ObjectResult {
  // Rectangle coordinates of detected object: left, right, top, down
  std::vector<int> rect;
  // Class id of detected object
  int class_id;
  // Confidence of detected object
  float confidence;
};


// Generate visualization colormap for each class
std::vector<int> GenerateColorMap(int num_class);


// Visualiztion Detection Result
cv::Mat VisualizeResult(const cv::Mat& img,
                     const std::vector<ObjectResult>& results,
                     const std::vector<std::string>& lable_list,
                     const std::vector<int>& colormap);


class ObjectDetector {
 public:
  explicit ObjectDetector(const std::string& model_dir, bool use_gpu = false) {
    config_.load_config(model_dir);
    threshold_ = config_.draw_threshold_;
    preprocessor_.Init(config_.preprocess_info_, config_.arch_);
    LoadModel(model_dir, use_gpu);
  }

  // Load Paddle inference model
  void LoadModel(
    const std::string& model_dir,
    bool use_gpu);

  // Run predictor
  void Predict(
      const cv::Mat& img,
      std::vector<ObjectResult>* result);

  // Get Model Label list
  const std::vector<std::string>& GetLabelList() const {
    return config_.label_list_;
  }

 private:
  // Preprocess image and copy data to input buffer
  void Preprocess(const cv::Mat& image_mat);
  // Postprocess result
  void Postprocess(
      const cv::Mat& raw_mat,
      std::vector<ObjectResult>* result);

  std::unique_ptr<paddle::PaddlePredictor> predictor_;
  Preprocessor preprocessor_;
  ImageBlob inputs_;
  std::vector<float> output_data_;
  float threshold_;
  ConfigPaser config_;
};

}  // namespace PaddleDetection
