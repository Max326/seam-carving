#pragma once

#include <opencv2/core/mat.hpp>
#include <vector>

cv::Mat seamCarving(const cv::Mat& image, const cv::Size& out_size);

std::vector<std::vector<float>> calculateEnergy(const cv::Mat& image);