#pragma once

#include <opencv2/core/mat.hpp>
#include <vector>

cv::Mat seamCarving(const cv::Mat& image, const cv::Size& out_size);

std::vector<std::vector<float>> calculateEnergy(const cv::Mat& image);
float calculatePixelsEnergy(const cv::Mat& image, const int& x, const int& y);
std::vector<std::vector<float>> updateEnergy(const cv::Mat& image, const std::vector<int>& seam, bool vertical, const std::vector<std::vector<float>>& oldEnergy);

std::vector<int> findVerticalSeam(const cv::Mat& image, std::vector<std::vector<float>>& energy);
std::vector<int> findHorizontalSeam(const cv::Mat& image, std::vector<std::vector<float>>& energy);

cv::Mat removePixels(const cv::Mat& image, const std::vector<int>& seam, bool vertical);