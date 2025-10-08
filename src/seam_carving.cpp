#include "seam_carving.h"

cv::Mat seamCarving(const cv::Mat& image, const cv::Size& out_size) {
  (void)out_size;
  return image;
}

std::vector<std::vector<float>> calculateEnergy(const cv::Mat& image) {
    int height = image.rows;
    int width = image.cols;
    std::vector<std::vector<float>> energy(height, std::vector<float>(width, 0.0f));

    for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int xLeft  = std::max(x - 1, 0);
			int xRight = std::min(x + 1, width - 1);
			int yUp    = std::max(y - 1, 0);
			int yDown  = std::min(y + 1, height - 1);

			cv::Vec3f left  = image.at<cv::Vec3f>(y, xLeft);
			cv::Vec3f right = image.at<cv::Vec3f>(y, xRight);
			cv::Vec3f up    = image.at<cv::Vec3f>(yUp, x);
			cv::Vec3f down  = image.at<cv::Vec3f>(yDown, x);

			float dx = cv::norm(right - left);
			float dy = cv::norm(down - up);

			energy[y][x] = dx + dy;
      	}
    }

    return energy;
}

void findVerticalSeam(const cv::Mat& image){
	std::vector<std::vector<float>> energy = calculateEnergy(image);

	int height = image.rows;
	int width = image.cols;

	std::vector<std::vector<float>> dp(height, std::vector<float>(width, 0.0f));

	dp[0] = energy[0];

	for(int y = 1; y < height; ++y){
		for(int x = 0; x < width; ++x){
			float min_energy = dp[y-1][x];
			if(x > 0){
				min_energy = std::min(min_energy, dp[y-1][x-1]);
			}
			if(x < width - 1){
				min_energy = std::min(min_energy, dp[y-1][x+1]);
			}
			dp[y][x] = energy[y][x] + min_energy;
		}
	}

	// Backtrack to find the seam
}

//TODO: seam finding functions (horizontal and vertical)
//TODO: seam removing functions (horizontal and vertical)
//TODO: 