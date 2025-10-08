#include "seam_carving.h"

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

			cv::Vec3b left  = image.at<cv::Vec3b>(y, xLeft);
			cv::Vec3b right = image.at<cv::Vec3b>(y, xRight);
			cv::Vec3b up    = image.at<cv::Vec3b>(yUp, x);
			cv::Vec3b down  = image.at<cv::Vec3b>(yDown, x);

			float dx = cv::norm(right - left);
			float dy = cv::norm(down - up);

			energy[y][x] = dx + dy;
      	}
    }

    return energy;
}

std::vector<int> findVerticalSeam(const cv::Mat& image, std::vector<std::vector<float>>& energy) {
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
	std::vector<int> seam(height);
	seam.back() = std::min_element(dp.back().begin(), dp.back().end()) - dp.back().begin();

	for(int y = height - 2; y >= 0; --y){
		int prev_x = seam[y + 1];
		int start = std::max(prev_x - 1, 0);
		int end = std::min(prev_x + 1, width - 1);
		seam[y] = std::min_element(dp[y].begin() + start, dp[y].begin() + end + 1) - dp[y].begin();
	}

	return seam;
}

std::vector<int> findHorizontalSeam(const cv::Mat& image, std::vector<std::vector<float>>& energy) {
	int height = image.rows;
	int width = image.cols;

	std::vector<std::vector<float>> dp(height, std::vector<float>(width, 0.0f));

	for (int i = 0; i < height; ++i) {
		dp[i][0] = energy[i][0];
	}

	for (int x = 1; x < width; ++x) {
		for (int y = 0; y < height; ++y) {
			float min_energy = dp[y][x - 1];
			if (y > 0) {
				min_energy = std::min(min_energy, dp[y - 1][x - 1]);
			}
			if (y < height - 1) {
				min_energy = std::min(min_energy, dp[y + 1][x - 1]);
			}
			dp[y][x] = energy[y][x] + min_energy;
		}
	}

	// Backtrack to find the seam
	std::vector<int> seam(width);
	seam.back() = std::min_element(dp.back().begin(), dp.back().end()) - dp.back().begin();

	for(int x = width - 2; x >= 0; --x){
		int prev_y = seam[x + 1];
		int start = std::max(prev_y - 1, 0);
		int end = std::min(prev_y + 1, height - 1);
		seam[x] = std::min_element(dp[x].begin() + start, dp[x].begin() + end + 1) - dp[x].begin(); //! fix
	}

	return seam;
}

cv::Mat removePixels(const cv::Mat& image, const std::vector<int>& seam, bool vertical) {
	int height = image.rows;
	int width = image.cols;
	
	if (vertical) width--; else height--;
	cv::Mat newImage(height, width, image.type());

	if (vertical) {
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				newImage.at<cv::Vec3b>(y, x) = (x < seam[y]) ? image.at<cv::Vec3b>(y, x) : image.at<cv::Vec3b>(y, x + 1);
			}
		}
	} else {
		for (int x = 0; x < width; ++x) {
			for (int y = 0; y < height; ++y) {
				newImage.at<cv::Vec3b>(y, x) = (y < seam[x]) ? image.at<cv::Vec3b>(y, x) : image.at<cv::Vec3b>(y+1, x);
			}
		}
	}
	return newImage;
}

cv::Mat seamCarving(const cv::Mat& image, const cv::Size& out_size) {
	cv::Mat newImage = image.clone();
	std::vector<std::vector<float>> energy = calculateEnergy(image);

	// while (newImage.size() != out_size) {
	while (newImage.cols > out_size.width) {
		std::vector<int> vSeam = findVerticalSeam(newImage, energy);
		newImage = removePixels(newImage, vSeam, true);
		energy = calculateEnergy(newImage);

		// std::vector<int> hSeam = findVerticalSeam(out, energy);	
	}

 	return newImage;
}

//TODO: seam removing functions (horizontal and vertical)
//TODO: loop to remove seams until desired size is reached + update energy
//TODO: seam direction choice (horizontal vs vertical)

//TODO: sanitize inputs