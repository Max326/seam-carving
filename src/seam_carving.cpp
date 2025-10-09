#include "seam_carving.h"

std::vector<std::vector<float>> calculateEnergy(const cv::Mat& image) {
    int height = image.rows;
    int width = image.cols;
    std::vector<std::vector<float>> energy(height, std::vector<float>(width, 0.0f));

    for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			energy[y][x] = calculatePixelsEnergy(image, x, y);
      	}
    }

    return energy;
}

float calculatePixelsEnergy(const cv::Mat& image, const int& x, const int& y) {
	int height = image.rows;
	int width = image.cols;

	int xLeft  = std::max(x - 1, 0);
	int xRight = std::min(x + 1, width - 1);
	int yUp    = std::max(y - 1, 0);
	int yDown  = std::min(y + 1, height - 1);

	cv::Vec3b left  = image.at<cv::Vec3b>(y, xLeft);
	cv::Vec3b right = image.at<cv::Vec3b>(y, xRight);
	cv::Vec3b up    = image.at<cv::Vec3b>(yUp, x);
	cv::Vec3b down  = image.at<cv::Vec3b>(yDown, x);

	float dx = cv::norm(cv::Vec3f(right) - cv::Vec3f(left));
	float dy = cv::norm(cv::Vec3f(down) - cv::Vec3f(up));

	return dx + dy;
}

std::vector<std::vector<float>> updateEnergy(const cv::Mat& image, const std::vector<int>& seam, bool vertical, const std::vector<std::vector<float>>& oldEnergy) {
	const int height = image.rows;
	const int width = image.cols;
	
	if (vertical) {
		std::vector<std::vector<float>> energy(height, std::vector<float>(width-1));
		for (int y = 0; y < height; ++y) {
			const int sx = seam[y];

			for (int x = 0; x < width - 1; ++x) {
				const int oldx = (x < sx) ? x : x + 1;
				energy[y][x] = oldEnergy[y][oldx];
			}
			if (sx - 1 >= 0)
                energy[y][sx - 1] = calculatePixelsEnergy(image, sx - 1, y);
            if (sx < width - 1)
                energy[y][sx] = calculatePixelsEnergy(image, sx, y);
		}
		return energy;
	} else {
		std::vector<std::vector<float>> energy(height-1, std::vector<float>(width));
		for (int x = 0; x < width; ++x) {
			const int sy = seam[x];

			for (int y = 0; y < height - 1; ++y) {
				const int oldy = (y < sy) ? y : y + 1;
				energy[y][x] = oldEnergy[oldy][x];
			}
			if (sy - 1 >= 0)
                energy[sy - 1][x] = calculatePixelsEnergy(image, x, sy - 1);
            if (sy < height - 1)
                energy[sy][x] = calculatePixelsEnergy(image, x, sy);
        }
        return energy;
	}	
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

	// Backtracking
	std::vector<int> seam(height);
	seam.back() = std::min_element(dp.back().begin(), dp.back().end()) - dp.back().begin();

	for(int y = height - 2; y >= 0; --y){
		int prev_x = seam[y + 1];
		float min_val = dp[y][prev_x];
		int best_x = prev_x;

		if (prev_x > 0 && dp[y][prev_x - 1] < min_val) {
			min_val = dp[y][prev_x - 1];
			best_x = prev_x - 1;
		} 
		if (prev_x < width - 1 && dp[y][prev_x + 1] < min_val) {
			min_val = dp[y][prev_x + 1];
			best_x = prev_x + 1;
		}

		seam[y] = best_x;
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

	// Backtracking
	std::vector<int> seam(width);
	seam.back() = std::min_element(dp.begin(), dp.end(), 
		[](const std::vector<float>& a, const std::vector<float>& b) {
			return a.back() < b.back();
		}) - dp.begin();

	for (int x = width - 2; x >= 0; --x) {
		int prev_y = seam[x + 1];
		float min_val = dp[prev_y][x];
		int best_y = prev_y;

		if (prev_y > 0 && dp[prev_y - 1][x] < min_val) {
			min_val = dp[prev_y - 1][x];
			best_y = prev_y - 1;
		} else if (prev_y < height - 1 && dp[prev_y + 1][x] < min_val) {
			min_val = dp[prev_y + 1][x];
			best_y = prev_y + 1;
		}

		seam[x] = best_y;
	}

	return seam;
}

static float seamCost(const std::vector<int>& seam, const std::vector<std::vector<float>>& energy, bool vertical) {
	float totalCost = 0.0f;
	if (vertical){
		for (size_t y = 0; y < seam.size(); ++y) {
			totalCost += energy[y][seam[y]];
		}
	} else {
		for (size_t x = 0; x < seam.size(); ++x) {
			totalCost += energy[seam[x]][x];
		}
	}
	
	return totalCost;
}

cv::Mat removePixels(const cv::Mat& image, const std::vector<int>& seam, bool vertical) {
	int height = image.rows;
	int width = image.cols;
	
	if (vertical) width--; else height--;
	cv::Mat newImage(cv::Size(width, height), image.type());

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
	std::vector<std::vector<float>> energy = calculateEnergy(newImage);

	// while (newImage.size() != out_size) {
	while (newImage.cols > out_size.width || newImage.rows > out_size.height) {
		bool removeVertical = (newImage.cols > out_size.width);
		bool removeHorizontal = (newImage.rows > out_size.height);

		std::vector<int> vSeam, hSeam;
		
		if (removeVertical && !removeHorizontal) {
			vSeam = findVerticalSeam(newImage, energy);
			newImage = removePixels(newImage, vSeam, true);
			energy = updateEnergy(newImage, vSeam, true, energy);
			continue;
		}

		if (!removeVertical && removeHorizontal) {
			hSeam = findHorizontalSeam(newImage, energy);
			newImage = removePixels(newImage, hSeam, false);
			energy = updateEnergy(newImage, hSeam, false, energy);
			continue;
		}

		vSeam = findVerticalSeam(newImage, energy);
		hSeam = findHorizontalSeam(newImage, energy);

		float vCost = seamCost(vSeam, energy, true);
		float hCost = seamCost(hSeam, energy, false);

		float vAvg = vCost / static_cast<float>(energy.size());
        float hAvg = hCost / static_cast<float>(energy[0].size());

        // require 5% advantage to switch directions
        const float switch_thresh = 1.05f;

        // If nearly tied, prefer the axis with more pixels left to remove
        int vLeft = newImage.cols - out_size.width;
        int hLeft = newImage.rows - out_size.height;

        bool takeV = (vAvg * switch_thresh < hAvg) || (std::fabs(vAvg - hAvg) <= 1e-6f && vLeft >= hLeft);

        if (takeV) {
			newImage = removePixels(newImage, vSeam, true);
			energy = updateEnergy(newImage, vSeam, true, energy);
		}
        else {
			newImage = removePixels(newImage, hSeam, false);
			energy = updateEnergy(newImage, hSeam, false, energy);
		}       

		// energy = calculateEnergy(newImage);
		// energy = updateEnergy(newImage, vSeam, true, energy);

		// std::vector<int> hSeam = findVerticalSeam(out, energy);	
	}

 	return newImage;
}

//TODO: seam removing functions (horizontal and vertical)
//TODO: loop to remove seams until desired size is reached + update energy
//TODO: seam direction choice (horizontal vs vertical)

//TODO: sanitize inputs
//TODO: run with command line arguments