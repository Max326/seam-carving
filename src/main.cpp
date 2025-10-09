#include <opencv2/highgui.hpp>
#include <iostream>

#include "seam_carving.h"

int main(int, char** argv) {
    const auto in = cv::imread(argv[1]);

	if (in.empty()) {
		std::cout << "Error: Could not load image." << std::endl;
		return -1;
	}

    auto out = seamCarving(in, cv::Size(500, 330));
  
  	cv::imwrite("./output.png", out);
  	return 0;
}
