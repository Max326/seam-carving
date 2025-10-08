#include <opencv2/highgui.hpp>

#include "seam_carving.h"

int main(int, char** argv) {
    const auto in = cv::imread(argv[1]);
    auto out = seamCarving(in, cv::Size(500, 250));
  
  	cv::imwrite("./output.png", out);
  	return 0;
}
