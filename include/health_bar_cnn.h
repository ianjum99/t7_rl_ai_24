#ifndef HEALTH_BAR_CNN_H
#define HEALTH_BAR_CNN_H

#include <opencv2/opencv.hpp>

// Function to predict health percentage using the CNN model
float predict_health(cv::Mat image);

#endif  // HEALTH_BAR_CNN_H

