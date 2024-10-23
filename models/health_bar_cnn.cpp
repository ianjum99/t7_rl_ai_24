#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include "health_bar_cnn.h"

// Inference function to predict health percentage
float predict_health(cv::Mat image) {
    torch::jit::script::Module module;
    try {
        // Load the CNN model (TorchScript model)
        module = torch::jit::load("models/saved/health_bar_cnn.pt");
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        return -1;
    }

    // Convert OpenCV image to tensor
    torch::Tensor input_tensor = torch::from_blob(image.data, {1, image.rows, image.cols, 3}, torch::kByte);
    input_tensor = input_tensor.permute({0, 3, 1, 2});  // Convert to [N, C, H, W]
    
    // Forward pass
    torch::Tensor output = module.forward({input_tensor}).toTensor();
    
    return output.item<float>() * 100.0;  // Return health percentage
}

