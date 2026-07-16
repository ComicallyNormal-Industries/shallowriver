#include "glogger.hpp"
#include <iostream> 

void TRTLogger::log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept {
    
    // Only print errors and warnings to avoid spamming your console
    if (severity <= nvinfer1::ILogger::Severity::kWARNING) {
        std::cout << "[TRT] " << msg << std::endl;
    }
}

