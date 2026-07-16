void TRTLogger::log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kINFO) {
        std::cout << "[TensorRT] " << msg << std::endl;
    }
}

