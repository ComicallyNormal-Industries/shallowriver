#include "runner.hpp"
#include <iostream>

int main(int argc, char** argv) {
    std::cout << "Starting shallowriver on ARM Jetson..." << std::endl;

    // 1. Instantiate your runner class
    runner model_runner;

    // 2. Run your setup to initialize the camera, pointers, and TensorRT


    // 3. Start the main execution loop!
    // We return the integer it passes back so the OS knows if it crashed or closed cleanly.
    int exit_code = model_runner.run();

    std::cout << "shallowriver shutting down with code: " << exit_code << std::endl;
    return exit_code;
}
