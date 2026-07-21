#include "runner.hpp"
#include "intrinsics.hpp"
#include <iostream>
#include <string>    

int run_inference(){
    //Instantiate your runner class
    runner model_runner;

    //Run inference runner, Return the integer exit code
    int exit_code = model_runner.run();

    std::cout << "shallowriver shutting down with code: " << exit_code << std::endl;
 	return exit_code;
}

void display_help(){
	std::cerr << "Valid options: --calibrate\n";
	std::cerr << "--calibrate --c: run camera intrinsics calibration " << "\n";
	std::cerr << "--run --r: run inference " << "\n"; 
	std::cerr << "--help --h: display help " << "\n";
}

int main(int argc, char** argv) {
    std::cout << "Starting shallowriver on ARM Jetson..." << std::endl;

    //Check for command line arguments
    if (argc > 1) {
        std::string mode = argv[1];
        
        if (mode == "--calibrate" || mode == "--c") {
            std::cout << "Routing to camera calibration tool..." << std::endl;
            return intrinsics::run_calibration();
        }
		else if (mode == "--run" || mode == "--r"){
			std::cout << "Running inference ..." << std::endl;
			return run_inference();
		}
		else if (mode == "--help" || mode == "--h"){
			display_help();
			return 0;
		}
		else if (mode == "--generate" || mode == "--g") {
            std::cout << "Running charuco board generator..." << std::endl;
            return intrinsics::generate_charuco();
        }
        else {
			std::cerr << "Unknown argument: " << mode << "\n";
			display_help();
			return -1;
        }
    }
	else {
		std::cerr << "No arguments detected! Please run camera intrinsics or inference" << "\n";
		display_help();
		return -1;
	}
    return 0;
}
