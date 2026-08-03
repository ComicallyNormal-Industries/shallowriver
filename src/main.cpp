#include "runner.hpp"
#include "intrinsics.hpp"
#include <iostream>
#include <string>    

// Updated to accept the camera_mode
int run_inference(int operating_mode, int camera_mode){
    runner model_runner;

    int exit_code = model_runner.run(operating_mode, camera_mode);

    std::cout << "shallowriver shutting down with code: " << exit_code << std::endl;
    return exit_code;
}

void display_help(){
    std::cerr << "Valid options:\n";
    std::cerr << "  --calibrate --c [1|2]      : run camera intrinsics calibration on camera 1 or 2\n";
    std::cerr << "  --run       --r [1|2|both] : run inference on camera 1, 2, or both\n"; 
    std::cerr << "  --engine    --e [1|2|both] : recompile .engine files and run inference\n";
    std::cerr << "  --generate  --g          : run charuco board generator\n";
    std::cerr << "  --help      --h          : display help\n";
}

int main(int argc, char** argv) {
    std::cout << "Starting shallowriver on ARM Jetson..." << std::endl;

    // Check for at least one command line argument
    if (argc < 2) {
        std::cerr << "No arguments detected! Please run camera intrinsics or inference\n";
        display_help();
        return -1;
    }

    std::string mode = argv[1];
    std::string cam_arg = (argc > 2) ? argv[2] : "1"; 

    if (mode == "--calibrate" || mode == "--c") {
        int cam_id = (cam_arg == "2") ? 2 : 1; // Strict fallback to 1 if anything other than "2" is typed
        
        std::cout << "Routing to camera calibration tool for Camera " << cam_id << "..." << std::endl;
        
        // (You will need to update run_calibration() to accept this integer)
        return intrinsics::run_calibration(cam_id);
    }
    else if (mode == "--run" || mode == "--r" || mode == "--engine" || mode == "--e") {
        
        int operating_mode = (mode == "--engine" || mode == "--e") ? 2 : 1;
        int cam_mode = 1; // Default
        
        if (cam_arg == "2") {
            cam_mode = 2;
        } else if (cam_arg == "both") {
            cam_mode = 3;
        }

        std::cout << "Running inference on camera(s): " << (cam_mode == 3 ? "both" : std::to_string(cam_mode)) << " ..." << std::endl;
        return run_inference(operating_mode, cam_mode);
    }
    else if (mode == "--generate" || mode == "--g") {
        std::cout << "Running charuco board generator..." << std::endl;
        return intrinsics::generate_charuco();
    }
    else if (mode == "--help" || mode == "--h") {
        display_help();
        return 0;
    }
    else {
        std::cerr << "Unknown argument: " << mode << "\n";
        display_help();
        return -1;
    }
    
    return 0;
}