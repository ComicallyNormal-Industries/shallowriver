#include "runner.hpp"
#include "intrinsics.hpp"
#include <iostream>
#include <string>
#include <vector>

// start the pipline runner
int run_inference(int operating_mode, int camera_mode, bool enable_logging, bool enable_print){
    runner model_runner;

    int exit_code = model_runner.run(operating_mode, camera_mode, enable_logging, enable_print);

    std::cout << "shallowriver shutting down with code: " << exit_code << std::endl;
    return exit_code;
}

void display_help(){
    std::cerr << "Valid options:\n";
    std::cerr << "  --calibrate --c [1|2]      : run camera intrinsics calibration on camera 1 or 2\n";
    std::cerr << "  --run       --r [1|2|both] : run inference on camera 1, 2, or both\n";
    std::cerr << "  --engine    --e [1|2|both] : recompile .engine files and run inference\n";
    std::cerr << "  --generate  --g          : run charuco board generator\n";
    std::cerr << "  --log       --l          : enable per-frame fps/latency CSV logging (with --run/--engine)\n";
    std::cerr << "  --print     --p          : print pipeline benchmark stats to the terminal (with --run/--engine)\n";
    std::cerr << "  --help      --h          : display help\n";
}

// parse command line arguments
int main(int argc, char** argv) {
    std::cout << "Starting shallowriver on ARM Jetson..." << std::endl;

    // Pull --log/--l and --print/--p out from wherever they appear on the command line,
    // so either can be combined with any mode (e.g. "--run both --log --print") without
    // disturbing the positional mode/camera arguments below.
    bool enable_logging = false;
    bool enable_print = false;
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--log" || arg == "--l") {
            enable_logging = true;
        } else if (arg == "--print" || arg == "--p") {
            enable_print = true;
        } else {
            args.push_back(arg);
        }
    }

    // Check for at least one command line argument
    if (args.empty()) {
        std::cerr << "No arguments detected! Please run camera intrinsics or inference\n";
        display_help();
        return -1;
    }

    std::string mode = args[0];
    std::string cam_arg = (args.size() > 1) ? args[1] : "1";

    if (mode == "--calibrate" || mode == "--c") {
        int cam_id = (cam_arg == "2") ? 2 : 1;
        
        std::cout << "Routing to camera calibration tool for Camera " << cam_id << "..." << std::endl;
        
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
        if (enable_logging) {
            std::cout << "Per-frame fps/latency logging enabled (--log)." << std::endl;
        }
        if (enable_print) {
            std::cout << "Pipeline benchmark stats printing enabled (--print)." << std::endl;
        }
        return run_inference(operating_mode, cam_mode, enable_logging, enable_print);
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