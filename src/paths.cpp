#include "paths.hpp"
#include <cstdlib>
#include <filesystem>

namespace fs = std::filesystem;

#ifndef SHALLOWRIVER_INSTALL_ASSET_DIR
#define SHALLOWRIVER_INSTALL_ASSET_DIR "/usr/share/shallowriver/res"
#endif

namespace {

std::string xdg_data_home() {
    if (const char* xdg = std::getenv("XDG_DATA_HOME"); xdg && *xdg) {
        return xdg;
    }
    const char* home = std::getenv("HOME");
    return (home ? std::string(home) : std::string(".")) + "/.local/share";
}

}

namespace paths {

std::string asset_dir() {
    if (const char* override_dir = std::getenv("SHALLOWRIVER_ASSET_DIR")) {
        return override_dir;
    }
    if (fs::is_directory("res")) {
        return "res";
    }
    return SHALLOWRIVER_INSTALL_ASSET_DIR;
}

std::string data_dir() {
    if (const char* override_dir = std::getenv("SHALLOWRIVER_DATA_DIR")) {
        return override_dir;
    }
    if (fs::is_directory("res")) {
        return "res";
    }
    fs::path dir = fs::path(xdg_data_home()) / "shallowriver";
    fs::create_directories(dir);
    return dir.string();
}

}
