#pragma once
#include <string>

// Resolves where shallowriver reads/writes its runtime files. Both functions
// prefer a local "res/" directory (dev builds: the post-build copy step in
// CMakeLists.txt populates <build>/res). Outside of that, asset_dir() falls back to
// the read-only location baked in at install time (SHALLOWRIVER_INSTALL_ASSET_DIR,
// e.g. /usr/share/shallowriver/res) and data_dir() falls back to a writable
// per-user directory, since calibration/engine/log files can't live under
// /usr/share. Both can be overridden with SHALLOWRIVER_ASSET_DIR /
// SHALLOWRIVER_DATA_DIR for custom deployments.
namespace paths {

    std::string asset_dir();

    std::string data_dir();

}
