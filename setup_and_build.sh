#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "==> Updating package lists..."
apt update

echo "==> Installing NVIDIA JetPack SDK..."
apt install -y nvidia-jetpack

echo "==> Installing CMake..."
apt install -y cmake

echo "==> Installing Eigen3 development headers..."
apt install -y libeigen3-dev

echo "==> Setting up build directory..."
mkdir -p build
cd build

echo "==> Configuring project with CMake..."
cmake ..

echo "==> Compiling project with Make..."
make -j$(nproc)

echo "==> Build process completed successfully!"