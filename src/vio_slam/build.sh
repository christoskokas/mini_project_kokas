echo "Downloading and building ceres-solver.2-1-0 ..."

mkdir packages
cd packages

git clone --branch 2.1.0 --depth 1 https://github.com/ceres-solver/ceres-solver.git

sudo apt-get install cmake

sudo apt-get install libgoogle-glog-dev libgflags-dev

sudo apt-get install libatlas-base-dev

sudo apt-get install libeigen3-dev

sudo apt-get install libsuitesparse-dev

cd ceres-solver
mkdir build
cd build
cmake ..
make -j3
make install

cd ../../

echo "Downloading and building Pangolin ..."

git clone --recursive --branch v0.8 --depth 1 https://github.com/stevenlovegrove/Pangolin.git

cd Pangolin

./scripts/install_prerequisites.sh recommended

cmake -B build
cmake --build build

cd ../

echo "Downloading and building yaml-cpp ..."

git clone --recursive --branch yaml-cpp-0.7.0 --depth 1 https://github.com/jbeder/yaml-cpp

cd yaml-cpp
mkdir build
cd build
cmake ..
make -j3
make install

cd ../../

sudo apt install libopencv-dev python3-opencv

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
