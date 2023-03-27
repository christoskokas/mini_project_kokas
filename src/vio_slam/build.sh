echo "building Thirdparty/ceres-solver.2-1-0 ..."

sudo apt-get install cmake

sudo apt-get install libgoogle-glog-dev libgflags-dev

sudo apt-get install libatlas-base-dev

sudo apt-get install libeigen3-dev

sudo apt-get install libsuitesparse-dev

cd Thirdparty/ceres-solver-2.1.0
mkdir build
cd build
cmake ..
make -j3
make install

cd ../../../

sudo apt install libopencv-dev python3-opencv

sudo apt-get install python3-catkin-tools

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
