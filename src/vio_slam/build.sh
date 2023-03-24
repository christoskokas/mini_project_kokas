echo "building Thirdparty/ceres-solver.2-1-0 ..."

cd Thirdparty/ceres-solver-2.1.0
mkdir build
cd build
cmake ../ceres-solver-2.1.0
make -j3
make install

cd ../../../

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
