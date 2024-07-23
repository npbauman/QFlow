cd NWQ-Sim
CURRDIR=$(pwd)
git submodule init
git submodule update
cd vqe/nlopt
mkdir build; cd build
cmake ..; make -j4
# cd ../../..
cd $CURRDIR
mkdir build; cd build
cmake .. -DCMAKE_BUILD_TYPE=Release; make -j4