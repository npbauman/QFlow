
git submodule init
git submodule update
cd NWQ-Sim
# git checkout dev/qflow
git checkout main
git submodule init
git submodule update
CURRDIR=$(pwd)
cd vqe/nlopt
mkdir build; cd build
cmake ..; make -j4
# cd ../../..
cd $CURRDIR
mkdir build; cd build
cmake .. -DCMAKE_BUILD_TYPE=Release; make -j4