mkdir build
cd build
cmake .. -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DLLAMA_MPI=1
cmake --build . --target speculative --config Release
