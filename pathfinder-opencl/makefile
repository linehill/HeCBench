include ../../common/make.config

# link objects(binaries) together
pathfinder: main.o OpenCL.o
	$(CXX) $(CC_FLAGS) main.o OpenCL.o $(OCL_LDFLAGS) -o pathfinder

main.o: main.cpp
	$(CXX) $(CXX_FLAGS) $(OCL_CPPFLAGS) -c main.cpp -o main.o

OpenCL.o: OpenCL.cpp kernels.cl
	$(CXX) $(CXX_FLAGS) $(OCL_CPPFLAGS) -c OpenCL.cpp -o OpenCL.o

clean:
	clear
	@rm -f pathfinder *.o *.out *~ *\#
