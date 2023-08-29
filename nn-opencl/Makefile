include ../../common/make.config

SRC = nearestNeighbor.cpp clutils.cpp utils.cpp

EXE = main

$(EXE):
	$(CXX) $(CXX_FLAGS) -fopenmp $(SRC) -o $(EXE) $(OCL_CPPFLAGS) $(OCL_LDFLAGS)

clean:
	rm -f $(EXE)
