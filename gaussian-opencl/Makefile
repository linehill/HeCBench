include ../../common/make.config

SRC = clutils.cpp gaussianElim.cpp utils.cpp

EXE = main

$(EXE): $(SRC)
	$(CXX) $(CXX_FLAGS) $(KERNEL_DIM) -fopenmp $(SRC) -o $(EXE) $(OCL_CPPFLAGS) $(OCL_LDFLAGS)

clean: 
	rm -f $(EXE) 
