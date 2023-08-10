include ../../common/make.config

SRC = clutils.cpp gaussianElim.cpp utils.cpp

EXE = gaussian

$(EXE):
	$(CXX) $(CXX_FLAGS) $(KERNEL_DIM) -fopenmp $(SRC) -o $(EXE) $(OCL_CPPFLAGS) $(OCL_LDFLAGS)

clean: 
	rm -f $(EXE) 
