include ../../common/make.config

SRC = euler3d.cpp

EXE = euler3d

$(EXE):$(SRC)
	$(CXX) $(CXX_FLAGS) $(KERNEL_DIM) $(SRC) -o $(EXE) $(OCL_CPPFLAGS) $(OCL_LDFLAGS) $(FLAGS)

release:$(SRC)
	$(CXX) $(CXX_FLAGS) $(KERNEL_DIM) $(SRC) -o $(EXE) $(OCL_CPPFLAGS) $(OCL_LDFLAGS) $(FLAGS)

errmsg:$(SRC)
	$(CXX) $(CXX_FLAGS) $(KERNEL_DIM) $(SRC) -o $(EXE) $(OCL_CPPFLAGS) $(OCL_LDFLAGS) -D ERRMSG $(FLAGS)

ptx:$(SRC)
	$(CXX) $(CXX_FLAGS) $(KERNEL_DIM) $(SRC) -o $(EXE) $(OCL_CPPFLAGS) $(OCL_LDFLAGS) -D PTX_MSG $(FLAGS)

res:$(SRC)
	$(CXX) $(CXX_FLAGS) $(KERNEL_DIM) $(SRC) -o $(EXE) $(OCL_CPPFLAGS) $(OCL_LDFLAGS) -D RES_MSG $(FLAGS)

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result*
