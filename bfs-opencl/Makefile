include ../../common/make.config

SRC = bfs.cpp timer.cc

EXE = main

$(EXE): $(SRC)
	$(CXX) $(CXX_FLAGS) -fopenmp $(SRC) -o $(EXE) $(OCL_CPPFLAGS) $(OCL_LDFLAGS)

profile:$(SRC)
	$(CXX) $(CXX_FLAGS) -fopenmp $(SRC) -o $(EXE) $(OCL_CPPFLAGS) $(OCL_LDFLAGS) -D PROFILING

release:$(SRC)
	$(CXX) $(CXX_FLAGS) -fopenmp $(SRC) -o $(EXE) $(OCL_CPPFLAGS) $(OCL_LDFLAGS)

errmsg:$(SRC)
	$(CXX) $(CXX_FLAGS) -fopenmp $(SRC) -o $(EXE) $(OCL_CPPFLAGS) $(OCL_LDFLAGS) -D ERRMSG PTX_MSG

ptx:$(SRC)
	$(CXX) $(CXX_FLAGS) -fopenmp $(SRC) -o $(EXE) $(OCL_CPPFLAGS) $(OCL_LDFLAGS) -D PTX_MSG

res:$(SRC)
	$(CXX) $(CXX_FLAGS) -fopenmp $(SRC) -o $(EXE) $(OCL_CPPFLAGS) $(OCL_LDFLAGS) -D RES_MSG

debug: $(SRC)
	$(CXX) $(CXX_FLAGS) $(SRC) -o $(EXE) $(OCL_CPPFLAGS) $(OCL_LDFLAGS)

run:
	./$(EXE)

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result*
