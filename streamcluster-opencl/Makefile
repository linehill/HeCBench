include ../../common/make.config

SRC = streamcluster.cpp
EXE = streamcluster

$(EXE):$(SRC)
	$(CXX) $(CXX_FLAGS)  $(SRC) -o $(EXE) $(OCL_CPPFLAGS) $(OCL_LDFLAGS)

errmsg:$(SRC)
	$(CXX) $(CXX_FLAGS) $(SRC) -o $(EXE) $(OCL_CPPFLAGS) $(OCL_LDFLAGS) -D ERRMSG

ptx:$(SRC)
	$(CXX) $(CXX_FLAGS) $(SRC) -o $(EXE) $(OCL_CPPFLAGS) $(OCL_LDFLAGS) -D PTX_MSG

res:$(SRC)
	$(CXX) $(CXX_FLAGS) $(SRC) -o $(EXE) $(OCL_CPPFLAGS) $(OCL_LDFLAGS) -D RES_MSG

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result*
