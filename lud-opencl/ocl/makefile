include ../../../common/make.config

EXE = lud

$(EXE): lud.cpp
	$(CXX) $(CXX_FLAGS) $(KERNEL_DIM) $(OCL_CPPFLAGS) -o $(EXE) lud.cpp ../common/common.c $(OCL_LDFLAGS)

clean:
	rm -f $(EXE)
