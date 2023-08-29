include ../../common/make.config

EXE = main

$(EXE): hotspot.c hotspot.h OpenCL_helper_library.c OpenCL_helper_library.h
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -o $(EXE) OpenCL_helper_library.c hotspot.c $(OCL_CPPFLAGS) $(OCL_LDFLAGS)

clean:
	rm -f $(EXE) output.out
