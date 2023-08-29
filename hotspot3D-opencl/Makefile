include ../../common/make.config

OUTPUT = *.out

EXE = main

$(EXE): 3D.c CL_helper.c
	$(CC) $^ $(CCFLAGS) $(OCL_CPPFLAGS) -o $@ $(OCL_LDFLAGS) -lm

clean:
	rm -f 3D $(OUTPUT)
