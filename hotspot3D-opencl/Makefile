include ../../common/make.config

OUTPUT = *.out

3D: 3D.c CL_helper.c
	$(CC) $^ $(CCFLAGS) $(OCL_CPPFLAGS) -o $@ $(OCL_LDFLAGS) -lm

clean:
	rm -f 3D $(OUTPUT) 
