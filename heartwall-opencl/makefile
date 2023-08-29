include ../../common/make.config

ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

EXE = main

$(EXE):	./main.o \
			./kernel/kernel_gpu_opencl_wrapper.o \
			./util/avi/avilib.o \
			./util/avi/avimod.o \
			./util/file/file.o \
			./util/timer/timer.o \
			./util/opencl/opencl.o
	$(CC) 		./main.o \
			./kernel/kernel_gpu_opencl_wrapper.o \
			./util/avi/avilib.o \
			./util/avi/avimod.o \
			./util/file/file.o \
			./util/timer/timer.o \
			./util/opencl/opencl.o \
			-lm $(OCL_LDFLAGS) -o $(EXE)

# ======================================================================================================================================================150
#	MAIN FUNCTION
# ======================================================================================================================================================150

./main.o:	./main.h \
			./main.c
	$(CC) $(CC_FLAGS) $(OUTPUT) $(KERNEL_DIM) ./main.c \
			-c \
			-o ./main.o

# ======================================================================================================================================================150
#	KERNELS
# ======================================================================================================================================================150

./kernel/kernel_gpu_opencl_wrapper.o:	./kernel/kernel_gpu_opencl_wrapper.h \
										./kernel/kernel_gpu_opencl_wrapper.c
	$(CC) $(CC_FLAGS) $(KERNEL_DIM)	./kernel/kernel_gpu_opencl_wrapper.c \
			-c \
			-o ./kernel/kernel_gpu_opencl_wrapper.o \
			$(OCL_CPPFLAGS)

# ======================================================================================================================================================150
#	UTILITIES
# ======================================================================================================================================================150

./util/avi/avilib.o:./util/avi/avilib.h \
					./util/avi/avilib.c
	$(CC) $(CC_FLAGS)	./util/avi/avilib.c \
			-c \
			-o ./util/avi/avilib.o

./util/avi/avimod.o:./util/avi/avilib.h \
					./util/avi/avimod.c
	$(CC) $(CC_FLAGS)	./util/avi/avimod.c \
			-c \
			-o ./util/avi/avimod.o

./util/file/file.o:	./util/file/file.h \
					./util/file/file.c
	$(CC) $(CC_FLAGS)	./util/file/file.c \
			-c \
			-o ./util/file/file.o

./util/timer/timer.o:	./util/timer/timer.h \
						./util/timer/timer.c
	$(CC) $(CC_FLAGS)	./util/timer/timer.c \
			-c \
			-o ./util/timer/timer.o

./util/opencl/opencl.o:	./util/opencl/opencl.h \
						./util/opencl/opencl.c
	$(CC) $(CC_FLAGS)	./util/opencl/opencl.c \
			-c \
			-o ./util/opencl/opencl.o \
			$(OCL_CPPFLAGS)

clean:
	rm -f *.o $(EXE) \
		./kernel/*.o \
		./util/avi/*.o \
		./util/file/*.o \
		./util/timer/*.o \
		./util/opencl/*.o
