include ../../common/make.config

ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

OMP_LIB = -lgomp
OMP_FLAG = -fopenmp

EXE = main

# link objects together
$(EXE):		main.o \
			./kernel/kernel_gpu_opencl_wrapper.o \
			./util/num/num.o \
			./util/timer/timer.o \
			./util/opencl/opencl.o
	$(CC)	main.o \
			./kernel/kernel_gpu_opencl_wrapper.o \
			./util/num/num.o \
			./util/timer/timer.o \
			./util/opencl/opencl.o \
			-lm $(OCL_LDFLAGS) \
			-o $(EXE)

# compile function files into objects
main.o:		main.h \
			main.c \
			./kernel/kernel_gpu_opencl_wrapper.h \
			./kernel/kernel_gpu_opencl_wrapper.c \
			./util/num/num.h \
			./util/num/num.c \
			./util/timer/timer.h \
			./util/timer/timer.c \
			./util/opencl/opencl.h \
			./util/opencl/opencl.c
	$(CC) $(CC_FLAGS)	$(OUTPUT) $(KERNEL_DIM) -c main.c \
			-o main.o

./kernel/kernel_gpu_opencl_wrapper.o:	./kernel/kernel_gpu_opencl_wrapper.h \
										./kernel/kernel_gpu_opencl_wrapper.c
	$(CC) $(CC_FLAGS) $(OCL_CPPFLAGS) $(KERNEL_DIM)						./kernel/kernel_gpu_opencl_wrapper.c \
										-c \
										-o ./kernel/kernel_gpu_opencl_wrapper.o

./util/num/num.o:	./util/num/num.h \
					./util/num/num.c
	$(CC) $(CC_FLAGS)			./util/num/num.c \
					-c \
					-o ./util/num/num.o

./util/timer/timer.o:	./util/timer/timer.h \
						./util/timer/timer.c
	$(CC) $(CC_FLAGS)				./util/timer/timer.c \
						-c \
						-o ./util/timer/timer.o

./util/opencl/opencl.o:	./util/opencl/opencl.h \
						./util/opencl/opencl.c
	$(CC) $(CC_FLAGS) $(OCL_CPPFLAGS)		./util/opencl/opencl.c \
						-c \
						-o ./util/opencl/opencl.o

# delete all object and executable files
clean:
	rm	*.o $(EXE) \
		./kernel/*.o \
		./util/num/*.o \
		./util/timer/*.o \
		./util/opencl/*.o \
