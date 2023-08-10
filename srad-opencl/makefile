include ../../common/make.config

srad:	./main.o \
			./kernel/kernel_gpu_opencl_wrapper.o \
			./util/opencl/opencl.o \
			./util/graphics/graphics.o \
			./util/graphics/resize.o \
			./util/timer/timer.o
	$(CC)	./main.o \
			./kernel/kernel_gpu_opencl_wrapper.o \
			./util/opencl/opencl.o \
			./util/graphics/graphics.o \
			./util/graphics/resize.o \
			./util/timer/timer.o \
			-lm $(OCL_LDFLAGS) -o srad

# ======================================================================================================================================================150
#	MAIN FUNCTION
# ======================================================================================================================================================150

./main.o:	./main.h \
			./main.c
	$(CC) $(CC_FLAGS) $(KERNEL_DIM) ./main.c \
			-c \
			-o ./main.o

# ======================================================================================================================================================150
#	KERNELS
# ======================================================================================================================================================150

./kernel/kernel_gpu_opencl_wrapper.o:	./kernel/kernel_gpu_opencl_wrapper.h \
										./kernel/kernel_gpu_opencl_wrapper.c
	$(CC) $(CC_FLAGS) $(KERNEL_DIM) ./kernel/kernel_gpu_opencl_wrapper.c \
			-c \
			-o ./kernel/kernel_gpu_opencl_wrapper.o \
			$(OCL_CPPFLAGS)

# ======================================================================================================================================================150
#	UTILITIES
# ======================================================================================================================================================150

./util/graphics/graphics.o:	./util/graphics/graphics.h \
							./util/graphics/graphics.c
	$(CC) $(CC_FLAGS) ./util/graphics/graphics.c \
			-c \
			-o ./util/graphics/graphics.o

./util/graphics/resize.o:	./util/graphics/resize.h \
							./util/graphics/resize.c
	$(CC) $(CC_FLAGS) ./util/graphics/resize.c \
			-c \
			-o ./util/graphics/resize.o

./util/timer/timer.o:	./util/timer/timer.h \
						./util/timer/timer.c
	$(CC) $(CC_FLAGS) ./util/timer/timer.c \
			-c \
			-o ./util/timer/timer.o

./util/opencl/opencl.o:	./util/opencl/opencl.h \
						./util/opencl/opencl.c
	$(CC) $(CC_FLAGS) ./util/opencl/opencl.c \
			-c \
			-o ./util/opencl/opencl.o \
			$(OCL_CPPFLAGS)

clean:
	rm -f   ./*.o srad\
		./kernel/*.o \
		./util/opencl/*.o \
		./util/graphics/*.o \
		./util/timer/*.o
