include ../../common/make.config


main:	./main.o \
		./kernel/kernel_gpu_opencl_wrapper.o \
		./kernel/kernel_gpu_opencl_wrapper_2.o \
		./util/timer/timer.o \
		./util/num/num.o \
		./util/opencl/opencl.o
	$(CC)	./main.o \
			./kernel/kernel_gpu_opencl_wrapper.o \
			./kernel/kernel_gpu_opencl_wrapper_2.o \
			./util/timer/timer.o \
			./util/num/num.o \
			./util/opencl/opencl.o \
			-lm $(OCL_LDFLAGS) -o main

main.o:	./common.h \
		./main.h \
		./main.c
	$(CC) $(CC_FLAGS) $(KERNEL_DIM) ./main.c \
			-c \
			-o ./main.o

# ======================================================================================================================================================150
#	KERNELS
# ======================================================================================================================================================150

./kernel/kernel_gpu_opencl_wrapper.o:	./common.h \
										./kernel/kernel_gpu_opencl_wrapper.h \
										./kernel/kernel_gpu_opencl_wrapper.c
	$(CC) $(CC_FLAGS) $(KERNEL_DIM) $(OCL_CPPFLAGS) -c ./kernel/kernel_gpu_opencl_wrapper.c \
			-o ./kernel/kernel_gpu_opencl_wrapper.o

./kernel/kernel_gpu_opencl_wrapper_2.o:	./common.h \
										./kernel/kernel_gpu_opencl_wrapper_2.h \
										./kernel/kernel_gpu_opencl_wrapper_2.c
	$(CC)  $(CC_FLAGS) $(KERNEL_DIM) $(OCL_CPPFLAGS) -c ./kernel/kernel_gpu_opencl_wrapper_2.c \
			-o ./kernel/kernel_gpu_opencl_wrapper_2.o

# ======================================================================================================================================================150
#	UTILITIES
# ======================================================================================================================================================150

./util/timer/timer.o:	./common.h \
						./util/timer/timer.h \
						./util/timer/timer.c
	$(CC)  $(CC_FLAGS) ./util/timer/timer.c \
			-c \
			-o ./util/timer/timer.o

./util/num/num.o:	./common.h \
					./util/num/num.h \
					./util/num/num.c
	$(CC) $(CC_FLAGS) ./util/num/num.c \
			-c \
			-o ./util/num/num.o

./util/opencl/opencl.o:	./common.h \
						./util/opencl/opencl.h \
						./util/opencl/opencl.c
	$(CC)  $(CC_FLAGS) ./util/opencl/opencl.c \
			-c \
			-o ./util/opencl/opencl.o \
			$(OCL_CPPFLAGS)

# ======================================================================================================================================================150
#	END
# ======================================================================================================================================================150

# ========================================================================================================================================================================================================200
#	DELETE
# ========================================================================================================================================================================================================200

clean:
	rm -f   *.o main \
		./kernel/*.o \
		./util/timer/*.o \
		./util/num/*.o \
		./util/opencl/*.o \
                output.txt
# ========================================================================================================================================================================================================200
#	END
# ========================================================================================================================================================================================================200
