CC          = hipcc
OPTIMIZE    = yes
DUMP        = yes
DEBUG       = no


# Standard flags
CFLAGS := $(EXTRA_CFLAGS)

# Compilation flags for clang-based platforms (amd, chipstar).
ifeq ($(findstring $(HIP_PLATFORM), nvidia nvcc),)
  CFLAGS += -Wall
endif

# Linker flags
LDFLAGS = -lm

# Enable checksum and output file
ifeq ($(DEBUG),yes)
	CFLAGS += -g  
	TEST_ON = -DTEST_CHECKSUM 
endif

ifeq ($(DUMP),yes)
	OUTPUT  = -DOUTPUT
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
	CFLAGS += -O3
endif


main:	./main.o \
	./kernel/kernel.o \
	./util/avi/avilib.o \
	./util/avi/avimod.o \
	./util/file/file.o \
	./util/timer/timer.o 
	$(CC)	$(CFLAGS) ./main.o \
		./kernel/kernel.o \
		./util/avi/avilib.o \
		./util/avi/avimod.o \
		./util/file/file.o \
		./util/timer/timer.o \
		$(LDFLAGS) -o main

# ======================================================================================================================================================150
#	MAIN FUNCTION
# ======================================================================================================================================================150

./main.o:	./main.h \
	./main.cu
	$(CC)	$(OUTPUT) $(KERNEL_DIM) ./main.cu \
		-I./util/timer/ \
		-I./util/file/ \
		-c \
		-o ./main.o 

# ======================================================================================================================================================150
#	KERNELS
# ======================================================================================================================================================150

./kernel/kernel.o:./kernel/kernel.h ./kernel/kernel.cu
	$(CC) $(CFLAGS) $(TEST_ON) $(KERNEL_DIM) ./kernel/kernel.cu \
		-c \
		-o ./kernel/kernel.o 

# ======================================================================================================================================================150
#	UTILITIES
# ======================================================================================================================================================150

./util/avi/avilib.o:./util/avi/avilib.h ./util/avi/avilib.c
	$(CC)	$(CFLAGS)  \
		-I./util/avi \
		-o ./util/avi/avilib.o \
		-c ./util/avi/avilib.c

./util/avi/avimod.o:./util/avi/avilib.h ./util/avi/avimod.c
	$(CC)	$(CFLAGS)  \
		-I./util/avi \
		-o ./util/avi/avimod.o \
		-c ./util/avi/avimod.c

./util/file/file.o:	./util/file/file.h ./util/file/file.c
	$(CC)	$(CFLAGS)  \
		-I./util/file \
		-I./util/timer \
		-o ./util/file/file.o \
		-c ./util/file/file.c

./util/timer/timer.o:	./util/timer/timer.h ./util/timer/timer.c
	$(CC)	$(CFLAGS) \
		-I./util/timer \
		-o ./util/timer/timer.o \
		-c ./util/timer/timer.c


clean:
	rm	-rf *.o main \
		./kernel/*.o \
		./util/avi/*.o \
		./util/file/*.o \
		./util/timer/*.o 

run: main
	./main 104
