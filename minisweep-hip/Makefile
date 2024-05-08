#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++14 -I../minisweep-cuda

# Compilation flags for clang-based platforms (amd, chipstar).
ifeq ($(findstring $(HIP_PLATFORM), nvidia nvcc),)
  CFLAGS += -Wall
endif

# Compilation flags for NVidia platform.
ifneq ($(findstring $(HIP_PLATFORM), nvidia nvcc),)
  # sm_60 is minimum for atomicAdd(double *, double).
  CFLAGS += -arch=sm_60
endif

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu ../minisweep-cuda/kernels.cu \
          ../minisweep-cuda/utils.cu \
          ../minisweep-cuda/utils.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) --niterations 100
	./$(program) --niterations 1 --ncell_x 32 --ncell_y 32 --ncell_z 64 --ne 64 --na 32 --nblock_z 64
