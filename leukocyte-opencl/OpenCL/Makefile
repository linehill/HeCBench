include ../../../common/make.config

ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

# Matrix library path
MATRIX_DIR = ../meschach_lib

leukocyte: detect_main.o avilib.o find_ellipse.o find_ellipse_opencl.o track_ellipse.o track_ellipse_opencl.o misc_math.o OpenCL_helper_library.o $(MATRIX_DIR)/meschach.a
	$(CC)  -I$(MATRIX_DIR) $(CC_FLAGS)  avilib.o find_ellipse.o find_ellipse_opencl.o track_ellipse.o track_ellipse_opencl.o misc_math.o OpenCL_helper_library.o detect_main.o -o leukocyte $(MATRIX_DIR)/meschach.a $(OCL_LDFLAGS) -lm

%.o: %.c
	$(CC) -I$(MATRIX_DIR) $(OUTPUT) $(CC_FLAGS) $(OCL_CPPFLAGS) -o $@ -c $<

detect_main.o: detect_main.c find_ellipse.h track_ellipse.h avilib.h

find_ellipse.o: avilib.h

track_ellipse.o: find_ellipse.h avilib.h

$(MATRIX_DIR)/meschach.a:
	cd $(MATRIX_DIR); ./configure --with-all; make all; make clean

clean:
	rm -f *.o *~ leukocyte
