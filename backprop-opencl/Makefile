include ../../common/make.config

main: backprop.o imagenet.o facetrain.o backprop_kernel.cl  backprop_ocl.cpp
	$(CXX) $(CXX_FLAGS) backprop_ocl.cpp backprop.o imagenet.o facetrain.o -o main $(OCL_CPPFLAGS) $(OCL_LDFLAGS)

backprop.o: backprop.c 
	$(CC) $(CC_FLAGS) backprop.c -c
	
facetrain.o: facetrain.c
	$(CC) $(CC_FLAGS) facetrain.c -c

imagenet.o: imagenet.c
	$(CC) $(CC_FLAGS) imagenet.c -c

clean:
	rm -f *.o *~ main *.linkinfo
