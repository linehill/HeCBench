include ../../common/make.config

kmeans: cluster.o getopt.o read_input.o kmeans_clustering.o rmse.o kmeans.o
	$(CXX) -fPIE kmeans.o cluster.o getopt.o read_input.o kmeans_clustering.o rmse.o -o kmeans $(OCL_LDFLAGS)

cluster.o: cluster.c kmeans.h
	$(CC) $(CC_FLAGS) -o $@ cluster.c -c

getopt.o: getopt.c
	$(CC)  $(CC_FLAGS) -o $@ getopt.c -c

read_input.o: read_input.c kmeans.h
	$(CC) $(CC_FLAGS) -o $@ -c read_input.c

rmse.o: rmse.c kmeans.h
	$(CC) $(CC_FLAGS) -o $@ rmse.c -c

kmeans.o: kmeans.cpp kmeans.h
	$(CXX) $(CXX_FLAGS) $(KERNEL_DIM) $(OCL_CPPFLAGS) -o $@  -c kmeans.cpp

clean:
	rm -f *.o *~ kmeans *.linkinfo
