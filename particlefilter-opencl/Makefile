include ../../common/make.config


ALL: OCL_particlefilter_naive OCL_particlefilter_double OCL_particlefilter_single

OCL_particlefilter_naive: ex_particle_OCL_naive_seq.o
	$(CXX) ex_particle_OCL_naive_seq.o $(OCL_LDFLAGS) -o OCL_particlefilter_naive -lm

OCL_particlefilter_double: ex_particle_OCL_double_seq.o
	$(CXX) ex_particle_OCL_double_seq.o $(OCL_LDFLAGS) -o OCL_particlefilter_double -lm

OCL_particlefilter_single: ex_particle_OCL_single_seq.o
	$(CXX) ex_particle_OCL_single_seq.o $(OCL_LDFLAGS) -o OCL_particlefilter_single -lm


ex_particle_OCL_naive_seq.o: ex_particle_OCL_naive_seq.cpp
	$(CXX) $(CXX_FLAGS) $(OCL_CPPFLAGS)  $(INCLUDES) ex_particle_OCL_naive_seq.cpp -c

ex_particle_OCL_double_seq.o: ex_particle_OCL_double_seq.cpp
	$(CXX) $(CXX_FLAGS) $(OCL_CPPFLAGS)  $(INCLUDES) ex_particle_OCL_double_seq.cpp -c

ex_particle_OCL_single_seq.o: ex_particle_OCL_single_seq.cpp
	$(CXX) $(CXX_FLAGS) $(OCL_CPPFLAGS)  $(INCLUDES) ex_particle_OCL_single_seq.cpp -c

clean:
	rm -f *.o *~ OCL_particlefilter_naive OCL_particlefilter_double OCL_particlefilter_single *.linkinfo output.txt
