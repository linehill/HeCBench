include ../../common/make.config

ifdef VERIFY
override VERIFY = -DVERIFY
endif

ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

ifdef TIMER
override TIMER = -DTIMER
endif

EXE = main

$(EXE): hybridsort.c  bucketsort.c mergesort.c timer/timer.c
	$(CC) $(CC_FLAGS) -o $(EXE) $(VERIFY) $(OUTPUT) $(TIMER) $(HISTO_WG_SIZE_0) $(BUCKET_WG_SIZE_0) $(BUCKET_WG_SIZE_1) $(MERGE_WG_SIZE_0) $(MERGE_WG_SIZE_1) $^ $(OCL_CPPFLAGS) $(OCL_LDFLAGS) -lm

clean:
	rm -f $(EXE)
