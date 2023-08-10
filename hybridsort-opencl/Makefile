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

hybridsort: hybridsort.c bucketsort.h mergesort.h bucketsort.c mergesort.c
	$(CC) $(CC_FLAGS) -o hybridsort $(VERIFY) $(OUTPUT) $(TIMER) $(HISTO_WG_SIZE_0) $(BUCKET_WG_SIZE_0) $(BUCKET_WG_SIZE_1) $(MERGE_WG_SIZE_0) $(MERGE_WG_SIZE_1) hybridsort.c bucketsort.c mergesort.c $(OCL_CPPFLAGS) $(OCL_LDFLAGS) -lm

clean:
	rm -f  hybridsort
