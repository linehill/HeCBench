
#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif
#include <fcntl.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/cl.h>
#include "bucketsort.h"
#include "mergesort.h"
#include "./timer/timer.h"

#define VERIFY Y
#define TIMER Y



////////////////////////////////////////////////////////////////////////////////

// Use a static data size for simplicity
//
#define SIZE (128024000)

#define DATA_SIZE (1024)
#define MAX_SOURCE_SIZE (0x100000)
#define HISTOGRAM_SIZE (1024 * sizeof(unsigned int))

////////////////////////////////////////////////////////////////////////////////
int compare(const void *a, const void *b) {
	if(*((float *)a) < *((float *)b)) return -1;
	else if(*((float *)a) > *((float *)b)) return 1;
	else return 0;
}

////////////////////////////////////////////////////////////////////////////////
cl_float4*runMergeSort(int listsize, int divisions,
                               cl_float4 *d_origList, cl_float4 *d_resultList,
                               int *sizes, int *nullElements,
                       unsigned int *origOffsets);

int main(int argc, char** argv)
{
    int err;                            // error code returned from api calls
    
    unsigned int correct;               // number of correct results returned

    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation
    unsigned int *results;

    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    
    cl_mem input;                       // device memory used for the input array
    cl_mem output;                      // device memory used for the output array
    
    // Fill our data set with random float values
    //
    
    int numElements = 0 ;
    int noFileGiven = (argc == 1) || (strcmp(argv[1],"r") == 0);
    if(noFileGiven) {
        printf("using random-generated floats\n");
        numElements = SIZE;
	}
    else {
        printf("reading floats from %s\n", argv[1]);
	FILE *fp = NULL;
        fp = fopen(argv[1],"r");
        if(fp == NULL) {
            printf("Error reading file \n");
            exit(EXIT_FAILURE);
        }
        int count = 0;
        float c;
        
        while(fscanf(fp,"%f",&c) != EOF) {
            count++;
        }
        fclose(fp);
        
        numElements = count;
    }
    printf("Sorting list of %d floats.\n", numElements);
    int mem_size = (numElements + (DIVISIONS*4))*sizeof(float);
	// Allocate enough for the input list
	float *cpu_idata = (float *)malloc(mem_size);
    float *cpu_odata = (float *)malloc(mem_size);
	// Allocate enough for the output list on the cpu side
	float *d_output = (float *)malloc(mem_size);
	// Allocate enough memory for the output list on the gpu side
	float *gpu_odata = (float *)malloc(mem_size);
	float datamin = FLT_MAX;
	float datamax = -FLT_MAX;
    
    if(noFileGiven) {
        for (int i = 0; i < numElements; i++) {
        // Generate random floats between 0 and 1 for the input data
	cpu_idata[i] = ((float) rand() / RAND_MAX);
  
        //Compare data at index to data minimum, if less than current minimum, set that element as new minimum
	datamin = fminf(cpu_idata[i], datamin);
        //Same as above but for maximum
	datamax = fmaxf(cpu_idata[i], datamax);
	}
    } else {
        FILE *fp;
        fp = fopen(argv[1],"r");
        for(int i = 0; i < numElements; i++) {
            fscanf(fp,"%f",&cpu_idata[i]);
            datamin = fminf(cpu_idata[i], datamin);
            datamax = fmaxf(cpu_idata[i], datamax);
        }
	fclose(fp);
    }
    memcpy(cpu_odata, cpu_idata, mem_size);
    uint64_t gpu_start = get_time();

    init_bucketsort(numElements);
    int *sizes = (int*) malloc(DIVISIONS * sizeof(int));
    int *nullElements = (int*) malloc(DIVISIONS * sizeof(int));
    unsigned int *origOffsets = (unsigned int *) malloc((DIVISIONS + 1) * sizeof(int));

    uint64_t bucketsort_start = get_time();
    bucketSort(cpu_idata,d_output,numElements,sizes,nullElements,datamin,datamax, origOffsets);
    uint64_t bucketsort_diff = get_time() - bucketsort_start;

    finish_bucketsort();

    cl_float4 *d_origList = (cl_float4*) d_output;
    cl_float4 *d_resultList = (cl_float4*) cpu_idata;
    
    int newlistsize = 0;
    for(int i = 0; i < DIVISIONS; i++){
        newlistsize += sizes[i] * 4;
    }

    init_mergesort(newlistsize);
    uint64_t mergesort_start = get_time();
    cl_float4 *mergeresult = runMergeSort(newlistsize,DIVISIONS,d_origList,d_resultList,sizes,nullElements,origOffsets);
    uint64_t mergesort_diff = get_time() - mergesort_start;
    finish_mergesort();
    gpu_odata = (float*)mergeresult;

#ifdef TIMER
    uint64_t gpu_diff = get_time() - gpu_start;
    float gpu_msec = (float)gpu_diff / 1000.0f;
    float bucketsort_msec = (float)bucketsort_diff / 1000.0f;
    float mergesort_msec = (float)mergesort_diff / 1000.0f;

    printf("GPU execution time: %0.3f ms  \n", bucketsort_msec+mergesort_msec);
    printf("  --Bucketsort execution time: %0.3f ms \n", bucketsort_msec);
    printf("  --Mergesort execution time: %0.3f ms \n", mergesort_msec);
#endif
#ifdef VERIFY
    uint64_t cpu_start = get_time(), cpu_diff;
    
    qsort(cpu_odata, numElements, sizeof(float), compare);
    cpu_diff = get_time() - cpu_start;
    float cpu_msec = (float)cpu_diff / 1000.0f;
    printf("CPU execution time: %0.3f ms  \n", cpu_msec);
    printf("Checking result...");
    
	// Result checking
	int count = 0;
	for(int i = 0; i < numElements; i++){
		if(cpu_odata[i] != gpu_odata[i])
		{
			printf("Sort missmatch on element %d: \n", i);
			printf("CPU = %f : GPU = %f\n", cpu_odata[i], gpu_odata[i]);
			count++;
			break;
		}
    }
	if(count == 0) printf("PASSED.\n");
	else printf("FAILED.\n");
#endif
    
#ifdef OUTPUT
    FILE *tp1;
    const char filename3[]="./hybridoutput.txt";
    tp1 = fopen(filename3,"w");
    for(int i = 0; i < SIZE; i++) {
        fprintf(tp1,"%f ",cpu_idata[i]);
    }
    
    fclose(tp1);
#endif
    

//    printf("%d \n",cpu_odata[1]);
//    int summy = 0;
//    for(int i =0; i < HISTOGRAM_SIZE; i++)
//        summy+=cpu_odata[i];
//    printf("%d \n", summy);
    return 0;
}



