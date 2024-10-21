/*
*  This file is part of Christian's OpenMP software lab 
*
*  Copyright (C) 2016 by Christian Terboven <terboven@itc.rwth-aachen.de>
*  Copyright (C) 2016 by Jonas Hahnfeld <hahnfeld@itc.rwth-aachen.de>
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this program; if not, write to the Free Software
*  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
*
*/

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/time.h>
#include <omp.h>

#include <iostream>
#include <algorithm>

#include <cstdlib>
#include <cstdio>

#include <cmath>
#include <ctime>
#include <cstring>

#include <string.h>
#include <fstream> 
#include <iomanip>

const int chunkSize = 256;

void runTests();

/**
  * helper routine: check if array is sorted correctly
  */
bool isSorted(int ref[], int data[], const size_t size){
	std::sort(ref, ref + size);
	for (size_t idx = 0; idx < size; ++idx){
		if (ref[idx] != data[idx]) {
			return false;
		}
	}
	return true;
}


/**
  * sequential merge step (straight-forward implementation)
  */
// TODO: cut-off could also apply here (extra parameter?)
// TODO: optional: we can also break merge in two halves
void MsMergeSequential(int *out, int *in, long begin1, long end1, long begin2, long end2, long outBegin) {
	long left = begin1;
	long right = begin2;

	long idx = outBegin;

	while (left < end1 && right < end2) {
		if (in[left] <= in[right]) {
			out[idx] = in[left];
			left++;
		} else {
			out[idx] = in[right];
			right++;
		}
		idx++;
	}

	while (left < end1) {
		out[idx] = in[left];
		left++, idx++;
	}

	while (right < end2) {
		out[idx] = in[right];
		right++, idx++;
	}
}


/**
  * sequential MergeSort
  */
// TODO: remember one additional parameter (depth)
// TODO: recursive calls could be taskyfied
// TODO: task synchronization also is required
void MsSequential(int *array, int *tmp, bool inplace, long begin, long end) {
	if (begin < (end - 1)) {
		const long half = (begin + end) / 2;
		MsSequential(array, tmp, !inplace, begin, half);
		MsSequential(array, tmp, !inplace, half, end);
		if (inplace) {
			MsMergeSequential(array, tmp, begin, half, half, end, begin);
		} else {
			MsMergeSequential(tmp, array, begin, half, half, end, begin);
		}
	} else if (!inplace) {
		tmp[begin] = array[begin];
	}
}

void MsParallel(int *array, int *tmp, bool inplace, long begin, long end, int depth) {

    //printf("thread %d, begin %ld, end %ld, depth %d\n", omp_get_thread_num(), begin, end, depth);

    if (begin < (end - 1)) {
        const long half = (begin + end) / 2;

        if (depth > 0) {
            #pragma omp task 
            MsParallel(array, tmp, !inplace, begin, half, depth - 1);
            
            #pragma omp task
            MsParallel(array, tmp, !inplace, half, end, depth - 1);
            
        } else {
            MsParallel(array, tmp, !inplace, begin, half, depth);
            MsParallel(array, tmp, !inplace, half, end, depth);
        }

        #pragma omp taskwait

        if (inplace) {
            MsMergeSequential(array, tmp, begin, half, half, end, begin);
        } else {
            MsMergeSequential(tmp, array, begin, half, half, end, begin);
        }
    } else if (!inplace) {
        tmp[begin] = array[begin];
    }
}


/**
  * Serial or Parallel MergeSort
  */
// TODO: this function should create the parallel region
// TODO: good point to compute a good depth level (cut-off)

void MsMergeSort(int *array, int *tmp, const size_t size, int parallel, int depth, int nthread) {

    if (parallel == 0) { //sequential
        MsSequential(array, tmp, true, 0, size);
    } 
    else if(parallel == 1) { //fixed depth, variable number of threads
        #pragma omp parallel num_threads(nthread)
        {
            #pragma omp single
            {
                depth = log(size / chunkSize + 1);
                MsParallel(array, tmp, true, 0, size, depth);
            }
        }
    }
    else if(parallel == 2) { //fixed number of threads, variable depth
        #pragma omp parallel num_threads(8)
        {
            #pragma omp single
            {
                MsParallel(array, tmp, true, 0, size, depth);
            }
        }
    }
    else {
        printf("Invalid parallel flag\n");
    }
}


/** 
  * @brief program entry point
  */
int main(int argc, char* argv[]) {
    struct timeval t1, t2;
    double etime;
    bool parallel = 0;
    int depth;
    int nthread;

    if (argc > 1) { // with 0 arguments, run tests
        if (argc < 3) {
            fprintf(stderr, "Usage: %s <size> <mode [0,1,2]> <depth|numthread>\n", argv[0]);
            fprintf(stderr, "Mode 0: sequential\nMode 1: specify nthread\nMode 2: specify cut-off\n");
            return 1; 
        }

        const size_t stSize = strtol(argv[1], NULL, 10);
        parallel = atoi(argv[2]);

        if (parallel == 1 && argc == 4) {
            nthread = atoi(argv[3]);
        } 
        else if (parallel == 2 && argc == 4) {
            depth = atoi(argv[3]);
        }
        else if (parallel != 0) {
            printf("Missing depth|nthread\n");
            fprintf(stderr, "Usage: %s <size> <mode [0,1,2]> <depth|numthread>\n", argv[0]);
            fprintf(stderr, "Mode 0: sequential\nMode 1: specify nthread\nMode 2: specify cut-off\n");
            return 1;
        }

        int *data = (int*) malloc(stSize * sizeof(int));
        int *tmp = (int*) malloc(stSize * sizeof(int));
        int *ref = (int*) malloc(stSize * sizeof(int));

        if (data == NULL || tmp == NULL || ref == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            return 1; 
        }

        printf("Initialization...\n");

        srand(95); 
        for (size_t idx = 0; idx < stSize; ++idx) {
            data[idx] = (int)(stSize * ((double)rand() / RAND_MAX));
        }
        memcpy(ref, data, stSize * sizeof(int));

        double dSize = (stSize * sizeof(int)) / 1024.0 / 1024.0;
        printf("Sorting %zu elements of type int (%f MiB)...\n", stSize, dSize);

        gettimeofday(&t1, NULL);
        MsMergeSort(data, tmp, stSize, parallel, depth, nthread);
        gettimeofday(&t2, NULL);

        etime = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        etime = etime / 1000.0;

        printf("done, took %f sec. Verification...", etime);
        if (isSorted(ref, data, stSize)) {
            printf(" successful.\n");
        } else {
            printf(" FAILED.\n");
        }

        free(data);
        free(tmp);
        free(ref);
    } else {
        runTests();
    }

    return 0;
}

void runTests() {
    std::ofstream logFile("test_results.csv");
    if (!logFile) {
        std::cerr << "Error opening log file!" << std::endl;
        return;
    }

    std::cout << "Testing..." << std::endl;

    logFile << "Size,Depth,Time (seconds)\n"; 

    std::vector<int> nthreads = {0, 1, 2, 4, 8};

    for (size_t size = 10; size <= 1e8; size *= 10) {
        for (int nthread : nthreads) {
            int *data = (int*)malloc(size * sizeof(int));
            int *tmp = (int*)malloc(size * sizeof(int));
            int *ref = (int*)malloc(size * sizeof(int));

            if (data == NULL || tmp == NULL || ref == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                return;
            }

            srand(95);
            for (size_t idx = 0; idx < size; ++idx) {
                data[idx] = (int)(size * ((double)rand() / RAND_MAX));
            }
            memcpy(ref, data, size * sizeof(int));

            printf("Testing size %.0e, thread %d\n", (double)size, nthread);

            struct timeval t1, t2;
            gettimeofday(&t1, NULL);
            if (nthread == 0) {
                MsMergeSort(data, tmp, size, 0, 0, 0);
            } else {
                MsMergeSort(data, tmp, size, 1, 0, nthread);
            }
            gettimeofday(&t2, NULL);

            double etime = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
            etime = etime / 1000.0;

            logFile << size << "," << nthread << "," << etime << "\n";

            free(data);
            free(tmp);
            free(ref);
        }
    }

    logFile.close();
    std::cout << "Testing complete. Results saved to test_results.csv." << std::endl;
}
