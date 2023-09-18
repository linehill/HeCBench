/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h> 
#include <chrono>
#include <hip/hip_runtime_api.h>
#include <hipsparse/hipsparse.h>
#include "utils.h"

#define CHECK_HIP(func)                                                  \
{                                                                        \
    hipError_t status = (func);                                          \
    if (status != hipSuccess) {                                          \
        printf("HIP API failed at line %d with error: %s (%d)\n",        \
               __LINE__, hipGetErrorString(status), status);             \
        return EXIT_FAILURE;                                             \
    }                                                                    \
}

#define CHECK_HIPSPARSE_ERROR_CASE__(token_) \
    case token_:                             \
        fprintf(stderr, "HIPSPARSE API failed at line %d with error: %s\n",         \
               __LINE__, #token_); \
        break

#define CHECK_HIPSPARSE(error)                                                      \
    {                                                                                     \
        auto local_error = (error);                                                       \
        if(local_error != HIPSPARSE_STATUS_SUCCESS)                                       \
        {                                                                                 \
            fprintf(stderr, "hipSPARSE error: ");                                         \
            switch(local_error)                                                           \
            {                                                                             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_SUCCESS);                   \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_NOT_INITIALIZED);           \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_ALLOC_FAILED);              \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_INVALID_VALUE);             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_ARCH_MISMATCH);             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_MAPPING_ERROR);             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_EXECUTION_FAILED);          \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_INTERNAL_ERROR);            \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED); \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_ZERO_PIVOT);                \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_NOT_SUPPORTED);             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES);    \
            }                                                                             \
            fprintf(stderr, "\n");                                                        \
            return local_error;                                                           \
        }                                                                                 \
    }                                                                                     


int main(int argc, char *argv[])
{
  int repeat = 1;

  if (argc != 8) {
    printf("Single-precision batched dense matrix - dense matrix multiplication into sparse matrix,\n");
    printf("where the sparse matrix is represented in CSR (Compressed Sparse Row) storage format\n");
    printf("Usage %s <number of batches> <M> <K> <N> <number of non-zero elements> <repeat> <verify>\n", argv[0]);
    printf("SDDM (A, B, C) where (A: M * K, B: K * N, C: M * N)\n");
    return 1;
  }

  int b, m, k, n, nnz, verify;

  b = atoi(argv[1]); // number of batches
  m = atoi(argv[2]);
  k = atoi(argv[3]);
  n = atoi(argv[4]);
  nnz = atoi(argv[5]);
  repeat = atoi(argv[6]);
  verify = atoi(argv[7]);

  const int A_num_rows   = m;
  const int A_num_cols   = k;
  const int B_num_rows   = A_num_cols;
  const int B_num_cols   = n;
  const int C_nnz     = nnz;
  const int lda       = A_num_cols;
  const int ldb       = B_num_cols;
  const int A_size    = lda * A_num_rows;
  const int B_size    = ldb * B_num_rows;
  const int C_size    = A_num_rows * B_num_cols;

  const size_t value_size_bytes  = b * C_nnz * sizeof(float);
  const size_t colidx_size_bytes = b * C_nnz * sizeof(int);
  const size_t rowidx_size_bytes = b * (A_num_rows + 1) * sizeof(size_t);

  float *hA = (float*) malloc (b * A_size * sizeof(float));
  float *hB = (float*) malloc (b * B_size * sizeof(float));

  // contain the result of A * B
  float *hC = (float*) malloc (b * C_size * sizeof(float));

  float *hC_values = (float*) malloc (value_size_bytes);
  int *hC_columns = (int*) malloc (colidx_size_bytes);
  int *hC_offsets = (int*) malloc (rowidx_size_bytes);
  float *hC_result  = (float*) malloc (value_size_bytes);

  for (int i = 0; i < b; i++) {
    init_matrix(hA + i * A_size, A_num_rows, A_num_cols, A_size);
    init_matrix(hB + i * B_size, B_num_rows, B_num_cols, B_size);
    init_matrix(hC + i * C_size, A_num_rows, B_num_cols, C_nnz);

    // initialize the CSR structures for the C matrix
    init_csr(hC_offsets + i * (A_num_rows+1),
             hC_values + i * C_nnz,
             hC_columns + i * C_nnz,
             hC + i * C_size,
             A_num_rows, B_num_cols, C_nnz);
  }

  if (verify) {
    printf("Computing the reference SDDMM results (batch size = %d)..\n", b);
    for (int i = 0; i < b; i++) {
      sddmm (hA + i * A_size,
             hB + i * B_size,
             hC + i * C_size,
             hC_result + i * C_nnz,
             hC_offsets + i * (A_num_rows+1),
             hC_columns + i * C_nnz,
             A_num_cols, A_num_rows, B_num_cols);
    }
    printf("Done\n");
  }

  float alpha        = 1.0f;
  float beta         = 0.0f;
  //--------------------------------------------------------------------------
  // Device memory management
  int *dC_columns;
  size_t *dC_offsets;
  float *dC_values, *dB, *dA;
  CHECK_HIP( hipMalloc((void**) &dA, b * A_size * sizeof(float)) )
  CHECK_HIP( hipMalloc((void**) &dB, b * B_size * sizeof(float)) )
  CHECK_HIP( hipMalloc((void**) &dC_offsets, rowidx_size_bytes) )
  CHECK_HIP( hipMalloc((void**) &dC_columns, colidx_size_bytes) )
  CHECK_HIP( hipMalloc((void**) &dC_values,  value_size_bytes) )

  CHECK_HIP( hipMemcpy(dA, hA, b * A_size * sizeof(float),
                       hipMemcpyHostToDevice) )
  CHECK_HIP( hipMemcpy(dB, hB, b * B_size * sizeof(float),
                       hipMemcpyHostToDevice) )
  CHECK_HIP( hipMemcpy(dC_offsets, hC_offsets, rowidx_size_bytes,
                       hipMemcpyHostToDevice) )
  CHECK_HIP( hipMemcpy(dC_columns, hC_columns, colidx_size_bytes,
                       hipMemcpyHostToDevice) )
  CHECK_HIP( hipMemcpy(dC_values, hC_values, value_size_bytes,
                       hipMemcpyHostToDevice) )
  //--------------------------------------------------------------------------
  // HIPSPARSE APIs
  hipsparseHandle_t     handle = NULL;
  hipsparseDnMatDescr_t matA, matB;
  hipsparseSpMatDescr_t matC;
  void*                 dBuffer    = NULL;
  size_t                bufferSize = 0;
  CHECK_HIPSPARSE( hipsparseCreate(&handle) )
  // Create dense matrix A
  CHECK_HIPSPARSE( hipsparseCreateDnMat(&matA, A_num_rows, A_num_cols, lda, dA,
                                        HIP_R_32F, HIPSPARSE_ORDER_ROW) )
  CHECK_HIPSPARSE( hipsparseDnMatSetStridedBatch(matA, b, A_size) )

  // Create dense matrix B
  CHECK_HIPSPARSE( hipsparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                        HIP_R_32F, HIPSPARSE_ORDER_ROW) )
  CHECK_HIPSPARSE( hipsparseDnMatSetStridedBatch(matB, b, B_size) )

  // Create sparse matrix C in CSR format
  CHECK_HIPSPARSE( hipsparseCreateCsr(&matC, A_num_rows, B_num_cols, C_nnz,
                                      dC_offsets, dC_columns, dC_values,
                                      HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                                      HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F) )
  CHECK_HIPSPARSE( hipsparseCsrSetStridedBatch(matC, b, A_num_rows+1, C_nnz) )

  // allocate an external buffer if needed
  CHECK_HIPSPARSE( hipsparseSDDMM_bufferSize(
                               handle,
                               HIPSPARSE_OPERATION_NON_TRANSPOSE,
                               HIPSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, matB, &beta, matC, HIP_R_32F,
                               HIPSPARSE_SDDMM_ALG_DEFAULT, &bufferSize) )
  CHECK_HIP( hipMalloc(&dBuffer, bufferSize) )

  // execute preprocess (optional)
  CHECK_HIPSPARSE( hipsparseSDDMM_preprocess(
                                handle,
                                HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, HIP_R_32F,
                                HIPSPARSE_SDDMM_ALG_DEFAULT, dBuffer) )

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    // execute SpMM
    CHECK_HIPSPARSE( hipsparseSDDMM(handle,
                                    HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                    HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, matB, &beta, matC, HIP_R_32F,
                                    HIPSPARSE_SDDMM_ALG_DEFAULT, dBuffer) )
  }
  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of SDDMM: %f (us)\n", (time * 1e-3f) / repeat);

  // destroy matrix/vector descriptors
  CHECK_HIPSPARSE( hipsparseDestroyDnMat(matA) )
  CHECK_HIPSPARSE( hipsparseDestroyDnMat(matB) )
  CHECK_HIPSPARSE( hipsparseDestroySpMat(matC) )
  CHECK_HIPSPARSE( hipsparseDestroy(handle) )

  //--------------------------------------------------------------------------
  if (verify) {
    // device result check
    CHECK_HIP( hipMemcpy(hC_values, dC_values, value_size_bytes,
                         hipMemcpyDeviceToHost) )
    int correct = 1;

    for (int i = 0; i < b; i++) {
      float *v =  hC_values + i * C_nnz;
      float *r =  hC_result + i * C_nnz;
      for (int j = 0; j < C_nnz; j++) {
        if (fabsf(v[j] - r[j]) > 1e-2f) {
          printf("@batch%d index%d: %f != %f\n", i, j, v[j], r[j]);
          correct = 0;
          break;
        }
      }
      if (!correct) break;
    }
    if (correct)
        printf("sddmm_csr_batched_example test PASSED\n");
    else
        printf("sddmm_csr_batched_example test FAILED: wrong result\n");
  }
  //--------------------------------------------------------------------------
  // device memory deallocation
  CHECK_HIP( hipFree(dBuffer) )
  CHECK_HIP( hipFree(dA) )
  CHECK_HIP( hipFree(dB) )
  CHECK_HIP( hipFree(dC_offsets) )
  CHECK_HIP( hipFree(dC_columns) )
  CHECK_HIP( hipFree(dC_values) )

  free(hA);
  free(hB);
  free(hC);
  free(hC_values);
  free(hC_columns);
  free(hC_offsets);
  free(hC_result);

  return EXIT_SUCCESS;
}
