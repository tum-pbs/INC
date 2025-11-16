template <typename T>
__global__ void csrExtractDiag(const T* csr_values, const int* csr_col_ind, const int* csr_row_ptr,
  T* matrix_diag, const int matrix_shape)
{
  for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < matrix_shape; row += blockDim.x * gridDim.x)
  {
     bool diagonal_found = false;
     for(size_t index = csr_row_ptr[row]; index < csr_row_ptr[row+1]; index++){
       if(csr_col_ind[index]==row){
         matrix_diag[row] = csr_values[index];
         diagonal_found = true;
         break;
       }
     }
     if (diagonal_found==false) matrix_diag[row]=0;
  }
}

double csrMatrixTrace(cublasHandle_t blasHandle, const double* csr_values, const int* csr_col_ind, const int* csr_row_ptr,
  const int matrix_shape, cudaStream_t stream)
{
  double trace;
  double* matrix_diag;
  cdpErrchk(cudaMalloc(&matrix_diag, matrix_shape*sizeof(double)));
  int minGridSize = 0, blockSize = 0, gridSize = 0;
  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, csrExtractDiag<double>, 0, 0);
  gridSize = (matrix_shape + blockSize - 1) / blockSize;
  csrExtractDiag<<<gridSize, blockSize>>>(csr_values, csr_col_ind, csr_row_ptr, matrix_diag, matrix_shape);
  print_array<<<1,1,0,0>>>(matrix_diag, matrix_shape);
  CUDA_CHECK_RETURN(cudaDeviceSynchronize());
  cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasDasum(blasHandle, matrix_shape, matrix_diag, 1, &trace));
  cudaFree(matrix_diag);
  return trace;
}

// without preconditioner

__host__ void CgSolveLauncherFloat(float * csr_valuesA, int* csr_row_ptr, int* csr_col_ind,
  const float* rhs, const int nnz_a,  const int matrix_shape, const float *x_old, const float* tol_gpu, int max_it,
  // float *s,  float *p, float *r, float *rhat, float *v, float *t,
  float * x,  const bool laplace_rank_deficient, const int residual_reset_steps,
  bool* warning, cudaStream_t stream)
{
  /*  Computes an un-preconditioned CG loop on SINGLE precision
      supports features:
        - exit loop when tolerance is reached (vector norm on residual)
        - exit loop when max_it is reached
        - treat rank-deficient matrices (rank=rowcount-1) by multiplying with matrix of ones implicitly
        - re-calculation of the true residual every residual_reset_steps
  */
  bool residual_reset_bool = 1;
  if (residual_reset_steps==0) {
    residual_reset_bool = 0;
  }
  // 0. Set all helper variables to zero, Initialise handles
  float rho = 1., rhoprev = 1., alpha = 1., beta = 1., one = 1.;
  float negative_one = -1.;
  float zero = 0., temp = 0;
  float rank_deficient_scaling = 1;
  float *p, *r, *rhat;
  cdpErrchk(cudaMalloc((void**)&p , matrix_shape*sizeof(float)));
  cdpErrchk(cudaMalloc((void**)&r , matrix_shape*sizeof(float)));
  cdpErrchk(cudaMalloc((void**)&rhat , matrix_shape*sizeof(float)));
  float * gpuOne;
  cdpErrchk(cudaMalloc((void**)&gpuOne, sizeof(float)));
  cdpErrchk(cudaMemcpy(gpuOne, &one, sizeof(float), cudaMemcpyHostToDevice));
  //printf("rho %f   rhoprev %f   alpha %f   omega %f    ones %f\n", rho, rhoprev, alpha, omega,one);
  float tol;
  cdpErrchk(cudaMemcpy(&tol, tol_gpu, sizeof(float), cudaMemcpyDeviceToHost));
  cusparseMatDescr_t descrA;
  cdpErrchk_sparse(cusparseCreateMatDescr(&descrA));
  cdpErrchk_sparse(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
  cdpErrchk_sparse(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
  cusparseHandle_t  sparseHandle;
  cublasHandle_t    blasHandle;
  cdpErrchk_sparse(cusparseCreate(&sparseHandle));
  cdpErrchk_blas(cublasCreate(&blasHandle));
  CUDA_CHECK_RETURN(cudaDeviceSynchronize());
  cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasScopy(blasHandle, matrix_shape, x_old, 1, x, 1));
  //cdpErrchk_blas(cublasScopy(matrix_shape, x_old, 1, x, 1));
  // 1. Initial residual r_0 = b - A*x_0
  cusparseSetStream(sparseHandle, stream); cdpErrchk_sparse(cusparseScsrmv(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrix_shape, matrix_shape, nnz_a,
                                  &negative_one, descrA, csr_valuesA, csr_row_ptr, csr_col_ind,
                                  x , &zero, r));
  //CUDA_CHECK_RETURN(cudaDeviceSynchronize());
  if(laplace_rank_deficient){  // compute (A + e^Te) * x instead
    rank_deficient_scaling = csrMatrixTrace(blasHandle, csr_valuesA, csr_col_ind, csr_row_ptr, matrix_shape, stream) / matrix_shape / matrix_shape / matrix_shape;
    cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasSasum(blasHandle, matrix_shape, x, 1, &temp));
    temp *= rank_deficient_scaling;
    cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasSaxpy(blasHandle, matrix_shape, &temp, gpuOne, 0, r, 1));
    //printf("new lapRankdef with scaling %f \n", rank_deficient_scaling);
  }
  cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasSaxpy(blasHandle, matrix_shape, &one, rhs, 1, r, 1));
  // 2. set rhat=r and p=r, compute rho = <r,r>
  cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasScopy(blasHandle, matrix_shape, r, 1, rhat, 1));
  cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasScopy(blasHandle, matrix_shape, r, 1, p, 1));
  cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasSdot(blasHandle, matrix_shape, r, 1, r, 1, &rho));
  // MAIN LOOP
  size_t i = 0;
  for (; i < max_it; i++) {
    //printf("CG step %d", i);
    if (residual_reset_bool){
      if ((i+1)%residual_reset_steps == 0){
        cusparseSetStream(sparseHandle, stream); cdpErrchk_sparse(cusparseScsrmv(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrix_shape, matrix_shape, nnz_a,
                                        &negative_one, descrA, csr_valuesA, csr_row_ptr, csr_col_ind,
                                        x , &zero, r));
        //CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        if(laplace_rank_deficient){  // compute (A + e^Te) * x instead
          //rank_deficient_scaling = 1. / csrMatrixTrace(blasHandle, csr_valuesA, csr_col_ind, csr_row_ptr, matrix_shape);
          //cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasSaxpy(blasHandle, matrix_shape, &rank_deficient_scaling, x, 1, r, 1));
          cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasSasum(blasHandle, matrix_shape, x, 1, &temp));
          temp *= rank_deficient_scaling;
          cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasSaxpy(blasHandle, matrix_shape, &temp, gpuOne, 0, r, 1));
        }
        cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasSaxpy(blasHandle, matrix_shape, &one, rhs, 1, r, 1));
        // 2. set rhat=r and p=r, compute rho = <r,r>
        cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasScopy(blasHandle, matrix_shape, r, 1, rhat, 1));
        cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasScopy(blasHandle, matrix_shape, r, 1, p, 1));
        cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasSdot(blasHandle, matrix_shape, r, 1, r, 1, &rho));
      }
    }
    // 3. compute rhat as A*p
    cusparseSetStream(sparseHandle, stream); cdpErrchk_sparse(cusparseScsrmv(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrix_shape, matrix_shape, nnz_a,
                                    &one, descrA, csr_valuesA, csr_row_ptr, csr_col_ind, p, &zero, rhat));
    if(laplace_rank_deficient){  // compute (A + e^Te) * p instead
      //cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasSaxpy(blasHandle, matrix_shape, &rank_deficient_scaling, p, 1, rhat, 1));
      cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasSasum(blasHandle, matrix_shape, p, 1, &temp));
      temp *= rank_deficient_scaling;
      cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasSaxpy(blasHandle, matrix_shape, &temp, gpuOne, 0, rhat, 1));
    }
    // 4. compute alpha = rho/<p,rhat>
    // printf("step 4\n");
    cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasSdot(blasHandle, matrix_shape, p, 1, rhat, 1, &temp));
    alpha = rho/temp;
    // 5.  x = x + alpha*p
    // printf("step 5\n");
    cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasSaxpy(blasHandle, matrix_shape, &alpha, p, 1, x, 1));
    // 6. r_new = r - alpha*rhat
    // printf("step 6\n");
    temp = -alpha;
    cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasSaxpy(blasHandle, matrix_shape, &temp, rhat, 1, r, 1));
    // 7. convergence check norm(Ax-b)  =~= r < tol
    // printf("step 7\n");
    cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasSnrm2(blasHandle, matrix_shape, r , 1, &temp));
    //printf("   residual %f\n", temp);
    if (temp < tol) {
      break;
    }
    // 8. rhoprev = rho; rho = <r,r>
    // printf("step 8\n");
    rhoprev = rho;
    cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasSdot(blasHandle, matrix_shape, r, 1, r, 1, &rho));
    // 9. beta = rho/rhoprev
    // printf("step 9\n");
    beta = rho/rhoprev;
    // 10. p = r + beta*p
    // printf("step 10\n");
    cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasSscal(blasHandle, matrix_shape, &beta, p, 1));
    cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasSaxpy(blasHandle, matrix_shape, &one, r, 1, p, 1));
  }
  if(temp>tol*100){
    printf("Unprecondition linear solve raised warning with resid: %f\n", temp);
    cudaMemset(warning, true, sizeof(bool));
  }
  cdpErrchk_sparse(cusparseDestroy(sparseHandle));
  cdpErrchk_blas(cublasDestroy(blasHandle));
  cdpErrchk(cudaFree(p));
  cdpErrchk(cudaFree(r));
  cdpErrchk(cudaFree(rhat));
  cdpErrchk(cudaFree(gpuOne));
}





// with preconditioner

__host__ void CgSolveLauncherDouble(double * csr_valuesA, double* ilu_csr_values, int* csr_row_ptr, int* csr_col_ind,
  const double* rhs, int nnz_a,  int matrix_shape, const double *x_old, const float* tol_gpu, int max_it,
  // float *s,  float *p, float *r, float *rhat, float *v, float *t,
  double * x,  bool laplace_rank_deficient, const int residual_reset_steps, const bool calculate_preconditioner,
  bool* warning, cudaStream_t stream)
{
  cusparseHandle_t  sparseHandle;
  cublasHandle_t    blasHandle;
  cdpErrchk_sparse(cusparseCreate(&sparseHandle));
  cdpErrchk_blas(cublasCreate(&blasHandle));
  // double * ichol_csr_values;
  // cdpErrchk(cudaMalloc((void**) &ichol_csr_values, nnz_a*sizeof(double)));
  // double * ichol_csr_values = csr_valuesA;
  cusparseMatDescr_t descrA, descrL, descrU;
  cdpErrchk_sparse(cusparseCreateMatDescr(&descrA));
  cdpErrchk_sparse(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
  cdpErrchk_sparse(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
  cdpErrchk_sparse(cusparseCreateMatDescr(&descrL));
  cdpErrchk_sparse(cusparseSetMatIndexBase(descrL, CUSPARSE_INDEX_BASE_ZERO));
  cdpErrchk_sparse(cusparseSetMatType(descrL, CUSPARSE_MATRIX_TYPE_GENERAL));
  cdpErrchk_sparse(cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER));
  cdpErrchk_sparse(cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_UNIT));
  cdpErrchk_sparse(cusparseCreateMatDescr(&descrU));
  cdpErrchk_sparse(cusparseSetMatIndexBase(descrU, CUSPARSE_INDEX_BASE_ZERO));
  cdpErrchk_sparse(cusparseSetMatType(descrU, CUSPARSE_MATRIX_TYPE_GENERAL));
  cdpErrchk_sparse(cusparseSetMatDiagType(descrU, CUSPARSE_DIAG_TYPE_NON_UNIT));
  cdpErrchk_sparse(cusparseSetMatFillMode(descrU, CUSPARSE_FILL_MODE_UPPER));
  csrsv2Info_t  infoL, infoU;
  cdpErrchk_sparse(cusparseCreateCsrsv2Info(&infoL));
  cdpErrchk_sparse(cusparseCreateCsrsv2Info(&infoU));
  int pBufferSizeL;
  int pBufferSizeU;
  int pBufferSize;
  void* pBuffer;
  cusparseDcsrsv2_bufferSize(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrix_shape, nnz_a,
    descrL, ilu_csr_values, csr_row_ptr, csr_col_ind, infoL, &pBufferSizeL);
  cusparseDcsrsv2_bufferSize(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrix_shape, nnz_a,
    descrU, ilu_csr_values, csr_row_ptr, csr_col_ind, infoU, &pBufferSizeU);
  cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;
  // cusparseOperation_t transpose_op;
  // int * transpose_csr_col_ind;
  // int * transpose_csr_row_ptr;
  // double * transpose_csr_values;
  if (calculate_preconditioner){
    cdpErrchk_blas(cublasDcopy(blasHandle, nnz_a, csr_valuesA, 1, ilu_csr_values, 1));
    int structural_zero;
    int numerical_zero;
    csrilu02Info_t infoA;
    cdpErrchk_sparse(cusparseCreateCsrilu02Info(&infoA));
    int pBufferSizeA;
    cdpErrchk_sparse(cusparseDcsrilu02_bufferSize(sparseHandle, matrix_shape, nnz_a, descrA,
                      ilu_csr_values, csr_row_ptr, csr_col_ind, infoA, &pBufferSizeA));
    pBufferSize = max(pBufferSizeA,max(pBufferSizeL,pBufferSizeU));
    cdpErrchk(cudaMalloc((void**)&pBuffer, pBufferSize));
    cdpErrchk_sparse(cusparseDcsrilu02_analysis(sparseHandle, matrix_shape, nnz_a, descrA,
                      ilu_csr_values, csr_row_ptr, csr_col_ind, infoA, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer));
    status = cusparseXcsrilu02_zeroPivot(sparseHandle, infoA, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status){
       printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
    }
    //print_array<<<1,1,0,0>>>(ichol_csr_values, nnz_a);
    cdpErrchk_sparse(cusparseDcsrilu02(sparseHandle, matrix_shape, nnz_a, descrA,
                      ilu_csr_values, csr_row_ptr, csr_col_ind, infoA, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer));
    status = cusparseXcsrilu02_zeroPivot(sparseHandle, infoA, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status){
      printf("L(%d,%d) is zero\n", numerical_zero, numerical_zero);
    }
    cusparseDestroyCsrilu02Info(infoA);
  }
  else {
    pBufferSize = max(pBufferSizeL,pBufferSizeU);
    cdpErrchk(cudaMalloc((void**)&pBuffer, pBufferSize));
  }
  //printf("solve ana L\n");
  cdpErrchk_sparse(cusparseDcsrsv2_analysis(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    matrix_shape, nnz_a, descrL, ilu_csr_values, csr_row_ptr, csr_col_ind, infoL, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer));
  cdpErrchk_sparse(cusparseDcsrsv2_analysis(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    matrix_shape, nnz_a, descrU, ilu_csr_values, csr_row_ptr, csr_col_ind, infoU, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer));
  //printf("allocating helper\n");
  // 0. Set all helper variables to zero, Initialise handles
  double rho = 1., rhoprev = 1., alpha = 1., beta = 1., one = 1.;
  double negative_one = -1.;
  double zero = 0., temp = 0;
  double rank_deficient_scaling = 1;
  double *p, *r, *t, *z, *rhat;
  cdpErrchk(cudaMalloc((void**)&p , matrix_shape*sizeof(double)));
  cdpErrchk(cudaMalloc((void**)&r , matrix_shape*sizeof(double)));
  cdpErrchk(cudaMalloc((void**)&t , matrix_shape*sizeof(double)));
  cdpErrchk(cudaMalloc((void**)&z , matrix_shape*sizeof(double)));
  cdpErrchk(cudaMalloc((void**)&rhat , matrix_shape*sizeof(double)));
  double * gpuOne;
  cdpErrchk(cudaMalloc((void**)&gpuOne, sizeof(double)));
  cdpErrchk(cudaMemcpy(gpuOne, &one, sizeof(double), cudaMemcpyHostToDevice));
  //printf("rho %f   rhoprev %f   alpha %f   omega %f    ones %f\n", rho, rhoprev, alpha, omega,one);
  float tol;
  cdpErrchk(cudaMemcpy(&tol, tol_gpu, sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK_RETURN(cudaStreamSynchronize(stream));
  cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasDcopy(blasHandle, matrix_shape, x_old, 1, x, 1));
  //cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasDcopy(matrix_shape, x_old, 1, x, 1));
  //printf("starting\n");
  // 1. Initial residual r_0 = b - A*x_0
  cusparseSetStream(sparseHandle, stream);
  cdpErrchk_sparse(cusparseDcsrmv(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrix_shape, matrix_shape, nnz_a,
                                  &negative_one, descrA, csr_valuesA, csr_row_ptr, csr_col_ind,
                                  x , &zero, r));
  cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasDaxpy(blasHandle, matrix_shape, &one, rhs, 1, r, 1));
  // 2. Compute z_0 = L^-T L^-1 r_0
  cusparseSetStream(sparseHandle, stream);
  cdpErrchk_sparse(cusparseDcsrsv2_solve(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrix_shape,  nnz_a, &one, descrL,
                                        ilu_csr_values, csr_row_ptr, csr_col_ind, infoL, r, t, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer));
  cdpErrchk_sparse(cusparseDcsrsv2_solve(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrix_shape,  nnz_a, &one, descrU,
                                        ilu_csr_values, csr_row_ptr, csr_col_ind, infoU, t, z, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer));
  // 3. set rhat=r and p=r, compute rho = <r,r>
  cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasDcopy(blasHandle, matrix_shape, z, 1, p, 1));
  // compute rho = <r,z>
  cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasDdot(blasHandle, matrix_shape, r, 1, z, 1, &rho));
  //printf("rho %f\n",rho );
  // MAIN LOOP
  size_t i = 0;
  for (; i < max_it; i++) {
    if ((i+1)%500 == 0){
      cusparseSetStream(sparseHandle, stream);
      cdpErrchk_sparse(cusparseDcsrmv(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrix_shape, matrix_shape, nnz_a,
                                      &negative_one, descrA, csr_valuesA, csr_row_ptr, csr_col_ind,
                                      x , &zero, r));
      cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasDaxpy(blasHandle, matrix_shape, &one, rhs, 1, r, 1));
      // 2. Compute z_0 = L^-T L^-1 r_0
      cusparseSetStream(sparseHandle, stream);
      cdpErrchk_sparse(cusparseDcsrsv2_solve(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrix_shape, nnz_a, &one, descrL,
                                            ilu_csr_values, csr_row_ptr, csr_col_ind, infoL, r, t, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer));
      cdpErrchk_sparse(cusparseDcsrsv2_solve(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrix_shape, nnz_a, &one, descrU,
                                            ilu_csr_values, csr_row_ptr, csr_col_ind, infoU, t, z, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer));
      cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasDcopy(blasHandle, matrix_shape, z, 1, p, 1));
      // compute rho = <r,z>
      cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasDdot(blasHandle, matrix_shape, r, 1, z, 1, &rho));
    }
    cusparseSetStream(sparseHandle, stream);
    cdpErrchk_sparse(cusparseDcsrmv(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrix_shape, matrix_shape, nnz_a,
                                    &one, descrA, csr_valuesA, csr_row_ptr, csr_col_ind,
                                    p , &zero, rhat));
    // temp = <p, rhat>  and alpha = rho/temp
    cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasDdot(blasHandle, matrix_shape, p, 1, rhat, 1, &temp));
    alpha = rho/temp;
    // x = x + alpha*p
    cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasDaxpy(blasHandle, matrix_shape, &alpha, p, 1, x, 1));
    // r_new = r - alpha*rhat
    temp = -alpha;
    cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasDaxpy(blasHandle, matrix_shape, &temp, rhat, 1, r, 1));
    // convergence check norm(Ax-b)  =~= r < tol
    cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasDnrm2(blasHandle, matrix_shape, r , 1, &temp));
    // printf("%f\n", temp);
    if (temp < tol) {
      break;
    }
    //  Compute z = L^-T L^-1 r
    cusparseSetStream(sparseHandle, stream);
    cdpErrchk_sparse(cusparseDcsrsv2_solve(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrix_shape, nnz_a, &one, descrL,
                                          ilu_csr_values, csr_row_ptr, csr_col_ind, infoL, r, t, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer));
    cdpErrchk_sparse(cusparseDcsrsv2_solve(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrix_shape, nnz_a, &one, descrU,
                                          ilu_csr_values, csr_row_ptr, csr_col_ind, infoU, t, z, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer));
    // printf("zn\n");
    // print_array<<<1,1,0,0>>>(z, matrix_shape);
    // CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    // beta  = rho/rhoprev   with rho = <r,z> and rhoprev = <r_old, z_old>
    rhoprev = rho;
    cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasDdot(blasHandle, matrix_shape, r, 1, z, 1, &rho));
    beta = rho/rhoprev;
    // p = z + beta*p
    cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasDscal(blasHandle, matrix_shape, &beta, p, 1));
    cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasDaxpy(blasHandle, matrix_shape, &one, z, 1, p, 1));
  }
  if(temp>tol*100){
    printf("Linear solve raised warning");
    cudaMemset(warning, true, sizeof(bool));
  }
  //printf("final prec %f after iter %d\n", temp, i);
  cdpErrchk_sparse(cusparseDestroy(sparseHandle));
  cdpErrchk_blas(cublasDestroy(blasHandle));
  cdpErrchk(cudaFree(p));
  cdpErrchk(cudaFree(r));
  cdpErrchk(cudaFree(t));
  cdpErrchk(cudaFree(z));
  cdpErrchk(cudaFree(rhat));
  cdpErrchk(cudaFree(pBuffer));
  //cdpErrchk(cudaFree(ichol_csr_values));
  cdpErrchk(cudaFree(gpuOne));
  cusparseDestroyCsrsv2Info(infoL);
  cusparseDestroyCsrsv2Info(infoU);
  cusparseDestroyMatDescr(descrA);
  cusparseDestroyMatDescr(descrL);
  cusparseDestroyMatDescr(descrU);
}