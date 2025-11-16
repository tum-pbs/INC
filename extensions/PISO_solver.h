#pragma once

#ifndef _INCLUDE_PISO_SOLVER
#define _INCLUDE_PISO_SOLVER

#include "domain_structs.h"

#include "PISO_multiblock_cuda.h"

void CorrectAdvectiveBoundaries(std::shared_ptr<Domain> domain, std::vector<std::shared_ptr<Boundary>> bounds);
index_t PISOstep(std::shared_ptr<Domain> domain, const index_t steps, const index_t correctorSteps, const torch::Tensor &timeStep, const bool advect);


#endif //_INCLUDE_PISO_SOLVER