
// --- Old divergence calculations, don't work for transformations ---

template<typename scalar_t>
__device__ scalar_t getVelocityAtWithBounds(const I4 pos, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain){
	// read value at location with support for 1-cell ghost layer
	// any position outside the domain will be treated as being on the ghost layer
	// only one coordinate may be outside the domain (corner ghost cells are not supported)
	
	
	const index_t flatPos = flattenIndex(pos, block);
	I4 tempPos = pos;
	
	for(index_t dim=0; dim<domain.numDims; ++dim)
	{
		index_t bound = dim*2;
		
		if(pos.a[dim]<0){// lower boundary
			switch(block.boundaries[bound].type){
			case BoundaryType::VALUE:
				tempPos.a[dim] = 0;
				return 2*block.boundaries[bound].sdb.velocity.a[tempPos.w] - block.velocity[flattenIndex(tempPos, block)]; //block.velocity[flattenIndex(pos, block)]; 
				break;
			case BoundaryType::DIRICHLET_VARYING:
			{
				tempPos.a[dim] = 0;
				scalar_t boundaryVelocity = block.boundaries[bound].vdb.velocity[flattenIndex(tempPos, block.boundaries[bound].vdb.stride)];
				return 2*boundaryVelocity - block.velocity[flattenIndex(tempPos, block)]; 
				break;
			}
			case BoundaryType::GRADIENT:
				// compute flux only from center cell?
				// velN = velC - grad*distance
				return block.velocity[flattenIndex(tempPos, block)] - block.boundaries[bound].snb.boundaryGradient.a[tempPos.w]; //* distance
				break;
			case BoundaryType::CONNECTED_GRID:
			{
				const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + block.boundaries[bound].cb.connectedGridIndex;
				//I4 otherPos = computeConnectedPos(tempPos, dim, &block.boundaries[bound].cb, domain);
				// the requested component is not necessarily the boundary axis here
				// boundaryAxis = bound>>1; connected to axes[0]
				// requestedAxis = pos.w; connected to axes[?]
				// pos.W == bA -> axes[0], pos.w == (bA+1)%dims -> axes[1], pos.w == (bA+2)%dims -> axes[2]
				//const index_t connectionIndex = posMod(tempPos.w-dim, domain.numDims); //needs positive mod
				//otherPos.w = block.boundaries[bound].cb.axes.a[connectionIndex]>>1;
				I4 otherPos = computeConnectedPosWithChannel(tempPos, dim, &block.boundaries[bound].cb, domain);
				const index_t connectionIndex = posMod(tempPos.w-dim, domain.numDims);
				
				scalar_t vel = p_connectedBlock->velocity[flattenIndex(otherPos, p_connectedBlock)];
				// TODO if the connection goes upper to upper or lower to lower, the velocity has to be inverted
				// (if it goes upper->lower or lower->upper it should be fine)
				if((connectionIndex==0 && (block.boundaries[bound].cb.axes.a[0]&1)==0) || (connectionIndex!=0 && (block.boundaries[bound].cb.axes.a[connectionIndex]&1)!=0)){
					vel = -vel;
				}
				return vel;
				break;
			}
			case BoundaryType::PERIODIC:
				// compute flux to cell on other side
				// special case of connection to another block
				tempPos.a[dim] = block.size.a[dim] - 1;
				return block.velocity[flattenIndex(tempPos, block)];
				break;
			default:
				return 0;
				break;
			}
		}
		
		bound = dim*2 + 1;
		
		if(pos.a[dim]>=block.size.a[dim]){// upper boundary
			switch(block.boundaries[bound].type){
			case BoundaryType::VALUE:
				tempPos.a[dim] = block.size.a[dim] - 1;
				return 2*block.boundaries[bound].sdb.velocity.a[tempPos.w] - block.velocity[flattenIndex(tempPos, block)]; 
				break;
			case BoundaryType::DIRICHLET_VARYING:
			{
				tempPos.a[dim] = 0;
				scalar_t boundaryVelocity = block.boundaries[bound].vdb.velocity[flattenIndex(tempPos, block.boundaries[bound].vdb.stride)];
				tempPos.a[dim] = block.size.a[dim] - 1;
				return 2*boundaryVelocity - block.velocity[flattenIndex(tempPos, block)]; 
				break;
			}
			case BoundaryType::GRADIENT:
				// compute flux only from center cell?
				// velP = - celC - grad*distance
				return -block.velocity[flattenIndex(tempPos, block)] - block.boundaries[bound].snb.boundaryGradient.a[tempPos.w]; //* distance
				break;
			case BoundaryType::CONNECTED_GRID:
			{
				const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + block.boundaries[bound].cb.connectedGridIndex;
				I4 otherPos = computeConnectedPosWithChannel(tempPos, dim, &block.boundaries[bound].cb, domain);
				const index_t connectionIndex = posMod(tempPos.w-dim, domain.numDims);
				
				scalar_t vel = p_connectedBlock->velocity[flattenIndex(otherPos, p_connectedBlock)];
				if((connectionIndex==0 && (block.boundaries[bound].cb.axes.a[0]&1)!=0) || (connectionIndex!=0 && (block.boundaries[bound].cb.axes.a[connectionIndex]&1)!=0)){
					vel = -vel;
				}
				return vel;
				break;
			}
			case BoundaryType::PERIODIC:
				// compute flux to cell on other side
				// special case of connection to another block
				tempPos.a[dim] = 0;
				return block.velocity[flattenIndex(tempPos, block)];
				break;
			default:
				return 0;
				break;
			}
		}
	}
	return block.velocity[flattenIndex(tempPos, block)];
}

template<typename scalar_t>
__device__ scalar_t getGlobalVelocityAtWithBounds(const I4 pos, const scalar_t *globalVectorField, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain){
	// read value at location with support for 1-cell ghost layer
	// any position outside the domain will be treated as being on the ghost layer
	// only one coordinate may be outside the domain (corner ghost cells are not supported)
	
	// globalVectorField is a combined field with components still in the local coordinate system
	
	const index_t flatPos = flattenIndex(pos, block);
	I4 tempPos = pos;
	
	for(index_t dim=0; dim<domain.numDims; ++dim)
	{
		index_t bound = dim*2;
		
		if(pos.a[dim]<0){// lower boundary
			switch(block.boundaries[bound].type){
			case BoundaryType::VALUE:
				tempPos.a[dim] = 0;
				return 2*block.boundaries[bound].sdb.velocity.a[tempPos.w] - globalVectorField[flattenIndexGlobal(tempPos, block, domain)]; //block.velocity[flattenIndex(pos, block)]; 
				break;
			case BoundaryType::DIRICHLET_VARYING:
			{
				tempPos.a[dim] = 0;
				scalar_t boundaryVelocity = block.boundaries[bound].vdb.velocity[flattenIndex(tempPos, block.boundaries[bound].vdb.stride)];
				return 2*boundaryVelocity - globalVectorField[flattenIndexGlobal(tempPos, block, domain)]; 
				break;
			}
			case BoundaryType::GRADIENT:
				// compute flux only from center cell?
				// velN = velC - grad*distance
				return globalVectorField[flattenIndexGlobal(tempPos, block, domain)] - block.boundaries[bound].snb.boundaryGradient.a[tempPos.w]; //* distance
				break;
			case BoundaryType::CONNECTED_GRID:
			{
				const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + block.boundaries[bound].cb.connectedGridIndex;
				I4 otherPos = computeConnectedPosWithChannel(tempPos, dim, &block.boundaries[bound].cb, domain);
				const index_t connectionIndex = posMod(tempPos.w-dim, domain.numDims);
				
				scalar_t vel = globalVectorField[flattenIndexGlobal(otherPos, p_connectedBlock, domain)];
				// TODO if the connection goes upper to upper or lower to lower, the velocity has to be inverted
				// (if it goes upper->lower or lower->upper it should be fine)
				if((connectionIndex==0 && (block.boundaries[bound].cb.axes.a[0]&1)==0) || (connectionIndex!=0 && (block.boundaries[bound].cb.axes.a[connectionIndex]&1)!=0)){
					vel = -vel;
				}
				return vel;
				break;
			}
			case BoundaryType::PERIODIC:
				// compute flux to cell on other side
				// special case of connection to another block
				tempPos.a[dim] = block.size.a[dim] - 1;
				return globalVectorField[flattenIndexGlobal(tempPos, block, domain)];
				break;
			default:
				return 0;
				break;
			}
		}
		
		bound = dim*2 + 1;
		
		if(pos.a[dim]>=block.size.a[dim]){// upper boundary
			switch(block.boundaries[bound].type){
			case BoundaryType::VALUE:
				tempPos.a[dim] = block.size.a[dim] - 1;
				return 2*block.boundaries[bound].sdb.velocity.a[tempPos.w] - globalVectorField[flattenIndexGlobal(tempPos, block, domain)]; 
				break;
			case BoundaryType::DIRICHLET_VARYING:
			{
				tempPos.a[dim] = 0;
				scalar_t boundaryVelocity = block.boundaries[bound].vdb.velocity[flattenIndex(tempPos, block.boundaries[bound].vdb.stride)];
				tempPos.a[dim] = block.size.a[dim] - 1;
				return 2*boundaryVelocity - globalVectorField[flattenIndexGlobal(tempPos, block, domain)]; 
				break;
			}
			case BoundaryType::GRADIENT:
				// compute flux only from center cell?
				// velP = - celC - grad*distance
				return -globalVectorField[flattenIndexGlobal(tempPos, block, domain)] - block.boundaries[bound].snb.boundaryGradient.a[tempPos.w]; //* distance
				break;
			case BoundaryType::CONNECTED_GRID:
			{
				const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + block.boundaries[bound].cb.connectedGridIndex;
				I4 otherPos = computeConnectedPosWithChannel(tempPos, dim, &block.boundaries[bound].cb, domain);
				const index_t connectionIndex = posMod(tempPos.w-dim, domain.numDims);
				
				scalar_t vel = globalVectorField[flattenIndexGlobal(otherPos, p_connectedBlock, domain)];
				if((connectionIndex==0 && (block.boundaries[bound].cb.axes.a[0]&1)!=0) || (connectionIndex!=0 && (block.boundaries[bound].cb.axes.a[connectionIndex]&1)!=0)){
					vel = -vel;
				}
				return vel;
				break;
			}
			case BoundaryType::PERIODIC:
				// compute flux to cell on other side
				// special case of connection to another block
				tempPos.a[dim] = 0;
				return globalVectorField[flattenIndexGlobal(tempPos, block, domain)];
				break;
			default:
				return 0;
				break;
			}
		}
	}
	return globalVectorField[flattenIndexGlobal(tempPos, block, domain)];
}

template <typename scalar_t>
__global__ void compute_pressure_RHS_divergence(DomainGPU<scalar_t> *p_domain, const scalar_t timeStep,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks){
	
	// divergence of colocated vector field
	// using central differences
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		
		const I4 pos = unflattenIndex(flatPos, s_block);
			
		scalar_t div = 0;
		for(index_t dim=0;dim<s_domain.numDims;++dim){
			//dim=1;
			I4 tempPos = pos;
			tempPos.w = dim;
			tempPos.a[dim] = pos.a[dim]-1;
			// this has to work on the global pressureRHS, not on block.velocity
			const scalar_t valN = getGlobalVelocityAtWithBounds(tempPos, s_domain.pressureRHS, s_block, s_domain);
			tempPos.a[dim] = pos.a[dim]+1;
			const scalar_t valP = getGlobalVelocityAtWithBounds(tempPos, s_domain.pressureRHS, s_block, s_domain);
			div += (valP - valN)*0.5;
			//break; //debug
		}
		
		s_domain.pressureRHSdiv[flatPos + s_block.globalOffset] = div/timeStep;
	)
}

#ifdef WITH_GRAD
template <typename scalar_t>
__global__ void compute_pressure_RHS_divergence_GRAD(DomainGPU<scalar_t> *p_domain, const scalar_t timeStep,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks){
	
	// divergence of colocated vector field
	// using central differences
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		index_t flatPosGlobal = flattenIndexGlobal(pos, s_block, s_domain);

		scalar_t divGrad = s_domain.pressureRHSdiv_grad[flatPosGlobal]/timeStep;

		// scatter version
		for(int dim=0;dim<s_domain.numDims;++dim){
			
			//I4 tempPos = pos;
			//tempPos.w = dim;
			
			// w.r.t. pressureRHS. cf. getGlobalVelocityAtWithBounds for boundary handling.
			scalar_t fac = -0.5;
			const BlockGPU<scalar_t> *p_blockN = &s_block;
			I4 tempPosN = pos;
			tempPosN.w = dim;
			if(pos.a[dim]==0){
				switch(s_block.boundaries[dim*2].type){
					case BoundaryType::DIRICHLET:
					case BoundaryType::DIRICHLET_VARYING:
						fac *= -1;
						break;
					case BoundaryType::NEUMANN:
						break;
					case BoundaryType::CONNECTED_GRID:
					{
						const index_t bound = dim*2;
						p_blockN = s_domain.blocks + s_block.boundaries[bound].cb.connectedGridIndex;
						tempPosN = computeConnectedPos(tempPosN, dim, &s_block.boundaries[bound].cb, s_domain);
						//tempPos.w = 0; // set by computeConnectedPos to connected axis
						//tempFlatPosGlobal = flattenIndex(tempPos, p_connectedBlock) + p_connectedBlock->globalOffset;
						//TODO
						break;
					}
					case BoundaryType::PERIODIC:
						tempPosN.a[dim] = s_block.size.a[dim]-1;
						break;
					default:
						break;
				}
			}else{
				tempPosN.a[dim] = pos.a[dim]-1;
			}
			scalar_t *p_pressureRHSGrad = s_domain.pressureRHS_grad + flattenIndexGlobal(tempPosN, p_blockN, s_domain);
			atomicAdd(p_pressureRHSGrad, fac*divGrad);
			
			fac = 0.5;
			const BlockGPU<scalar_t> *p_blockP = &s_block;
			I4 tempPosP = pos;
			tempPosP.w = dim;
			if(pos.a[dim]==s_block.size.a[dim]-1){
				switch(s_block.boundaries[dim*2+1].type){
					case BoundaryType::DIRICHLET:
					case BoundaryType::DIRICHLET_VARYING:
					case BoundaryType::NEUMANN:
						fac *= -1;
						tempPosP.a[dim] = pos.a[dim];
						break;
					case BoundaryType::CONNECTED_GRID:
					{
						const index_t bound = dim*2+1;
						p_blockP = s_domain.blocks + s_block.boundaries[bound].cb.connectedGridIndex;
						tempPosP = computeConnectedPos(tempPosP, dim, &s_block.boundaries[bound].cb, s_domain);
						break;
					}
					case BoundaryType::PERIODIC:
						tempPosP.a[dim] = 0;
						break;
					default:
						break;
				}
			}else{
				tempPosP.a[dim] = pos.a[dim]+1;
			}
			p_pressureRHSGrad = s_domain.pressureRHS_grad + flattenIndexGlobal(tempPosP, p_blockP, s_domain);
			atomicAdd(p_pressureRHSGrad, fac*divGrad);
		}

		
	)
}

#endif //WITH_GRAD


template <typename scalar_t>
__global__ void k_compute_velocity_divergence(DomainGPU<scalar_t> *p_domain, scalar_t *divergence, //const scalar_t timeStep,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks){
	
	// divergence of colocated vector field
	// using central differences
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		
		const I4 pos = unflattenIndex(flatPos, s_block);
			
		scalar_t div = 0;
		for(index_t dim=0;dim<s_domain.numDims;++dim){
			//dim=1;
			I4 tempPos = pos;
			tempPos.w = dim;
			tempPos.a[dim] = pos.a[dim]-1;
			// this has to work on the global pressureRHS, not on block.velocity
			const scalar_t valN = getVelocityAtWithBounds(tempPos, s_block, s_domain);
			tempPos.a[dim] = pos.a[dim]+1;
			const scalar_t valP = getVelocityAtWithBounds(tempPos, s_block, s_domain);
			div += (valP - valN)*0.5;
			//break; //debug
		}
		
		divergence[flatPos + s_block.globalOffset] = div;
	)
}

// --- Old velocity corrector tests ---

// DO NOT USE, this is a smoothing kernel
template <typename scalar_t>
__global__ void PISO_update_velocity_v2(DomainGPU<scalar_t> *p_domain, const scalar_t timeStep,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks){
	//TODO: non-orthogonal transformations
	//vel.x = pressureRHS.x - inv(A)*gradX(pressure)
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		const index_t flatPosGlobal = s_block.globalOffset + flatPos;
		
		const scalar_t ra = 1/s_domain.Adiag[flatPosGlobal];
		
		//scalar_t pressureGrad[3];
		//tempGetPressureGradientDimSwitch<scalar_t>(s_block, pos, s_domain, pressureGrad);
		const scalar_t p = s_block.pressure[flattenIndex(pos, s_block)];//getPressureAtWithBounds(pos, block, domain)
		
		
		for(int dim=0;dim<s_domain.numDims;++dim){
			scalar_t velUpdate = 0;
			
			I4 tempPos = pos;
			tempPos.w = dim;
			const int flatCompPosGlobal = flattenIndexGlobal(tempPos, s_block, s_domain);
			const scalar_t hbyA = s_domain.pressureRHS[flatCompPosGlobal];
			const scalar_t t = getTransformMetricOrthogonalDimSwitch<scalar_t>(tempPos, &s_block, s_domain.numDims);
			
			//lower boundary
			index_t bound = dim*2;
			if(pos.a[dim]!=0 || !isEmptyBound(bound, s_block.boundaries)){
				tempPos = pos;
				tempPos.w = dim;
				index_t tempFlatPosGlobal = 0;
				scalar_t pL = 0;
				scalar_t hbyAL = 0;
				scalar_t tL = 1; //transform metric
				if(pos.a[dim]==0 && s_block.boundaries[bound].type==BoundaryType::CONNECTED_GRID){
					const BlockGPU<scalar_t> *p_connectedBlock = s_domain.blocks + s_block.boundaries[bound].cb.connectedGridIndex;
					tempPos = computeConnectedPosWithChannel(tempPos, dim, &s_block.boundaries[bound].cb, s_domain);
					hbyAL = s_domain.pressureRHS[flattenIndexGlobal(tempPos, p_connectedBlock, s_domain)];
					tL = getTransformMetricOrthogonalDimSwitch<scalar_t>(tempPos, p_connectedBlock, s_domain.numDims);
					tempPos.w = 0;
					const index_t blockFlatIdx = flattenIndex(tempPos, p_connectedBlock);
					pL = p_connectedBlock->pressure[blockFlatIdx];
					tempFlatPosGlobal = blockFlatIdx + p_connectedBlock->globalOffset;
				}else {
					if(pos.a[dim]==0 && s_block.boundaries[bound].type==BoundaryType::PERIODIC){
						tempPos.a[dim] = s_block.size.a[dim]-1;
					}else{
						tempPos.a[dim] = pos.a[dim]-1;
					}
					hbyAL = s_domain.pressureRHS[flattenIndexGlobal(tempPos, s_block, s_domain)];
					tL = getTransformMetricOrthogonalDimSwitch<scalar_t>(tempPos, &s_block, s_domain.numDims);
					tempPos.w = 0;
					const index_t blockFlatIdx = flattenIndex(tempPos, s_block);
					pL = s_block.pressure[blockFlatIdx];
					tempFlatPosGlobal = blockFlatIdx + s_block.globalOffset;
				}
				const scalar_t raL = 1 /s_domain.Adiag[tempFlatPosGlobal];
				//const scalar_t coefficient = static_cast<scalar_t>(0.25)*(ra + raL);
				velUpdate += (hbyA + hbyAL - (ra + raL)*((t+tL)*static_cast<scalar_t>(0.5))*(p - pL) * timeStep) * static_cast<scalar_t>(0.25);
				//velUpdate += (hbyA + hbyAL - (t*ra + tL*raL)*(p - pL)) * static_cast<scalar_t>(0.25);
			}
			/*else{ //dim==0 && empty bound
				if(s_block.boundaries[bound].type==BoundaryType::DIRICHLET || s_block.boundaries[bound].type==BoundaryType::DIRICHLET_VARYING){
					//remove excess boundary coefficients for 0 gadient boundary
					//rowValues[0] += ra*0.5f;
				}
			}*/
			//upper boundary
			++bound;
			if(pos.a[dim]!=s_block.size.a[dim]-1 || !isEmptyBound(bound, s_block.boundaries)){
				tempPos = pos;
				tempPos.w = dim;
				index_t tempFlatPosGlobal = 0;
				scalar_t pU = 0;
				scalar_t hbyAU = 0;
				scalar_t tU = 1; //transform metric
				if(pos.a[dim]==s_block.size.a[dim]-1 && s_block.boundaries[bound].type==BoundaryType::CONNECTED_GRID){
					const BlockGPU<scalar_t> *p_connectedBlock = s_domain.blocks + s_block.boundaries[bound].cb.connectedGridIndex;
					tempPos = computeConnectedPosWithChannel(tempPos, dim, &s_block.boundaries[bound].cb, s_domain);
					hbyAU = s_domain.pressureRHS[flattenIndexGlobal(tempPos, p_connectedBlock, s_domain)];
					tU = getTransformMetricOrthogonalDimSwitch<scalar_t>(tempPos, p_connectedBlock, s_domain.numDims);
					tempPos.w = 0;
					const index_t blockFlatIdx = flattenIndex(tempPos, p_connectedBlock);
					pU = p_connectedBlock->pressure[blockFlatIdx];
					tempFlatPosGlobal = blockFlatIdx + p_connectedBlock->globalOffset;
				}else {
					if(pos.a[dim]==s_block.size.a[dim]-1 && s_block.boundaries[bound].type==BoundaryType::PERIODIC){
						tempPos.a[dim] = 0;
					}else{
						tempPos.a[dim] = pos.a[dim]+1;
					}
					hbyAU = s_domain.pressureRHS[flattenIndexGlobal(tempPos, s_block, s_domain)];
					tU = getTransformMetricOrthogonalDimSwitch<scalar_t>(tempPos, &s_block, s_domain.numDims);
					tempPos.w = 0;
					const index_t blockFlatIdx = flattenIndex(tempPos, s_block);
					pU = s_block.pressure[blockFlatIdx];
					tempFlatPosGlobal = blockFlatIdx + s_block.globalOffset;
				}
				const scalar_t raU = 1 /s_domain.Adiag[tempFlatPosGlobal];
				//const scalar_t coefficient = static_cast<scalar_t>(0.25)*(ra + raU);
				//velUpdate -= coefficient*(pU - p);
				velUpdate += (hbyA + hbyAU - (ra + raU)*((t+tU)*static_cast<scalar_t>(0.5))*(pU - p) * timeStep) * static_cast<scalar_t>(0.25);
				//velUpdate += (hbyA + hbyAU - (t*ra + tU*raU)*(p - pU)) * static_cast<scalar_t>(0.25);
			}
			
			s_domain.velocityResult[flatCompPosGlobal] = velUpdate;
		}
	)
}

// DO NOT USE, this is a smoothing kernel
template <typename scalar_t>
__global__ void PISO_update_velocity_v3(DomainGPU<scalar_t> *p_domain, const scalar_t timeStep,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks){
	//TODO: non-orthogonal transformations
	//vel.x = pressureRHS.x - inv(A)*gradX(pressure)
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		const index_t flatPosGlobal = s_block.globalOffset + flatPos;
		
		const scalar_t ra = 1/s_domain.Adiag[flatPosGlobal];
		
		//scalar_t pressureGrad[3];
		//tempGetPressureGradientDimSwitch<scalar_t>(s_block, pos, s_domain, pressureGrad);
		const scalar_t p = s_block.pressure[flattenIndex(pos, s_block)];//getPressureAtWithBounds(pos, block, domain)
		
		
		for(int dim=0;dim<s_domain.numDims;++dim){
			scalar_t velUpdate = 0;
			
			I4 tempPos = pos;
			tempPos.w = dim;
			const int flatCompPosGlobal = flattenIndexGlobal(tempPos, s_block, s_domain);
			const scalar_t hbyA = s_domain.pressureRHS[flatCompPosGlobal];
			const scalar_t t = getTransformMetricOrthogonalDimSwitch<scalar_t>(tempPos, &s_block, s_domain.numDims);
			
			//lower boundary
			index_t bound = dim*2;
			if(pos.a[dim]!=0 || !isEmptyBound(bound, s_block.boundaries)){
				tempPos = pos;
				tempPos.w = dim;
				index_t tempFlatPosGlobal = 0;
				scalar_t pL = 0;
				scalar_t hbyAL = 0;
				scalar_t tL = 1; //transform metric
				if(pos.a[dim]==0 && s_block.boundaries[bound].type==BoundaryType::CONNECTED_GRID){
					const BlockGPU<scalar_t> *p_connectedBlock = s_domain.blocks + s_block.boundaries[bound].cb.connectedGridIndex;
					tempPos = computeConnectedPosWithChannel(tempPos, dim, &s_block.boundaries[bound].cb, s_domain);
					hbyAL = s_domain.pressureRHS[flattenIndexGlobal(tempPos, p_connectedBlock, s_domain)];
					tL = getTransformMetricOrthogonalDimSwitch<scalar_t>(tempPos, p_connectedBlock, s_domain.numDims);
					tempPos.w = 0;
					const index_t blockFlatIdx = flattenIndex(tempPos, p_connectedBlock);
					pL = p_connectedBlock->pressure[blockFlatIdx];
					tempFlatPosGlobal = blockFlatIdx + p_connectedBlock->globalOffset;
				}else {
					if(pos.a[dim]==0 && s_block.boundaries[bound].type==BoundaryType::PERIODIC){
						tempPos.a[dim] = s_block.size.a[dim]-1;
					}else{
						tempPos.a[dim] = pos.a[dim]-1;
					}
					hbyAL = s_domain.pressureRHS[flattenIndexGlobal(tempPos, s_block, s_domain)];
					tL = getTransformMetricOrthogonalDimSwitch<scalar_t>(tempPos, &s_block, s_domain.numDims);
					tempPos.w = 0;
					const index_t blockFlatIdx = flattenIndex(tempPos, s_block);
					pL = s_block.pressure[blockFlatIdx];
					tempFlatPosGlobal = blockFlatIdx + s_block.globalOffset;
				}
				const scalar_t raL = 1 /s_domain.Adiag[tempFlatPosGlobal];
				//const scalar_t coefficient = static_cast<scalar_t>(0.25)*(ra + raL);
				//velUpdate += (hbyA + hbyAL - (ra + raL)*((t+tL)*static_cast<scalar_t>(0.5))*(p - pL)) * static_cast<scalar_t>(0.25);
				velUpdate += (hbyA + hbyAL - (t*ra + tL*raL)*(p - pL) * timeStep) * static_cast<scalar_t>(0.25);
			}
			/*else{ //dim==0 && empty bound
				if(s_block.boundaries[bound].type==BoundaryType::DIRICHLET || s_block.boundaries[bound].type==BoundaryType::DIRICHLET_VARYING){
					//remove excess boundary coefficients for 0 gadient boundary
					//rowValues[0] += ra*0.5f;
				}
			}*/
			//upper boundary
			++bound;
			if(pos.a[dim]!=s_block.size.a[dim]-1 || !isEmptyBound(bound, s_block.boundaries)){
				tempPos = pos;
				tempPos.w = dim;
				index_t tempFlatPosGlobal = 0;
				scalar_t pU = 0;
				scalar_t hbyAU = 0;
				scalar_t tU = 1; //transform metric
				if(pos.a[dim]==s_block.size.a[dim]-1 && s_block.boundaries[bound].type==BoundaryType::CONNECTED_GRID){
					const BlockGPU<scalar_t> *p_connectedBlock = s_domain.blocks + s_block.boundaries[bound].cb.connectedGridIndex;
					tempPos = computeConnectedPosWithChannel(tempPos, dim, &s_block.boundaries[bound].cb, s_domain);
					hbyAU = s_domain.pressureRHS[flattenIndexGlobal(tempPos, p_connectedBlock, s_domain)];
					tU = getTransformMetricOrthogonalDimSwitch<scalar_t>(tempPos, p_connectedBlock, s_domain.numDims);
					tempPos.w = 0;
					const index_t blockFlatIdx = flattenIndex(tempPos, p_connectedBlock);
					pU = p_connectedBlock->pressure[blockFlatIdx];
					tempFlatPosGlobal = blockFlatIdx + p_connectedBlock->globalOffset;
				}else {
					if(pos.a[dim]==s_block.size.a[dim]-1 && s_block.boundaries[bound].type==BoundaryType::PERIODIC){
						tempPos.a[dim] = 0;
					}else{
						tempPos.a[dim] = pos.a[dim]+1;
					}
					hbyAU = s_domain.pressureRHS[flattenIndexGlobal(tempPos, s_block, s_domain)];
					tU = getTransformMetricOrthogonalDimSwitch<scalar_t>(tempPos, &s_block, s_domain.numDims);
					tempPos.w = 0;
					const index_t blockFlatIdx = flattenIndex(tempPos, s_block);
					pU = s_block.pressure[blockFlatIdx];
					tempFlatPosGlobal = blockFlatIdx + s_block.globalOffset;
				}
				const scalar_t raU = 1 /s_domain.Adiag[tempFlatPosGlobal];
				//const scalar_t coefficient = static_cast<scalar_t>(0.25)*(ra + raU);
				//velUpdate -= coefficient*(pU - p);
				//velUpdate += (hbyA + hbyAU - (ra + raU)*((t+tU)*static_cast<scalar_t>(0.5))*(pU - p)) * static_cast<scalar_t>(0.25);
				velUpdate += (hbyA + hbyAU - (t*ra + tU*raU)*(pU - p) * timeStep) * static_cast<scalar_t>(0.25);
			}
			
			s_domain.velocityResult[flatCompPosGlobal] = velUpdate;
		}
	)
}


// pressure gradient differenced over +-1
template <typename scalar_t>
__global__ void PISO_update_velocity(DomainGPU<scalar_t> *p_domain, const scalar_t timeStep,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks){
	//vel.x = pressureRHS.x - inv(A)*gradX(pressure)
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		const index_t flatPosGlobal = s_block.globalOffset + flatPos;
		
		const scalar_t diag_inv = 1/s_domain.Adiag[flatPosGlobal];
		
		scalar_t pressureGrad[3];
		tempGetPressureGradientDimSwitch<scalar_t>(s_block, pos, s_domain, pressureGrad);
		
		for(int dim=0;dim<s_domain.numDims;++dim){
			I4 tempPos = pos;
			tempPos.w = 0;
			
			/*
			// TEST: one-sided differences at prescribed boundaries
			const bool use_bound_vel = false;
			scalar_t fac = 0.5;
			scalar_t add = 0;

			if(!(pos.a[dim]==0 && isEmptyBound(dim*2,s_block.boundaries))){
				tempPos.a[dim] = pos.a[dim]-1;
			}else{
				if(use_bound_vel){
					const index_t bound = dim*2;
					switch(s_block.boundaries[bound].type){
					case BoundaryType::DIRICHLET:
						// enforce flux
						add = 0.5 * s_block.boundaries[bound].sdb.velocity.a[dim];
						break;
					case BoundaryType::DIRICHLET_VARYING:
					{
						// enforce flux
						tempPos.a[dim] = 0;
						tempPos.w = dim;
						add = 0.5 * s_block.boundaries[bound].vdb.velocity[flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride)];
						tempPos.w = 0;
						tempPos.a[dim] = pos.a[dim];
						break;
					}
					case BoundaryType::GRADIENT:
					default:
						//add = 0;
						break;
					}
				}else{
					fac = 1;
				}
			}
			//const scalar_t valN = getPressureAtWithBounds(tempPos, s_block, s_domain);
			
			if(!(pos.a[dim]==s_block.size.a[dim]-1 && isEmptyBound(dim*2+1,s_block.boundaries))){
				tempPos.a[dim] = pos.a[dim]+1;
			}else{
				if(use_bound_vel){
					const index_t bound = dim*2+1;
					switch(s_block.boundaries[bound].type){
					case BoundaryType::VALUE:
						// enforce flux
						add = 0.5 * s_block.boundaries[bound].sdb.velocity.a[dim];
						break;
					case BoundaryType::DIRICHLET_VARYING:
					{
						// enforce flux
						tempPos.a[dim] = 0;
						tempPos.w = dim;
						add = 0.5 * s_block.boundaries[bound].vdb.velocity[flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride)];
						tempPos.w = 0;
						//tempPos.a[dim] = pos.a[dim];
						break;
					}
					case BoundaryType::GRADIENT:
					default:
						//add = 0;
						break;
					}
				}else{
					fac = 1;
				}
				
				tempPos.a[dim] = pos.a[dim];
			}
			//const scalar_t valP = getPressureAtWithBounds(tempPos, s_block, s_domain);

			//scalar_t velUpdate = - diag_inv * ((valP - valN)*fac - add);
			//scalar_t velUpdate = - diag_inv * ((valP - valN)*fac) + add;
			// TEST END */
			scalar_t velUpdate = - diag_inv * pressureGrad[dim] * timeStep;
			
			/*
			tempPos.a[dim] = pos.a[dim]-1;
			const scalar_t valN = getPressureAtWithBounds(tempPos, s_block, s_domain);
			tempPos.a[dim] = pos.a[dim]+1;
			const scalar_t valP = getPressureAtWithBounds(tempPos, s_block, s_domain);
			const scalar_t diag_inv = 1/s_domain.Adiag[flatPosGlobal];
			scalar_t velUpdate = - diag_inv * (valP - valN)*0.5;//*/
			//scalar_t velUpdate = valP;
			
			tempPos.w = dim;
			tempPos.a[dim] = pos.a[dim];
			const int flatCompPosGlobal = flattenIndexGlobal(tempPos, s_block, s_domain);
			velUpdate += s_domain.pressureRHS[flatCompPosGlobal];
			s_domain.velocityResult[flatCompPosGlobal] = velUpdate;
		}
	)
}

template <typename scalar_t>
__global__ void PISO_update_velocity_GRAD(DomainGPU<scalar_t> *p_domain, const scalar_t timeStep,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks){
	//vel.x = pressureRHS.x - inv(A)*gradX(pressure)
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		const index_t flatPosGlobal = s_block.globalOffset + flatPos;
		
		const scalar_t diag_inv = 1/s_domain.Adiag[flatPosGlobal];
		
		scalar_t pressureGradGrad[3] = {0};
		
		for(int dim=0;dim<s_domain.numDims;++dim){
			
			I4 tempPos = pos;
			tempPos.w = dim;
			int flatCompPosGlobal = flattenIndexGlobal(tempPos, s_block, s_domain);
			
			scalar_t velUpdateGrad = s_domain.velocityResult_grad[flatCompPosGlobal];
			
			// w.r.t. pressure rhs
			s_domain.pressureRHS_grad[flatCompPosGlobal] = velUpdateGrad;
			
			pressureGradGrad[dim] = - diag_inv * timeStep * velUpdateGrad;
			
			/* OLD
			velUpdateGrad = velUpdateGrad * timeStep;
			
			// w.r.t. pressure grad
			scalar_t fac = 0.5;

			const BlockGPU<scalar_t> *p_blockN = &s_block;
			I4 tempPosN = pos;
			if(pos.a[dim]==0){
				switch(s_block.boundaries[dim*2].type){
					case BoundaryType::DIRICHLET:
					case BoundaryType::DIRICHLET_VARYING:
					case BoundaryType::NEUMANN:
						fac = 1;
						break;
					case BoundaryType::CONNECTED_GRID:
					{
						const index_t bound = dim*2;
						p_blockN = s_domain.blocks + s_block.boundaries[bound].cb.connectedGridIndex;
						tempPosN = computeConnectedPos(tempPosN, dim, &s_block.boundaries[bound].cb, s_domain);
						//tempPos.w = 0; // set by computeConnectedPos to connected axis
						//tempFlatPosGlobal = flattenIndex(tempPos, p_connectedBlock) + p_connectedBlock->globalOffset;
						//TODO
						break;
					}
					case BoundaryType::PERIODIC:
						tempPosN.a[dim] = s_block.size.a[dim]-1;
						break;
					default:
						break;
				}
			}else{
				tempPosN.a[dim] = pos.a[dim]-1;
			}
			tempPosN.w = 0;
			
			const BlockGPU<scalar_t> *p_blockP = &s_block;
			I4 tempPosP = pos;
			if(pos.a[dim]==s_block.size.a[dim]-1){
				switch(s_block.boundaries[dim*2+1].type){
					case BoundaryType::DIRICHLET:
					case BoundaryType::DIRICHLET_VARYING:
					case BoundaryType::NEUMANN:
						fac = 1;
						tempPosP.a[dim] = pos.a[dim];
						break;
					case BoundaryType::CONNECTED_GRID:
					{
						const index_t bound = dim*2+1;
						p_blockP = s_domain.blocks + s_block.boundaries[bound].cb.connectedGridIndex;
						tempPosP = computeConnectedPos(tempPosP, dim, &s_block.boundaries[bound].cb, s_domain);
						break;
					}
					case BoundaryType::PERIODIC:
						tempPosP.a[dim] = 0;
						break;
					default:
						break;
				}
			}else{
				tempPosP.a[dim] = pos.a[dim]+1;
			}
			tempPosP.w = 0;
			
			//update here as fac can be changed by both lower and upper boundary
			scalar_t *p_pressureGrad = p_blockN->pressure_grad + flattenIndex(tempPosN, p_blockN);
			atomicAdd(p_pressureGrad, diag_inv*fac*velUpdateGrad);
			
			p_pressureGrad = p_blockP->pressure_grad + flattenIndex(tempPosP, p_blockP);
			atomicAdd(p_pressureGrad, -diag_inv*fac*velUpdateGrad);
			// END OLD */
			
			/* gather version, has issues at boundaries
			const BlockGPU<scalar_t> *p_block = &s_block;
			index_t tempFlatPosGlobal = 0;
			// used as upper by lower in fwd
			tempPos.w = dim;
			if(pos.a[dim]==0){
				switch(s_block.boundaries[dim*2].type){
					case BoundaryType::DIRICHLET:
					case BoundaryType::DIRICHLET_VARYING:
					case BoundaryType::NEUMANN:
						fac = -1;
						break;
					case BoundaryType::CONNECTED_GRID:
						const index_t bound = dim*2;
						p_block = s_domain.blocks + s_block.boundaries[bound].cb.connectedGridIndex;
						tempPos = computeConnectedPos(tempPos, dim, &s_block.boundaries[bound].cb, s_domain);
						//tempPos.w = 0; // set by computeConnectedPos to connected axis
						//tempFlatPosGlobal = flattenIndex(tempPos, p_connectedBlock) + p_connectedBlock->globalOffset;
						//TODO
						break;
					case BoundaryType::PERIODIC:
						tempPos.a[dim] = s_block.size.a[dim]-1;
						break;
					default:
						break;
				}
			}else{
				tempPos.a[dim] = pos.a[dim]-1;
			}
			// TODO: handle block boundaries/connections/periodic: getAdiagAtWithBounds(), getVelocityResultGradAtWithBounds()
			// -> just get right global idx (from block and local pos)
			flatCompPosGlobal = flattenIndexGlobal(tempPos, p_block, s_domain);
			tempPos.w = 0;
			tempFlatPosGlobal = flattenIndexGlobal(tempPos, p_block, s_domain);
			pressureGrad -= 1/s_domain.Adiag[tempFlatPosGlobal] * fac * s_domain.velocityResult_grad[flatCompPosGlobal];
			
			// used as lower by upper in fwd
			p_block = &s_block;
			tempPos.w = dim;
			if(pos.a[dim]==s_block.size.a[dim]-1){
				switch(s_block.boundaries[dim*2+1].type){
					case BoundaryType::DIRICHLET:
					case BoundaryType::DIRICHLET_VARYING:
					case BoundaryType::NEUMANN:
						fac = -1;
						tempPos.a[dim] = pos.a[dim];
						break;
					case BoundaryType::CONNECTED_GRID:
						const index_t bound = dim*2+1;
						p_block = s_domain.blocks + s_block.boundaries[bound].cb.connectedGridIndex;
						tempPos = computeConnectedPos(tempPos, dim, &s_block.boundaries[bound].cb, s_domain);
						break;
					case BoundaryType::PERIODIC:
						tempPos.a[dim] = 0;
						break;
					default:
						break;
				}
			}else{
				tempPos.a[dim] = pos.a[dim]+1;
			}
			// TODO: handle block boundaries/connections: getAdiagAtWithBounds(), getVelocityResultGradAtWithBounds()
			flatCompPosGlobal = flattenIndexGlobal(tempPos, s_block, s_domain);
			tempPos.w = 0;
			tempFlatPosGlobal = flattenIndexGlobal(tempPos, s_block, s_domain);
			pressureGrad += 1/s_domain.Adiag[tempFlatPosGlobal] * fac * s_domain.velocityResult_grad[flatCompPosGlobal];
			*/
		}
		
		// scatter pressure grad grad:
		tempScatterPressureGradientGradDimSwitch(pressureGradGrad, s_block, pos, s_domain);
		
		//const index_t flatPosGlobal = flattenIndexGlobal(pos, s_block, s_domain);
		//s_domain.pressureResult_grad[flatPosGlobal] = pressureGrad;
		// s_block.pressure_grad[flatPos] = pressureGrad;
	)
}