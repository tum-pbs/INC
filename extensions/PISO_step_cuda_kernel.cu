#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
//#include <cublas_v2.h>
//#include <cusparse.h>
//#include <cusolverSp.h>

#include <vector>
#include <limits>


#define LOGGING
#ifdef LOGGING
#define PROFILING
#endif
#include "logging.h"


static void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err) {
  if (err == cudaSuccess) return;
  std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
            << err << ") at " << file << ":" << line << std::endl;
  exit(10);
}
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)


/* static void CheckCublasErrorAux(const char* file, unsigned line, const char* statement, cublasStatus_t err) {
  if (err == CUBLAS_STATUS_SUCCESS) return;
  std::cerr << statement << " returned " << "("
            << err << ") at " << file << ":" << line << std::endl;
  exit(10);
}
#define CUBLAS_CHECK_RETURN(value) CheckCublasErrorAux(__FILE__, __LINE__, #value, value) */
/* 
static void CheckCusparseErrorAux(const char* file, unsigned line, const char* statement, cusparseStatus_t err) {
  if (err == CUSPARSE_STATUS_SUCCESS) return;
  std::cerr << statement << " returned " << "("
            << err << ") at " << file << ":" << line << std::endl;
  exit(10);
}
#define CUSPARSE_CHECK_RETURN(value) CheckCusparseErrorAux(__FILE__, __LINE__, #value, value)

static void CheckCusolverErrorAux(const char* file, unsigned line, const char* statement, cusolverStatus_t err) {
  if (err == CUSOLVER_STATUS_SUCCESS) return;
  std::cerr << statement << " returned " << "("
            << err << ") at " << file << ":" << line << std::endl;
  exit(10);
}
#define CUSOLVER_CHECK_RETURN(value) CheckCusolverErrorAux(__FILE__, __LINE__, #value, value)
 */
//const int WARP_SIZE = 32;

__device__ inline constexpr int divCeil(const int a, const int b){
	return (a + b - 1)/b;
}

enum class BoundaryType : int8_t{
	DIRICHLET=0,
	FIRST_TYPE=0,
	VALUE=0,
	DIRICHLET_VARYING=1,
	
	NEUMANN=10,
	SECOND_TYPE=10,
	GRADIENT=10,
	
	CONNECTED_GRID=20,
	PERIODIC=21
};

using index_t = size_t;
using dim_t = int8_t;

typedef union{
	int4 v;
	int32_t a[4];
} I4;

typedef union{
	float4 v;
	float a[4];
} F4;

template<typename scalar_t>
struct scalar4{
	scalar_t x;
	scalar_t y;
	scalar_t z;
	scalar_t w;
}

template<typename scalar_t>
typedef union{
	scalar4<scalar_t> v;
	scalar_t a[4];
} S4;



template <typename scalar_t>
struct StaticDirichletBoundary{
	scalar_t slip;
	S4<scalar_t> boundaryVelocity;
};

template <typename scalar_t>
struct StaticGradientBoundary{
	scalar_t slip;
	S4<scalar_t> boundaryGradient;
};

template <typename scalar_t>
struct VaryingDirichletBoundary{
	scalar_t slip;
	I4 boundarySize;
	I4 boundaryStride;
	scalar_t *boundaryVelocity;
};

template <typename scalar_t>
struct ConnectedBoundary{
	index_t connectedGridIndex;
	dim_t connectedFace;
	dim_t connectedAxis1;
	dim_t connectedAxis2;
};


template <typename scalar_t>
struct BoundaryInfo{
	BoundaryType type;
	union{
		StaticDirichletBoundary sdb;
		VaryingDirichletBoundary vdb;
		StaticGradientBoundary sgb;
		ConnectedBoundary cb;
	};
	/*
	index_t connectedGridIndex;
	//0,1,2 for positive/upper/same direction, -3,-2,-1 for negative/lower/inverse direction
	dim_t connectedFace;
	dim_t connectedAxis1;
	dim_t connectedAxis2;
	scalar_t prescribedValue;
	scalar_t slip;
	*/
};

template <typename scalar_t>
struct BlockInfo{
	I4 size;
	I4 stride;
	BoundaryInfo<scalar_t> bounds[6];
	scalar_t *scalarData;
	scalar_t *scalarResult;
	scalar_t *scalarRHS;
	scalar_t *velocity;
	scalar_t *velocityResult;
	scalar_t *velocityRHS;
	scalar_t *pressure;
	scalar_t *pressureRHS;
	scalar_t *pressureRHSdiv;
};

template <typename scalar_t>
struct DomainAtlas{
	dim_t numDims;
	index_t numBlocks;
	BlockInfo<scalar_t> *blockInfos;
};



template <typename scalar_t>
struct Domain{
	union{
		int4 size;
		int32_t sizes[4];
	};
	union{
		int4 stride;
		int32_t strides[4];
	};
	size_t numDims;
	scalar_t timeStep;
	scalar_t viscosity;
	BoundaryInfo<scalar_t> bounds[6];
};

__constant__ Domain<float> c_domain;

struct Indexing{
	int blocksPerGrid;
	int blockRepetitions;
};

template <typename scalar_t>
struct BlockAtlas{
	scalar_t **blockMap;
	Domain<scalar_t> **blockDomains;
};

/* //sizeof(Domain) must be multiples of sizeof(loadType) and aligned to sizeof(loadType)
template <typename loadType>
__device__ void loadDomainToShared(const BlockAtlas blockAtlas, const size_t gridIndex, Domain *domainShared){
	const loadType *srcPtr = static_cast<const loadType>(blockAtlas.blockDomains[gridIndex]);
	loadType *dstPtr = static_cast<loadType>(domainShared);
	const int numLoads = sizeof(Domain) / sizeof(loadType); //divCeil(sizeof(Domain), sizeof(loadType));
	const int loadIndex = threadIdx.x; // * sizeof(loadType);
	
	if(loadIndex<numLoads){
		dstPtr[loadIndex] = srcPtr[loadIndex];
	}
	__syncthreads();
}

// get position on neighbour grid from current position and boundary information
template <typename loadType>
__device__ int3 getPosFromBounds(const int3 pos, const char face, const char axis1, const char axis2, const BoundaryInfo<scalar_t> bounds){
	int *srcPos = {pos.x,pos.y,pos.z};
	int dstPos[3];
	
	if(bounds.connectedFace<0){
		dstPos[bounds.connectedFace+3] = 0;
	}else{
		dstPos[bounds.connectedFace] = targetDomain.sizes[bounds.connectedFace]-1;
	}
	
	if(bounds.connectedAxis1<0){
		dstPos[bounds.connectedAxis1+3] = targetDomain.sizes[bounds.connectedAxis1+3]-1 - srcPos[axis1];
	}else{
		dstPos[bounds.connectedAxis1] = srcPos[axis1];
	}
	if(bounds.connectedAxis2<0){
		dstPos[bounds.connectedAxis2+3] = targetDomain.sizes[bounds.connectedAxis2+3]-1 - srcPos[axis2];
	}else{
		dstPos[bounds.connectedAxis2] = srcPos[axis2];
	}
	
	return make_int3(dstPos[0], dstPos[1], dstPos[2]);
}

__constant__ Domain c_domain;
__constant__ Indexing c_indexing; */


__device__ inline int flattenIndex(const int x,const int y,const int z){
	return x + c_domain.stride.y*y + c_domain.stride.z*z;
}
__device__ inline int flattenIndex(const int x,const int y,const int z,const int c){
	return x + c_domain.stride.y*y + c_domain.stride.z*z + c_domain.stride.w*c;
}
__device__ inline int flattenIndex(const int3 pos){
	return pos.x + c_domain.stride.y*pos.y + c_domain.stride.z*pos.z;
}
__device__ inline int flattenIndex(const int4 pos){
	return pos.x + c_domain.stride.y*pos.y + c_domain.stride.z*pos.z + c_domain.stride.w*pos.w;
}
__device__ inline int flattenIndex(const int4 pos, const int4 stride){
	return pos.x + stride.y*pos.y + stride.z*pos.z + stride.w*pos.w;
}
template<int DIM>
__device__ inline int flattenIndex(const int* pos);
template<>
__device__ inline int flattenIndex<3>(const int* pos){
	return pos[0] + c_domain.stride.y*pos[1] + c_domain.stride.z*pos[2];
}
template<>
__device__ inline int flattenIndex<4>(const int* pos){
	return pos[0] + c_domain.stride.y*pos[1] + c_domain.stride.z*pos[2] + c_domain.stride.w*pos[3];
}

__device__ inline int4 unflattenIndex(const int idx){
	return make_int4(idx%c_domain.size.x, (idx/c_domain.stride.y)%c_domain.size.y, (idx/c_domain.stride.z)%c_domain.size.z, (idx/c_domain.stride.w)%c_domain.size.w);
} 

__host__ __device__ inline bool isEmptyBound(const int idx, const BoundaryInfo<float> *bounds){
	return bounds[idx].type==BoundaryType::VALUE || bounds[idx].type==BoundaryType::DIRICHLET_VARYING || bounds[idx].type==BoundaryType::GRADIENT;
}

__host__ int getCSRSize(const Domain<float> &domain){
	int csrSize = domain.stride.w*(2*domain.numDims+1);
	
	//X
	int sizeFaceX = domain.size.y*domain.size.z;
	// -x
	if(isEmptyBound(0,domain.bounds)){
		csrSize -= sizeFaceX;
	}
	// +x
	if(isEmptyBound(1,domain.bounds)){
		csrSize -= sizeFaceX;
	}
	
	//Y
	if(domain.numDims>1){
		int sizeFaceY = domain.size.x*domain.size.z;
		// -y
		if(isEmptyBound(2,domain.bounds)){
			csrSize -= sizeFaceY;
		}
		// +y
		if(isEmptyBound(3,domain.bounds)){
			csrSize -= sizeFaceY;
		}
	}
	
	//Z
	if(domain.numDims>2){
		int sizeFaceZ = domain.size.x*domain.size.y;
		// -z
		if(isEmptyBound(4,domain.bounds)){
			csrSize -= sizeFaceZ;
		}
		// +z
		if(isEmptyBound(5,domain.bounds)){
			csrSize -= sizeFaceZ;
		}
	}
	
	return csrSize;
}

struct RowMeta{
	int endOffset;
	int size;
};

//template<typename scalar_t>
__device__ RowMeta getCSRMatrixRowEndOffsetFromBlockBoundaries3D(const int flatPos, const Domain<float> &domain){
	const int4 pos = unflattenIndex(flatPos);
		
	int rowSize = 2*domain.numDims+1;
	int rowEndOffset=(flatPos+1)*rowSize;
	
	//subtract for open/closed bounds
	// number of cells with boundary at ? before and including current cell:
	//X
	// -x
	if(isEmptyBound(0,domain.bounds)){
		rowEndOffset -= (flatPos/domain.size.x)+1;
		if(pos.x==0){
			--rowSize;
		}
	}
	// +x
	if(isEmptyBound(1,domain.bounds)){
		rowEndOffset -= (flatPos + 1)/domain.size.x;
		if(pos.x==domain.size.x-1){
			--rowSize;
		}
	}
	//Y
	if(domain.numDims>1){
		// -y
		if(isEmptyBound(2,domain.bounds)){
			rowEndOffset -= domain.size.x*pos.z //previous slices
				+ (pos.y==0 ? pos.x+1 : domain.size.x); // current slice
			if(pos.y==0){
				--rowSize;
			}
		}
		// +y
		if(isEmptyBound(3,domain.bounds)){
			rowEndOffset -= domain.size.x*pos.z //previous slices
				+ (pos.y==(domain.size.y-1) ? pos.x+1 : 0); // current slice
			if(pos.y==domain.size.y-1){
				--rowSize;
			}
		}
	}
	//Z
	if(domain.numDims>2){
		// -z
		if(isEmptyBound(4,domain.bounds)){
			rowEndOffset -= (pos.z==0 ? flatPos+1 : domain.stride.z); // flatPos=pos.x+pos.y*size.y, stride.z=size.x*size.y
			if(pos.z==0){
				--rowSize;
			}
		}
		// +z
		if(isEmptyBound(5,domain.bounds)){
			rowEndOffset -= (pos.z==(domain.size.z-1) ? flatPos+1 - (pos.z*domain.stride.z): 0); // stride.z=size.x*size.y
			if(pos.z==domain.size.z-1){
				--rowSize;
			}
		}
	}
	return {rowEndOffset, rowSize};
}

template <typename scalar_t>
__device__ void computeFluxesNDLoop(const scalar_t* velocity, const int4 position, scalar_t* fluxes, const Domain<float> &domain){
	//scalar_t fluxes[6];
	//x
	//const int dimensions = 3;
	
	const I4 pos = {.v=position};
	
	for(int dim=0; dim<domain.numDims; ++dim)
	{
		I4 tempPos = {.v=position};
		tempPos.v.w = dim;
		
		const scalar_t velC = velocity[flattenIndex(tempPos.v)];
		//debug
		//if(position.x==0 && position.y==0 && position.z==0){
		//	printf("flat pos for component %d: %d", dim, flattenIndex(tempPos, 4));
		//}
		
		int bound = dim*2;
		
		if(pos.a[dim]==0){// lower boundary
			switch(domain.bounds[bound].type){
				case BoundaryType::DIRICHLET:
					// enforce flux
					fluxes[bound] = -domain.bounds[bound].sdb.boundaryVelocity[tempPos.v.w];
					break;
				case BoundaryType::DIRICHLET_VARYING:
					// enforce flux
					{
						tempPos.a[dim] = 0;
						fluxes[bound] = -domain.bounds[bound].vdb.boundaryVelocity[flattenIndex(tempPos.v, domain.bounds[bound].vdb.stride)];
					}
					break;
				case BoundaryType::GRADIENT:
					// compute flux only from center cell?
					// velN = velC - grad*distance & flux = (velN+velC)*0.5 ->
					fluxes[bound] = 0; //TODO velC + domain.bounds[bound].sgb.boundaryGradient.a[tempPos.v.w] * 0.5; //* distance
					break;
				case BoundaryType::CONNECTED_GRID:
					//TODO: handle multi-block grids, load from correct cell of the connected grid
				case BoundaryType::PERIODIC:
					// compute flux to cell on other side
					// special case of connection to another block
					{
						tempPos.a[dim] = domain.sizes[dim] - 1;
						const scalar_t velN = velocity[flattenIndex(tempPos.v)];
						fluxes[bound] = (velN + velC) * 0.5f;
					}
					break;
				default:
					fluxes[bound] = 0; //(velN + velC) * 0.5f;
					break;
			}
		}else{
			tempPos.a[dim] = pos.a[dim] - 1;
			const scalar_t velN = velocity[flattenIndex(tempPos.v)];
			fluxes[bound] = (velN + velC) * 0.5f;
		}
		
		bound = dim*2 + 1;
		
		if(pos.a[dim]==(domain.sizes[dim]-1)){// upper boundary
			switch(domain.bounds[bound].type){
				case BoundaryType::VALUE:
					// enforce flux
					fluxes[bound] = domain.bounds[bound].sdb.boundaryVelocity.a[tempPos.v.w];
					break;
				case BoundaryType::DIRICHLET_VARYING:
					// enforce flux
					{
						tempPos.a[dim] = 0;
						fluxes[bound] = domain.bounds[bound].vdb.boundaryVelocity[flattenIndex(tempPos.v, domain.bounds[bound].vdb.stride)];
					}
					break;
				case BoundaryType::GRADIENT:
					// compute flux only from center cell?
					// velP = grad*distance*2 + velC & flux = (velN+velC)*0.5 ->
					fluxes[bound] = 0; //TODO velC + domain.bounds[bound].sgb.boundaryGradient.a[tempPos.v.w] * 0.5; //* distance
					break;
				case BoundaryType::CONNECTED_GRID:
					//TODO: handle multi-block grids, load from correct cell of the connected grid
				case BoundaryType::PERIODIC:
					// compute flux to cell on other side
					// special case of connection to another block
					{
						tempPos.a[dim] = 0;
						const scalar_t velP = velocity[flattenIndex(tempPos.v)];
						fluxes[bound] = (velC + velP) * 0.5f;
					}
					break;
				default:
					fluxes[bound] = 0; //(velN + velC) * 0.5f;
					break;
			}
			
		}else{
			tempPos.a[dim] = pos.a[dim] + 1;
			const scalar_t velP = velocity[flattenIndex(tempPos.v)];
			fluxes[bound] = (velC + velP) * 0.5f;
		}
		
		
	}
}

template<typename scalar_t>
__device__ scalar_t getVelocityAtWithBounds(const scalar_t *grid, const int4 position, const Domain<float> &domain){
	// read value at location with support for 1-cell ghost layer
	// any position outside the domain will be treated as being on the ghost layer
	// only one coordinate may be outside the domain (corner ghost cells are not supported)
	
	const int flatPos = flattenIndex(position);
	I4 pos = {.v=position};
	
	for(int dim=0; dim<domain.numDims; ++dim)
	{
		int bound = dim*2;
		
		if(pos[dim]<0){// lower boundary
			switch(domain.bounds[bound].type){
				case BoundaryType::VALUE:
					pos[dim] = 0;
					return 2*domain.bounds[bound].sdb.boundaryVelocity[pos.v.w] - grid[flattenIndex(pos.v)]; 
					break;
				case BoundaryType::DIRICHLET_VARYING:
					{
						pos[dim] = 0;
						scalar_t boundaryVelocity = domain.bounds[bound].vdb.boundaryVelocity[flattenIndex(pos.v, domain.bounds[bound].vdb.stride)];
						return 2*boundaryVelocity - grid[flattenIndex(pos.v)]; 
					}
					break;
				case BoundaryType::GRADIENT:
					// compute flux only from center cell?
					// velN = velC - grad*distance
					return grid[flatPos] - domain.bounds[bound].sgb.boundaryGradient.a[pos.v.w]; //* distance
					break;
				case BoundaryType::CONNECTED_GRID:
					//TODO: handle multi-block grids, load from correct cell of the connected grid
				case BoundaryType::PERIODIC:
					// compute flux to cell on other side
					// special case of connection to another block
					pos[dim] = c_domain.sizes[dim] - 1;
					return grid[flattenIndex(pos.v)];
					break;
				default:
					return 0;
					break;
			}
		}
		
		bound = dim*2 + 1;
		
		if(pos[dim]>=domain.sizes[dim]){// upper boundary
			switch(domain.bounds[bound].type){
				case BoundaryType::VALUE:
					pos[dim] = c_domain.sizes[dim] - 1;
					return 2*domain.bounds[bound].sdb.boundaryVelocity[pos.v.w] - grid[flattenIndex(pos.v)]; 
					break;
				case BoundaryType::DIRICHLET_VARYING:
					{
						pos[dim] = 0;
						scalar_t boundaryVelocity = domain.bounds[bound].vdb.boundaryVelocity[flattenIndex(pos.v, domain.bounds[bound].vdb.stride)];
						pos[dim] = c_domain.sizes[dim] - 1;
						return 2*boundaryVelocity - grid[flattenIndex(pos.v)]; 
					}
					break;
				case BoundaryType::GRADIENT:
					// compute flux only from center cell?
					// velP = - celC - grad*distance
					return -grid[flatPos] - domain.bounds[bound].sgb.boundaryGradient.a[pos.v.w]; //* distance
					break;
				case BoundaryType::CONNECTED_GRID:
					//TODO: handle multi-block grids, load from correct cell of the connected grid
				case BoundaryType::PERIODIC:
					// compute flux to cell on other side
					// special case of connection to another block
					pos[dim] = 0;
					return grid[flattenIndex(pos.v)];
					break;
				default:
					return 0;
					break;
			}
		}
	}
	return grid[flatPos];
}

template<typename scalar_t>
__device__ scalar_t getPressureAtWithBounds(const scalar_t *grid, const I4 position, const Domain<float> &domain){
	// read value at location with support for 1-cell ghost layer
	// any position outside the domain will be treated as being on the ghost layer
	// only one coordinate may be outside the domain (corner ghost cells are not supported)
	
	const int flatPos = flattenIndex(position.v);
	int pos[4] = {position.v.x, position.v.y, position.v.z, position.v.w};
	
	for(int dim=0; dim<domain.numDims; ++dim)
	{
		int bound = dim*2;
		
		if(pos[dim]<0){// lower boundary
			switch(domain.bounds[bound].type){
				case BoundaryType::VALUE:
				case BoundaryType::DIRICHLET_VARYING:
				case BoundaryType::GRADIENT: //TODO: how to handle this case here?
					// enforce 0 pressure gradient to avoid changing the prescribed value
					pos[dim] = 0;
					return grid[flattenIndex<4>(pos)];
					break;
					// compute flux only from center cell?
					// velN = velC - grad*distance
					//return grid[flatPos] - domain.bounds[bound].prescribedValue; //* distance
					//break;
				case BoundaryType::CONNECTED_GRID:
					//TODO: handle multi-block grids, load from correct cell of the connected grid
				case BoundaryType::PERIODIC:
					// compute flux to cell on other side
					// special case of connection to another block
					pos[dim] += c_domain.sizes[dim];
					return grid[flattenIndex<4>(pos)];
					break;
				default:
					return 0;
					break;
			}
		}
		
		bound = dim*2 + 1;
		
		if(pos[dim]>=domain.sizes[dim]){// upper boundary
			switch(domain.bounds[bound].type){
				case BoundaryType::VALUE:
				case BoundaryType::DIRICHLET_VARYING:
				case BoundaryType::GRADIENT: //TODO: how to handle this case here?
					// enforce 0 pressure gradient to avoid changing the prescribed value
					pos[dim] = c_domain.sizes[dim] - 1;
					return grid[flattenIndex<4>(pos)];
					break;
				/* case BoundaryType::GRADIENT:
					// compute flux only from center cell?
					// velP = celC + grad*distance
					return grid[flatPos] + domain.bounds[bound].prescribedValue; //* distance
					break; */
				case BoundaryType::CONNECTED_GRID:
					//TODO: handle multi-block grids, load from correct cell of the connected grid
				case BoundaryType::PERIODIC:
					// compute flux to cell on other side
					// special case of connection to another block
					pos[dim] -= c_domain.sizes[dim];
					return grid[flattenIndex<4>(pos)];
					break;
				default:
					return 0;
					break;
			}
		}
	}
	return grid[flatPos];
}

template <typename scalar_t>
__device__ void computeFluxesMWI(const scalar_t* velocity, const scalar_t* pressure, const int4 position, scalar_t* fluxes, const Domain<float> &domain){
	// according to eq. (57) of "Unified formulation of the momentum-weighted interpolation for collocated variable arrangements"
	//scalar_t fluxes[6];
	//x
	//const int dimensions = 3;
	
	const I4 pos = {.v=position};
	
	for(int dim=0; dim<domain.numDims; ++dim)
	{
		I4 tempPos = {.v=position};
		
		tempPos.v.w = 0;
		const index_t flatPos = flattenIndex(tempPos.v);
		const scalar_t pC = pressure[flatPos];
		tempPos.a[dim] = pos.a[dim]-1;
		const scalar_t pW = getPressureAtWithBounds(pressure, tempPos, domain);
		tempPos.a[dim] = pos.a[dim]+1;
		const scalar_t pE = getPressureAtWithBounds(pressure, tempPos, domain);
		
		tempPos.v.w = dim;
		const index_t flatPosVel = flattenIndex(tempPos.v);
		
		const scalar_t velC = velocity[flatPosVel];
		//debug
		//if(position.x==0 && position.y==0 && position.z==0){
		//	printf("flat pos for component %d: %d", dim, flattenIndex(tempPos, 4));
		//}
		
		//lower boundary
		int bound = dim*2;
		
		
		if(pos.a[dim]==0){// lower boundary
			switch(domain.bounds[bound].type){
				case BoundaryType::DIRICHLET:
					// enforce flux
					tempPos.v.w = dim;
					fluxes[bound] = -domain.bounds[bound].sdb.boundaryVelocity[tempPos.v.w];
					break;
				case BoundaryType::DIRICHLET_VARYING:
					// enforce flux
					{
						tempPos.a[dim] = 0;
						tempPos.v.w = dim;
						fluxes[bound] = -domain.bounds[bound].vdb.boundaryVelocity[flattenIndex(tempPos.v, domain.bounds[bound].vdb.stride)];
					}
					break;
				case BoundaryType::GRADIENT:
					// compute flux only from center cell?
					// velN = velC - grad*distance & flux = (velN+velC)*0.5 ->
					fluxes[bound] = 0; //TODO velC + domain.bounds[bound].sgb.boundaryGradient.a[tempPos.v.w] * 0.5; //* distance
					break;
				case BoundaryType::CONNECTED_GRID:
					//TODO: handle multi-block grids, load from correct cell of the connected grid
				case BoundaryType::PERIODIC:
					// compute flux to cell on other side
					// special case of connection to another block
					{
						tempPos.a[dim] = domain.sizes[dim] - 1;
						tempPos.v.w = dim;
						const scalar_t velN = velocity[flattenIndex(tempPos.v)];
						fluxes[bound] = (velN + velC) * 0.5f;
					}
					break;
				default:
					fluxes[bound] = 0; //(velN + velC) * 0.5f;
					break;
			}
		}else{
			tempPos.a[dim] = pos.a[dim] - 1;
			tempPos.v.w = dim;
			const scalar_t velN = velocity[flattenIndex(tempPos.v)];
			fluxes[bound] = (velN + velC) * 0.5f;
		}
		{ // pressure correction
			fluxes[bound] -= pC - pW;
			
			tempPos.v.w = 0;
			tempPos.a[dim] = pos.a[dim]-2;
			const scalar_t pWW = getPressureAtWithBounds(pressure, tempPos, domain);
			
			fluxes[bound] += ((pC-pWW)*0.5f + (pE - pW)*0.5) * 0.5f;
		}
		
		// upper boundary
		bound = dim*2 + 1;
		
		if(pos.a[dim]==(domain.sizes[dim]-1)){
			switch(domain.bounds[bound].type){
				case BoundaryType::VALUE:
					// enforce flux
					tempPos.v.w = dim;
					fluxes[bound] = domain.bounds[bound].sdb.boundaryVelocity.a[tempPos.v.w];
					break;
				case BoundaryType::DIRICHLET_VARYING:
					// enforce flux
					{
						tempPos.a[dim] = 0;
						tempPos.v.w = dim;
						fluxes[bound] = domain.bounds[bound].vdb.boundaryVelocity[flattenIndex(tempPos.v, domain.bounds[bound].vdb.stride)];
					}
					break;
				case BoundaryType::GRADIENT:
					// compute flux only from center cell?
					// velP = grad*distance*2 + velC & flux = (velN+velC)*0.5 ->
					fluxes[bound] = 0; //TODO velC + domain.bounds[bound].sgb.boundaryGradient.a[tempPos.v.w] * 0.5; //* distance
					break;
				case BoundaryType::CONNECTED_GRID:
					//TODO: handle multi-block grids, load from correct cell of the connected grid
				case BoundaryType::PERIODIC:
					// compute flux to cell on other side
					// special case of connection to another block
					{
						tempPos.a[dim] = 0;
						tempPos.v.w = dim;
						const scalar_t velP = velocity[flattenIndex(tempPos.v)];
						fluxes[bound] = (velC + velP) * 0.5f;
					}
					break;
				default:
					fluxes[bound] = 0; //(velN + velC) * 0.5f;
					break;
			}
			
		}else{
			tempPos.a[dim] = pos.a[dim] + 1;
			tempPos.v.w = dim;
			const scalar_t velP = velocity[flattenIndex(tempPos.v)];
			fluxes[bound] = (velC + velP) * 0.5f;
		}
		{ // pressure correction
			fluxes[bound] -= pE - pC;
			
			tempPos.v.w = 0;
			tempPos.a[dim] = pos.a[dim]+2;
			const scalar_t pEE = getPressureAtWithBounds(pressure, tempPos, domain);
			
			fluxes[bound] += ((pE - pW)*0.5 + (pEE - pC)*0.5f) * 0.5f;
		}
		
		
	}
}

/*
__global__ void csrMatrixRowBlockBoundaries3D(int *csrMatrixRow, const BoundaryType *bounds){
	//precompute CSR matrix row sizes for per-block boundaries
	//bounds: -x,+x,-y,+y,-z,+z
	
	
	for(int r=0){
		const int flatPos;
		
		int rowEndOffset = getCSRMatrixRowEndOffsetFromBlockBoundaries3D(flatPos, bounds);
		
		csrMatrixRow[r+1] = rowEndOffset;
	}
}
 */

__device__ int findLowestColumnIndex(int *indices, int size){
	int value = INT_MAX;
	int index = -1;
	for(int i=0;i<size;++i){
		if(indices[i]>=0 && indices[i]<value){
			value = indices[i];
			index = i;
		}
	}
	return index;
}

template <typename scalar_t>
__global__ void PISO_build_matrix(
		const scalar_t* __restrict__ velocity, const scalar_t* __restrict__ pressure, 
		int32_t* __restrict__ csrMatrixIndex, int32_t* __restrict__ csrMatrixRow, scalar_t* __restrict__ csrMatrixValue,
		scalar_t* __restrict__ matrixDiagonal){
	
	//first naive implementation, 1 thread per cell, no sharing
	// each thread/cell builds its row in the matrix
	const int threadStride = gridDim.x*blockDim.x;
	const int globalThreadIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int repetitions = divCeil(c_domain.stride.w, threadStride);
	
	
	for(int r=0;r<repetitions;++r){
		
		const int flatPos = r*threadStride + globalThreadIndex;
		if(flatPos<c_domain.stride.w){
			const I4 pos = {.v=unflattenIndex(flatPos)};
			
			//get fluxes
			scalar_t fluxes[6];
			if(pressure==nullptr){
				computeFluxesNDLoop(velocity, pos.v, fluxes, c_domain);
			}else{
				computeFluxesMWI(velocity, pressure, pos.v, fluxes, c_domain);
			}
			
			//compute matrix
			
			//if(flatPos==0){
			//	csrMatrixRow[0] = 0;
			//}
			
			const RowMeta row = getCSRMatrixRowEndOffsetFromBlockBoundaries3D(flatPos, c_domain);
			int rowStartOffset = row.endOffset - row.size;
			
			csrMatrixRow[flatPos+1] = row.endOffset;
			//csrMatrixIndex[flatPos+1] = row.size;
			// csrMatrixIndex[flatPos] = pos.x;
			// csrMatrixIndex[flatPos + c_domain.stride.w + 1] = pos.y;
			// csrMatrixIndex[flatPos + c_domain.stride.w*2 + 2] = pos.z;
			
			
			// row entries have to be in ascending column order (for cublas)
			// for a default inner cell: -z,-y,-x,diag,+x,+y,+z
			// for boundary cell
			// - if the boundary is open or closed (no connection), the entry is simply removed
			// - if the boundary is periodic, the entry moves to the other end
			//   - grid size 1: the cell connects to itself
			//   - grid size 2: connects to same cell in both directions
			//   - cell at lower z border, domain is z-periodic: -y,-x,diag,+x,+y,+z,-z
			
			// alternative: compute flat indices, sort
			int indices[7]; // diag,-x,+x,-y,+y,-z,+z
			scalar_t rowValues[7];
			
			scalar_t diag = 1/c_domain.timeStep + c_domain.numDims*2*c_domain.viscosity; //1/dt;
			indices[0] = flatPos;
			
			for(int dim=0;dim<c_domain.numDims;++dim){
				// lower bound
				int bound=dim*2;
				diag -= fluxes[bound] * 0.5f;
				rowValues[bound+1] = fluxes[bound] * -0.5f - c_domain.viscosity;
				if(pos.a[dim]!=0 || !isEmptyBound(bound, c_domain.bounds)){
					I4 tempPos = {.v=pos.v};
					if(pos.a[dim]==0 && c_domain.bounds[bound].type==BoundaryType::PERIODIC){
						tempPos.a[dim] = c_domain.sizes[dim]-1;
					}else{
						tempPos.a[dim] = pos.a[dim]-1;
					}
					indices[bound+1] = flattenIndex(tempPos.v);
				}else{
					if(c_domain.bounds[bound].type==BoundaryType::VALUE){
						//add extra viscosity with self fixed value boundary
						diag += (1 - c_domain.bounds[bound].slip*2) * c_domain.viscosity;
					}
					indices[bound+1] = -1; //invalid/unused
				}
				//upper bound
				bound +=1;
				diag += fluxes[bound] * 0.5f;
				rowValues[bound+1] = fluxes[bound] * 0.5f - c_domain.viscosity;
				if(pos.a[dim]!=c_domain.sizes[dim]-1 || !isEmptyBound(bound, c_domain.bounds)){
					I4 tempPos = {.v=pos.v};
					if(pos.a[dim]==c_domain.sizes[dim]-1 && c_domain.bounds[bound].type==BoundaryType::PERIODIC){
						tempPos.a[dim] = 0;
					}else{
						tempPos.a[dim] = pos.a[dim]+1;
					}
					indices[bound+1] = flattenIndex(tempPos.v);
				}else{
					if(c_domain.bounds[bound].type==BoundaryType::VALUE){
						//add extra viscosity with self fixed value boundary
						diag += (1 - c_domain.bounds[bound].slip*2) * c_domain.viscosity;
					}
					indices[bound+1] = -1; //invalid/unused
				}

			}
			for(int dim=c_domain.numDims;dim<3;++dim){
				indices[dim*2+1] = -1; //invalid/unused
				indices[dim*2+2] = -1; //invalid/unused
			}

			
			rowValues[0] = diag;
			
			// sort, naive for now
			for(int i=0;i<row.size;++i){
				int colIndex = findLowestColumnIndex(indices, 7);
				
				if(colIndex<0){
					csrMatrixIndex[rowStartOffset + i] = -1;
					csrMatrixValue[rowStartOffset + i] = -1.0f;
				}else{
					csrMatrixIndex[rowStartOffset + i] = indices[colIndex];
					csrMatrixValue[rowStartOffset + i] = rowValues[colIndex];
				}
				
				indices[colIndex] = -1;
			}
			
			matrixDiagonal[flatPos] = diag;
		}
	}
}


template <typename scalar_t>
__global__ void kPISO_build_scalar_advection_RHS(
		const scalar_t* __restrict__ scalar,
		scalar_t* __restrict__ rhs){
	
	const int threadStride = gridDim.x*blockDim.x;
	const int globalThreadIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int repetitions = divCeil(c_domain.stride.w, threadStride);
	
	for(int r=0;r<repetitions;++r){
		const int flatPos = r*threadStride + globalThreadIndex;
		if(flatPos<c_domain.stride.w){
			//const I4 pos = {.v=unflattenIndex(flatPos)};
			scalar_t tempRHS = scalar[flatPos]/c_domain.timeStep;
			
			//TODO: pressure, external forces?
			
			rhs[flatPos] = tempRHS;
		}
	}
}

template <typename scalar_t>
__global__ void kPISO_build_advection_RHS(
		const scalar_t* __restrict__ velocity,
		const scalar_t* __restrict__ pressure,
		scalar_t* __restrict__ rhs){
	
	const int threadStride = gridDim.x*blockDim.x;
	const int globalThreadIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int repetitions = divCeil(c_domain.stride.w, threadStride);
	
	for(int r=0;r<repetitions;++r){
		const int flatPos = r*threadStride + globalThreadIndex;
		if(flatPos<c_domain.stride.w){
			const I4 pos = {.v=unflattenIndex(flatPos)};
			for(int dim=0;dim<c_domain.numDims;++dim){
				I4 tempPos = {.v=pos.v};
				tempPos.v.w = dim;
				const int tempFlatPos = flattenIndex(tempPos.v);
				scalar_t tempRHS = velocity[tempFlatPos]/c_domain.timeStep; // /c_domain.timeStep
				
				tempPos.v.w = 0;
				tempPos.a[dim] = pos.a[dim]-1;
				const scalar_t valN = getPressureAtWithBounds(pressure, tempPos, c_domain);
				tempPos.a[dim] = pos.a[dim]+1;
				const scalar_t valP = getPressureAtWithBounds(pressure, tempPos, c_domain);
				tempRHS -= (valP - valN)*0.5;
				
				//TODO: external forces
				
				rhs[tempFlatPos] = tempRHS;
			}
		}
	}
}

// wrap cusolver functions to work with AT_DISPATCH_FLOATING_TYPES
template<typename scalar_t>
inline void cusolverSpTcsrlsvlu(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA,
	const scalar_t *csrValA, const int *csrRowPtrA, const int *csrColIndA, const scalar_t *b, scalar_t tol, int reorder, scalar_t *x, int *singularity);
template<>
inline void cusolverSpTcsrlsvlu<float>(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA,
	const float *csrValA, const int *csrRowPtrA, const int *csrColIndA, const float *b, float tol, int reorder, float *x, int *singularity){
		CUSOLVER_CHECK_RETURN(cusolverSpScsrlsvqr(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity));
}
template<>
inline void cusolverSpTcsrlsvlu<double>(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA,
	const double *csrValA, const int *csrRowPtrA, const int *csrColIndA, const double *b, double tol, int reorder, double *x, int *singularity){
		CUSOLVER_CHECK_RETURN(cusolverSpDcsrlsvqr(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity));
}



// --- PRESSURE SOLVE ---

template <typename scalar_t>
__global__ void PISO_build_pressure_matrix(const scalar_t* __restrict__ advectionMatrixDiagonal, int32_t* __restrict__ csrMatrixIndex, int32_t* __restrict__ csrMatrixRow, scalar_t* __restrict__ csrMatrixValue){
	// again, build one row per thread
	// discrete laplace kernel multiplied with the diagonal of the advection matrix.
	const int threadStride = gridDim.x*blockDim.x;
	const int globalThreadIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int repetitions = divCeil(c_domain.stride.w, threadStride);
	
	const scalar_t invLaplace = -1;
	
	for(int r=0;r<repetitions;++r){
		
		const int flatPos = r*threadStride + globalThreadIndex;
		if(flatPos<c_domain.stride.w){
			const I4 pos = {.v=unflattenIndex(flatPos)};
			
			const RowMeta row = getCSRMatrixRowEndOffsetFromBlockBoundaries3D(flatPos, c_domain);
			int rowStartOffset = row.endOffset - row.size;
			
			csrMatrixRow[flatPos+1] = row.endOffset;
			
			// alternative: compute flat indices, sort
			int indices[7]; // diag,-x,+x,-y,+y,-z,+z
			scalar_t rowValues[7];
			
			const scalar_t ra = 1/advectionMatrixDiagonal[flatPos];
			rowValues[0] = -ra * c_domain.numDims; //(c_domain.numDims*2) / advectionMatrixDiagonal[flatPos] * invLaplace;
			indices[0] = flatPos;
			
			for(int dim=0;dim<c_domain.numDims;++dim){
				//lower boundary
				int bound = dim*2;
				if(pos.a[dim]!=0 || !isEmptyBound(bound, c_domain.bounds)){
					I4 tempPos = {.v=pos.v};
					if(pos.a[dim]==0 && c_domain.bounds[bound].type==BoundaryType::PERIODIC){
						tempPos.a[dim] = c_domain.sizes[dim]-1;
					}else{
						tempPos.a[dim] = pos.a[dim]-1;
					}
					const int tempFlatPos = flattenIndex(tempPos.v);
					//rowValues[bound+1] = -1 / advectionMatrixDiagonal[tempFlatPos] * invLaplace;
					const scalar_t raL = 1/advectionMatrixDiagonal[tempFlatPos];
					rowValues[0] -= raL*0.5f;
					rowValues[bound+1] = 0.5f*(ra + raL);
					indices[bound+1] = tempFlatPos;
				}else{ //dim==0 && empty bound
					if(c_domain.bounds[bound].type==BoundaryType::VALUE){
						//remove excess boundary coefficients for 0 gadient boundary
						rowValues[0] += ra*0.5f;
					}
					indices[bound+1] = -1; //invalid/unused
				}
				//upper boundary
				++bound;
				if(pos.a[dim]!=c_domain.sizes[dim]-1 || !isEmptyBound(bound, c_domain.bounds)){
					I4 tempPos = {.v=pos.v};
					if(pos.a[dim]==c_domain.sizes[dim]-1 && c_domain.bounds[bound].type==BoundaryType::PERIODIC){
						tempPos.a[dim] = 0;
					}else{
						tempPos.a[dim] = pos.a[dim]+1;
					}
					const int tempFlatPos = flattenIndex(tempPos.v);
					//rowValues[bound+1] = -1 / advectionMatrixDiagonal[tempFlatPos] * invLaplace;
					const scalar_t raU = 1/advectionMatrixDiagonal[tempFlatPos];
					rowValues[0] -= raU*0.5f;
					rowValues[bound+1] = 0.5f*(ra + raU);
					indices[bound+1] = tempFlatPos;
				}else{
					if(c_domain.bounds[bound].type==BoundaryType::VALUE){
						rowValues[0] += ra*0.5f;
					}
					indices[bound+1] = -1; //invalid/unused
				}
			}
			for(int dim=c_domain.numDims;dim<3;++dim){
				indices[dim*2+1] = -1; //invalid/unused
				indices[dim*2+2] = -1; //invalid/unused
			}
			
			// sort, naive for now
			for(int i=0;i<row.size;++i){
				int colIndex = findLowestColumnIndex(indices, 7);
				
				if(colIndex<0){
					csrMatrixIndex[rowStartOffset + i] = -1;
					csrMatrixValue[rowStartOffset + i] = -1.0f;
				}else{
					csrMatrixIndex[rowStartOffset + i] = indices[colIndex];
					csrMatrixValue[rowStartOffset + i] = rowValues[colIndex];
				}
				
				indices[colIndex] = -1;
			}
		}
	}
}

template <typename scalar_t>
__global__ void PISO_build_pressure_rhs(
		const scalar_t* __restrict__ velocity, const scalar_t* __restrict__ velocityPrediction,
		const scalar_t* __restrict__ advectionMatrixDiagonal,
		const int32_t* __restrict__ csrMatrixIndex, const int32_t* __restrict__ csrMatrixRow, const scalar_t* __restrict__ csrMatrixValue,
		scalar_t* __restrict__ rhs){
	
	// 
	
	const int threadStride = gridDim.x*blockDim.x;
	const int globalThreadIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int repetitions = divCeil(c_domain.stride.w, threadStride);
	
	for(int r=0;r<repetitions;++r){
		const int flatPos = r*threadStride + globalThreadIndex;
		if(flatPos<c_domain.stride.w){
			const int4 pos = unflattenIndex(flatPos);
			const scalar_t rDiag = 1 / advectionMatrixDiagonal[flatPos];
			
			// load row from CSR matrix
			int32_t csrIndices[7];
			scalar_t csrValues[7];
			const int32_t csrStart = csrMatrixRow[flatPos];
			const int32_t csrEnd = csrMatrixRow[flatPos+1];
			const int32_t rowSize = csrEnd - csrStart;
			for(int32_t i=0;i<rowSize && i<7;++i){
				csrIndices[i] = csrMatrixIndex[csrStart+i];
				csrValues[i] = csrMatrixValue[csrStart+i];
			}
			
			for(int dim=0;dim<c_domain.numDims;++dim){
				int4 compPos = pos;
				compPos.w = dim;
				const int flatCompPos = flattenIndex(compPos);
				
				const scalar_t vel = velocity[flatCompPos]/c_domain.timeStep;
				
				scalar_t H = 0;
				for(int32_t i=0;i<rowSize;++i){
					int32_t idx = csrIndices[i];
					if(idx!=flatPos){ //omit diagonal entries
						//int4 velPos = unflattenIndex(idx);
						//velPos.w = dim;
						// alternative: idx + dim*c_domain.stride.w
						H += csrValues[i] * velocityPrediction[idx + dim*c_domain.stride.w]; //H'u*
					}
				}
				
				rhs[flatCompPos] = rDiag *(vel-H);
				
				//TODO external forces/gravity
			}
		}
	}
}

template <typename scalar_t>
__global__ void PISO_build_pressure_rhs_div_v2(
		const scalar_t* __restrict__ pressureRHS,
		const scalar_t* __restrict__ advectionMatrixDiagonal,
		scalar_t* __restrict__ divergence){
	
	// 
	
	const int threadStride = gridDim.x*blockDim.x;
	const int globalThreadIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int repetitions = divCeil(c_domain.stride.w, threadStride);
	
	for(int r=0;r<repetitions;++r){
		const int flatPos = r*threadStride + globalThreadIndex;
		if(flatPos<c_domain.stride.w){
			const I4 pos = {.v=unflattenIndex(flatPos)};
			const scalar_t diag = advectionMatrixDiagonal[flatPos];
			const scalar_t rDiag = 1/diag;
			const scalar_t HC = pressureRHS[flatPos]*diag; // undo *rDiag from before
			scalar_t result = 0;
			
			for(int dim=0;dim<c_domain.numDims;++dim){
				
				//lower neighbour
				{
					I4 tempPos = pos;
					tempPos.a[dim] = pos.a[dim] -1;
					scalar_t diagL = getPressureAtWithBounds(advectionMatrixDiagonal, tempPos, c_domain);
					scalar_t rDiagL = 1/diagL;
					tempPos.v.w = dim;
					scalar_t HL = getVelocityAtWithBounds(pressureRHS, tempPos.v, c_domain) * diagL;
					//result -= 0.5*(rDiag + rDiagL) * 0.5 * (HC + HL)
					//result -= 0.25 * (rDiag + rDiagL) * (HC + HL)
					//result -= 0.25 * (rDiag*HC + rDiagL*HC + rDiag*HL + rDiagL*HL)
					//result -= 0.25 * (rDiagL*HC + rDiag*HL + rDiagL*HL) //rDiag*HC cancels with other side
					result -= 0.25 * (rDiagL*HC + (rDiag + rDiagL)*HL);
				}
				
				//upper neighbour
				{
					I4 tempPos = pos;
					tempPos.a[dim] = pos.a[dim] +1;
					scalar_t diagU = getPressureAtWithBounds(advectionMatrixDiagonal, tempPos, c_domain);
					scalar_t rDiagU = 1/diagU;
					tempPos.v.w = dim;
					scalar_t HU = getVelocityAtWithBounds(pressureRHS, tempPos.v, c_domain) * diagU;
					
					result += 0.25 * (rDiagU*HC + (rDiag + rDiagU)*HU);
				}
			}
			divergence[flatPos] = result;
		}
	}
}

template <typename scalar_t>
__global__ void compute_divergence(const scalar_t* __restrict__ vectorField, scalar_t* __restrict__ divergence){
	
	// divergence of colocated vector field
	// using central differences
	
	const int threadStride = gridDim.x*blockDim.x;
	const int globalThreadIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int repetitions = divCeil(c_domain.stride.w, threadStride);
	
	for(int r=0;r<repetitions;++r){
		const int flatPos = r*threadStride + globalThreadIndex;
		if(flatPos<c_domain.stride.w){
			const I4 pos = {.v=unflattenIndex(flatPos)};
			//int4 tempPos = pos;
			
			scalar_t div = 0;
			for(int dim=0;dim<c_domain.numDims;++dim){
				//dim=1;
				I4 tempPos = {.v=pos.v};
				tempPos.v.w = dim;
				tempPos.a[dim] = pos.a[dim]-1;
				const scalar_t valN = getVelocityAtWithBounds(vectorField, tempPos.v, c_domain);
				tempPos.a[dim] = pos.a[dim]+1;
				const scalar_t valP = getVelocityAtWithBounds(vectorField, tempPos.v, c_domain);
				div += (valP - valN)*0.5;
				//break; //debug
			}
			
			divergence[flatPos] = div;
		}
	}
}


template <typename scalar_t>
__global__ void PISO_update_velocity(const scalar_t* __restrict__ advectionMatrixDiagonal, const scalar_t* __restrict__ pressureRHS, const scalar_t* __restrict__ pressure, scalar_t* __restrict__ velocityPrediction){
	//vel.x = pressureRHS.x - inv(A)*gradX(pressure)
	
	const int threadStride = gridDim.x*blockDim.x;
	const int globalThreadIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const int repetitions = divCeil(c_domain.stride.w, threadStride);
	
	for(int r=0;r<repetitions;++r){
		const int flatPos = r*threadStride + globalThreadIndex;
		if(flatPos<c_domain.stride.w){
			const I4 pos = {.v=unflattenIndex(flatPos)};
			
			for(int dim=0;dim<c_domain.numDims;++dim){
				I4 tempPos = pos;
				tempPos.v.w = 0;
				tempPos.a[dim] = pos.a[dim]-1;
				const scalar_t valN = getPressureAtWithBounds(pressure, tempPos, c_domain);
				tempPos.a[dim] = pos.a[dim]+1;
				const scalar_t valP = getPressureAtWithBounds(pressure, tempPos, c_domain);
				const scalar_t diag_inv = 1/advectionMatrixDiagonal[flatPos];
				scalar_t velUpdate = - diag_inv * (valP - valN)*0.5;
				//scalar_t velUpdate = valP;
				
				tempPos.v.w = dim;
				tempPos.a[dim] = pos.a[dim];
				const int flatCompPos = flattenIndex(tempPos.v);
				velUpdate += pressureRHS[flatCompPos];
				velocityPrediction[flatCompPos] = velUpdate;
			}
		}
	}
}

__host__ void SetDomain(Domain<float> &domain, torch::Tensor vectorField, const float timeStep, const float viscosity, const float *slipData, const int32_t *boundsData, const float *boundsValues){
	domain.numDims = vectorField.dim()-2;
	domain.timeStep = timeStep;
	domain.viscosity = viscosity;

	switch(domain.numDims){
		case 1:
			domain.size = make_int4(vectorField.size(2), 1, 1, vectorField.size(1)); //xyzc
			break;
		case 2:
			domain.size = make_int4(vectorField.size(3), vectorField.size(2), 1, vectorField.size(1)); //xyzc
			break;
		case 3:
			domain.size = make_int4(vectorField.size(4), vectorField.size(3), vectorField.size(2), vectorField.size(1)); //xyzc
			break;
		default:
			std::cerr << "unsupported number of spatial dimensions: " << domain.numDims << std::endl;
			exit(10);
			break;
	}
	domain.stride = make_int4(1, domain.size.x, domain.size.x*domain.size.y, domain.size.x*domain.size.y*domain.size.z);

	if(boundsData!=nullptr && boundsValues!=nullptr){
		for(int i=0;i<domain.numDims*2;++i){
			domain.bounds[i].type = static_cast<BoundaryType>(boundsData[i]);
			domain.bounds[i].prescribedValue = boundsValues[i];
			domain.bounds[i].slip = 0;
		}
	}
	CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_domain, &domain, sizeof(Domain<float>)));
}

/* std::vector<torch::Tensor> PISO_step_cuda_forward(torch::Tensor data, torch::Tensor velocity, torch::Tensor pressureGuess,
		torch::Tensor boundariesCPU, torch::Tensor boundaryValuesCPU,
		torch::Tensor stepsCPU, torch::Tensor correctorStepsCPU, torch::Tensor timeStepCPU, torch::Tensor epsCPU,
		const int spatial_dimensions){
	//const int steps, const int correctorSteps, const scalar_t timeStep
	//velocity shape: NCDHW, channels first
	const auto batch_size = velocity.size(0);
	//const auto spatial_dimensions = velocity.dim()-2;
	Domain<float> domain;
	SetDomain(domain, velocity, timeStepCPU.data_ptr<float>()[0], 0, boundariesCPU.data_ptr<int32_t>(), boundaryValuesCPU.data_ptr<float>());


	int32_t steps = stepsCPU.data_ptr<int32_t>()[0];
	float eps = epsCPU.data_ptr<float>()[0];
	
	int sparseMatrixSize = getCSRSize(domain);
	LOG("CSR matrix size: " << sparseMatrixSize);
	auto valueOptions = torch::TensorOptions().dtype(velocity.dtype()).layout(torch::kStrided).device(velocity.device().type(), velocity.device().index());
	auto indexOptions = torch::TensorOptions().dtype(boundariesCPU.dtype()).layout(torch::kStrided).device(velocity.device().type(), velocity.device().index());
	auto csrMatrixValue = torch::zeros(sparseMatrixSize, valueOptions);
	auto csrMatrixIndex = torch::zeros(sparseMatrixSize, indexOptions);
	auto csrMatrixRow = torch::zeros(domain.stride.w+1, indexOptions);
	auto matrixDiagonal = torch::zeros(domain.stride.w, valueOptions);
	
	auto dataAdvectionRHS = torch::zeros_like(data);
	auto velocityPrediction = torch::zeros_like(velocity);
	auto velocityAdvectionRHS = torch::zeros_like(velocity);
	
	//pressure solve
	int32_t correctorSteps = correctorStepsCPU.data_ptr<int32_t>()[0];
	auto pMatrixValue = torch::zeros(sparseMatrixSize, valueOptions);
	auto pMatrixIndex = torch::zeros(sparseMatrixSize, indexOptions);
	auto pMatrixRow = torch::zeros(domain.stride.w+1, indexOptions);
	auto pressureRHS = torch::zeros_like(velocity);
	auto pressureRHSdiv = torch::zeros(domain.stride.w, valueOptions);
	//auto pressureUpdate = torch::zeros(domain.stride.w, valueOptions);
	
	//cublasHandle_t blasHandle = NULL;
	//cusparseHandle_t sparseHandle = NULL;
	cusolverSpHandle_t solverHandle = NULL;
	cusparseMatDescr_t descrA = NULL;
	
	
	//CUBLAS_CHECK_RETURN(cublasCreate_v2(&blasHandle));
	//CUSPARSE_CHECK_RETURN(cusparseCreate(&sparseHandle));
	CUSOLVER_CHECK_RETURN(cusolverSpCreate(&solverHandle));
	CUSPARSE_CHECK_RETURN(cusparseCreateMatDescr(&descrA));
	CUSPARSE_CHECK_RETURN(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
	CUSPARSE_CHECK_RETURN(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
	
	int singularity = -1;
	
	const int threads = 1024;
	const dim3 blocks((domain.stride.w + threads - 1) / threads, batch_size);
	const auto dispatchType = velocity.scalar_type();
	
	for(int32_t step=0; step<steps; ++step){
	
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		LOG("Dispatch A matrix kernel");
		BEGIN_SAMPLE;
		AT_DISPATCH_FLOATING_TYPES(dispatchType, "PISO_build_matrix", ([&] {
			PISO_build_matrix<scalar_t><<<blocks, threads>>>(
				velocity.data_ptr<scalar_t>(),
				pressureGuess.data_ptr<scalar_t>(),
				csrMatrixIndex.data_ptr<int32_t>(),
				csrMatrixRow.data_ptr<int32_t>(),
				csrMatrixValue.data_ptr<scalar_t>(),
				matrixDiagonal.data_ptr<scalar_t>()
			);
		}));
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		END_SAMPLE("Build A-Matrix");
		
		// advect data
		
		BEGIN_SAMPLE;
		AT_DISPATCH_FLOATING_TYPES(dispatchType, "kPISO_build_scalar_advection_RHS", ([&] {
			kPISO_build_scalar_advection_RHS<scalar_t><<<blocks, threads>>>(
				data.data_ptr<scalar_t>(),
				dataAdvectionRHS.data_ptr<scalar_t>()
			);
		}));
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		END_SAMPLE("Velocity Advection RHS");
		
		LOG("Dispatch advection solver kernel");
		BEGIN_SAMPLE;
		AT_DISPATCH_FLOATING_TYPES(dispatchType, "solveDensityAdvection", ([&] {
			cusolverSpTcsrlsvlu<scalar_t>( solverHandle,
				domain.stride.w, sparseMatrixSize,
				descrA,
				csrMatrixValue.data_ptr<scalar_t>(),
				csrMatrixRow.data_ptr<int32_t>(),
				csrMatrixIndex.data_ptr<int32_t>(),
				dataAdvectionRHS.data_ptr<scalar_t>(), //RHS, TODO: add pressure guess?
				eps,
				0, //reorder
				data.data_ptr<scalar_t>(), // result
				&singularity
			);
		}));
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		END_SAMPLE("Solve Advection");
		LOG("singularity: " << singularity);
		
		// advect velocity
		
		BEGIN_SAMPLE;
		AT_DISPATCH_FLOATING_TYPES(dispatchType, "kPISO_build_advection_RHS", ([&] {
			kPISO_build_advection_RHS<scalar_t><<<blocks, threads>>>(
				velocity.data_ptr<scalar_t>(),
				pressureGuess.data_ptr<scalar_t>(),
				velocityAdvectionRHS.data_ptr<scalar_t>()
			);
		}));
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		END_SAMPLE("Velocity Advection RHS");
		
		LOG("Dispatch advection solver kernel");
		BEGIN_SAMPLE;
		// should this be coupled for all components?
		for(int component=0;component<domain.numDims;++component){
			AT_DISPATCH_FLOATING_TYPES(dispatchType, "solveVelocityAdvection", ([&] {
				cusolverSpTcsrlsvlu<scalar_t>( solverHandle,
					domain.stride.w, sparseMatrixSize,
					descrA,
					csrMatrixValue.data_ptr<scalar_t>(),
					csrMatrixRow.data_ptr<int32_t>(),
					csrMatrixIndex.data_ptr<int32_t>(),
					velocityAdvectionRHS.data_ptr<scalar_t>() + (component*domain.stride.w), //RHS, the data being advected
					eps,
					0, //reorder
					velocityPrediction.data_ptr<scalar_t>() + (component*domain.stride.w), // result
					&singularity
				);
			}));
		}
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		END_SAMPLE("Solve Advection");
		LOG("singularity: " << singularity);
		
		// PRESSURE SOLVE
		
		// create LHS, this part is constant
		BEGIN_SAMPLE;
		AT_DISPATCH_FLOATING_TYPES(dispatchType, "PISO_build_pressure_matrix", ([&] {
			PISO_build_pressure_matrix<scalar_t><<<blocks, threads>>>(
				matrixDiagonal.data_ptr<scalar_t>(),
				pMatrixIndex.data_ptr<int32_t>(),
				pMatrixRow.data_ptr<int32_t>(),
				pMatrixValue.data_ptr<scalar_t>()
			);
		}));
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		END_SAMPLE("Build p-Matrix");
		
		for(int32_t correctorStep=0; correctorStep<correctorSteps; ++correctorStep){
			// create RHS, uses current velocityPrediction
			BEGIN_SAMPLE;
			AT_DISPATCH_FLOATING_TYPES(dispatchType, "PISO_pressure_RHS", ([&] {
				PISO_build_pressure_rhs<scalar_t><<<blocks, threads>>>(
					velocity.data_ptr<scalar_t>(), velocityPrediction.data_ptr<scalar_t>(),
					matrixDiagonal.data_ptr<scalar_t>(),
					csrMatrixIndex.data_ptr<int32_t>(),
					csrMatrixRow.data_ptr<int32_t>(),
					csrMatrixValue.data_ptr<scalar_t>(),
					pressureRHS.data_ptr<scalar_t>()
				);
			}));
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			END_SAMPLE("Build p-RHS");
			
			BEGIN_SAMPLE;
			AT_DISPATCH_FLOATING_TYPES(dispatchType, "PISO_pressure_RHS_divergence", ([&] {
				compute_divergence<scalar_t><<<blocks, threads>>>(
					pressureRHS.data_ptr<scalar_t>(),
					pressureRHSdiv.data_ptr<scalar_t>()
				);
			}));
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			END_SAMPLE("Divergence p-RHS");
			
			// BEGIN_SAMPLE;
			// AT_DISPATCH_FLOATING_TYPES(dispatchType, "PISO_build_pressure_rhs_div_v2", ([&] {
				// PISO_build_pressure_rhs_div_v2<scalar_t><<<blocks, threads>>>(
					// pressureRHS.data_ptr<scalar_t>(),
					// matrixDiagonal.data_ptr<scalar_t>(),
					// pressureRHSdiv.data_ptr<scalar_t>()
				// );
			// }));
			// CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			// END_SAMPLE("Divergence p-RHS");
			
			// solve pressure (step 1)
			BEGIN_SAMPLE;
			AT_DISPATCH_FLOATING_TYPES(dispatchType, "solvePressure", ([&] {
				cusolverSpTcsrlsvlu<scalar_t>( solverHandle,
					domain.stride.w, sparseMatrixSize,
					descrA,
					pMatrixValue.data_ptr<scalar_t>(),
					pMatrixRow.data_ptr<int32_t>(),
					pMatrixIndex.data_ptr<int32_t>(),
					pressureRHSdiv.data_ptr<scalar_t>(), //RHS
					eps,
					0, //reorder
					pressureGuess.data_ptr<scalar_t>(), // result
					&singularity
				);
			}));
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			END_SAMPLE("Solve Pressure");
			LOG("singularity: " << singularity);
			
			// correct velocity
			BEGIN_SAMPLE;
			AT_DISPATCH_FLOATING_TYPES(dispatchType, "solvePressure", ([&] {
				PISO_update_velocity<scalar_t><<<blocks, threads>>>( 
					matrixDiagonal.data_ptr<scalar_t>(),
					pressureRHS.data_ptr<scalar_t>(), //RHS
					pressureGuess.data_ptr<scalar_t>(),
					velocityPrediction.data_ptr<scalar_t>()
				);
			}));
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			END_SAMPLE("Update Velocity");
			//
		}
		
		
		BEGIN_SAMPLE;
		AT_DISPATCH_FLOATING_TYPES(dispatchType, "copyResult", ([&] {
			//CUDA_CHECK_RETURN(cudaMemcpy(data.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), domain.stride.w*sizeof(scalar_t), cudaMemcpyDeviceToDevice)); //dst, src, bytes, kind
			CUDA_CHECK_RETURN(cudaMemcpy(velocity.data_ptr<scalar_t>(), velocityPrediction.data_ptr<scalar_t>(), domain.stride.w*domain.numDims*sizeof(scalar_t), cudaMemcpyDeviceToDevice));
			//TODO: might need to reset some of the helper fields.
		}));
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		END_SAMPLE("Copy Result");
	}
	
	//cusparseDestroy(sparseHandle);
	//cublasDestroy(blasHandle);
	cusolverSpDestroy(solverHandle);
	
	return std::vector<torch::Tensor>{data, velocity, pressureGuess, matrixDiagonal,
		csrMatrixRow, csrMatrixIndex, csrMatrixValue, 
		pMatrixRow, pMatrixIndex, pMatrixValue, 
		pressureRHS, pressureRHSdiv};
} */


// --- Separated Steps ---

std::vector<torch::Tensor> SetupAdvectionMatrixEulerImplicite(torch::Tensor velocity, torch::Tensor pressureGuess,
		torch::Tensor timeStepCPU, torch::Tensor viscosityCPU, torch::Tensor boundariesCPU, torch::Tensor boundaryValuesCPU){
	// setup domain
	Domain<float> domain;
	SetDomain(domain, velocity, timeStepCPU.data_ptr<float>()[0], viscosityCPU.data_ptr<float>()[0], boundariesCPU.data_ptr<int32_t>(), boundaryValuesCPU.data_ptr<float>());
	int sparseMatrixSize = getCSRSize(domain);

	// allocate temp and output
	auto valueOptions = torch::TensorOptions().dtype(velocity.dtype()).layout(torch::kStrided).device(velocity.device().type(), velocity.device().index());
	auto indexOptions = torch::TensorOptions().dtype(boundariesCPU.dtype()).layout(torch::kStrided).device(velocity.device().type(), velocity.device().index());
	auto csrMatrixValue = torch::zeros(sparseMatrixSize, valueOptions);
	auto csrMatrixIndex = torch::zeros(sparseMatrixSize, indexOptions);
	auto csrMatrixRow = torch::zeros(domain.stride.w+1, indexOptions);
	auto matrixDiagonal = torch::zeros(domain.stride.w, valueOptions);

	const int threads = 1024;
	const dim3 blocks((domain.stride.w + threads - 1) / threads, 1);
	const auto dispatchType = velocity.scalar_type();

	// make A matrix
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	LOG("Dispatch A matrix kernel");
	BEGIN_SAMPLE;
	AT_DISPATCH_FLOATING_TYPES(dispatchType, "PISO_build_matrix", ([&] {
		PISO_build_matrix<scalar_t><<<blocks, threads>>>(
			velocity.data_ptr<scalar_t>(),
			nullptr, //pressureGuess.data_ptr<scalar_t>(),
			csrMatrixIndex.data_ptr<int32_t>(),
			csrMatrixRow.data_ptr<int32_t>(),
			csrMatrixValue.data_ptr<scalar_t>(),
			matrixDiagonal.data_ptr<scalar_t>()
		);
	}));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Build A-Matrix");

	return std::vector<torch::Tensor>{matrixDiagonal, csrMatrixRow, csrMatrixIndex, csrMatrixValue};
}

std::vector<torch::Tensor> SetupAdvectionScalarEulerImpliciteRHS(torch::Tensor scalarData,
		torch::Tensor timeStepCPU, torch::Tensor boundariesCPU, torch::Tensor boundaryValuesCPU){
	// setup domain
	Domain<float> domain;
	SetDomain(domain, scalarData, timeStepCPU.data_ptr<float>()[0], 0, boundariesCPU.data_ptr<int32_t>(), boundaryValuesCPU.data_ptr<float>());

	// allocate temp and output
	auto dataAdvectionRHS = torch::zeros_like(scalarData);

	const int threads = 1024;
	const dim3 blocks((domain.stride.w + threads - 1) / threads, 1);
	const auto dispatchType = scalarData.scalar_type();

	// make rhs
		
	BEGIN_SAMPLE;
	AT_DISPATCH_FLOATING_TYPES(dispatchType, "kPISO_build_scalar_advection_RHS", ([&] {
		kPISO_build_scalar_advection_RHS<scalar_t><<<blocks, threads>>>(
			scalarData.data_ptr<scalar_t>(),
			dataAdvectionRHS.data_ptr<scalar_t>()
		);
	}));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("scalar Advection RHS");

	return std::vector<torch::Tensor>{dataAdvectionRHS};
}

std::vector<torch::Tensor> SolveAdvectionScalarEulerImplicite(torch::Tensor csrMatrixValue, torch::Tensor csrMatrixRow, torch::Tensor csrMatrixIndex, torch::Tensor dataAdvectionRHS,
		torch::Tensor epsCPU){
	// setup domain
	Domain<float> domain;
	SetDomain(domain, dataAdvectionRHS, 1, 0, nullptr, nullptr);
	int sparseMatrixSize = csrMatrixValue.size(0);

	// allocate temp and output
	auto dataResult = torch::zeros_like(dataAdvectionRHS);
	float eps = epsCPU.data_ptr<float>()[0];
	int singularity = -1;
	const auto dispatchType = dataAdvectionRHS.scalar_type();

	// solve
	cusolverSpHandle_t solverHandle = NULL;
	cusparseMatDescr_t descrA = NULL;
	CUSOLVER_CHECK_RETURN(cusolverSpCreate(&solverHandle));
	CUSPARSE_CHECK_RETURN(cusparseCreateMatDescr(&descrA));
	CUSPARSE_CHECK_RETURN(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
	CUSPARSE_CHECK_RETURN(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
		
	LOG("Dispatch advection solver kernel");
	BEGIN_SAMPLE;
	AT_DISPATCH_FLOATING_TYPES(dispatchType, "solveDensityAdvection", ([&] {
		cusolverSpTcsrlsvlu<scalar_t>( solverHandle,
			domain.stride.w, sparseMatrixSize,
			descrA,
			csrMatrixValue.data_ptr<scalar_t>(),
			csrMatrixRow.data_ptr<int32_t>(),
			csrMatrixIndex.data_ptr<int32_t>(),
			dataAdvectionRHS.data_ptr<scalar_t>(), //RHS
			eps,
			0, //reorder
			dataResult.data_ptr<scalar_t>(), // result
			&singularity
		);
	}));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Solve Advection");
	LOG("singularity: " << singularity);

	cusolverSpDestroy(solverHandle);

	return std::vector<torch::Tensor>{dataResult};
}

std::vector<torch::Tensor> SetupAdvectionVelocityEulerImpliciteRHS(torch::Tensor velocity, torch::Tensor pressureGuess,
		torch::Tensor timeStepCPU, torch::Tensor boundariesCPU, torch::Tensor boundaryValuesCPU){
	// setup domain
	Domain<float> domain;
	SetDomain(domain, velocity, timeStepCPU.data_ptr<float>()[0], 0, boundariesCPU.data_ptr<int32_t>(), boundaryValuesCPU.data_ptr<float>());

	// allocate temp and output
	auto velocityAdvectionRHS = torch::zeros_like(velocity);

	const int threads = 1024;
	const dim3 blocks((domain.stride.w + threads - 1) / threads, 1);
	const auto dispatchType = velocity.scalar_type();

	// make rhs
	BEGIN_SAMPLE;
	AT_DISPATCH_FLOATING_TYPES(dispatchType, "kPISO_build_advection_RHS", ([&] {
		kPISO_build_advection_RHS<scalar_t><<<blocks, threads>>>(
			velocity.data_ptr<scalar_t>(),
			pressureGuess.data_ptr<scalar_t>(),
			velocityAdvectionRHS.data_ptr<scalar_t>()
		);
	}));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Velocity Advection RHS");
	
	return std::vector<torch::Tensor>{velocityAdvectionRHS};
}

std::vector<torch::Tensor> SolveAdvectionVelocityEulerImplicite(torch::Tensor csrMatrixValue, torch::Tensor csrMatrixRow, torch::Tensor csrMatrixIndex, torch::Tensor velocityAdvectionRHS,
		torch::Tensor epsCPU){
	// setup domain
	Domain<float> domain;
	SetDomain(domain, velocityAdvectionRHS, 1, 0, nullptr, nullptr);
	int sparseMatrixSize = csrMatrixValue.size(0);

	// allocate temp and output
	auto velocityResult = torch::zeros_like(velocityAdvectionRHS);
	float eps = epsCPU.data_ptr<float>()[0];
	int singularity = -1;
	const auto dispatchType = velocityAdvectionRHS.scalar_type();

	// solve
	cusolverSpHandle_t solverHandle = NULL;
	cusparseMatDescr_t descrA = NULL;
	CUSOLVER_CHECK_RETURN(cusolverSpCreate(&solverHandle));
	CUSPARSE_CHECK_RETURN(cusparseCreateMatDescr(&descrA));
	CUSPARSE_CHECK_RETURN(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
	CUSPARSE_CHECK_RETURN(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

	// solve
	LOG("Dispatch advection solver kernel");
	BEGIN_SAMPLE;
	// should this be coupled for all components?
	for(int component=0;component<domain.numDims;++component){
		AT_DISPATCH_FLOATING_TYPES(dispatchType, "solveVelocityAdvection", ([&] {
			cusolverSpTcsrlsvlu<scalar_t>( solverHandle,
				domain.stride.w, sparseMatrixSize,
				descrA,
				csrMatrixValue.data_ptr<scalar_t>(),
				csrMatrixRow.data_ptr<int32_t>(),
				csrMatrixIndex.data_ptr<int32_t>(),
				velocityAdvectionRHS.data_ptr<scalar_t>() + (component*domain.stride.w), //RHS, the data being advected
				eps,
				0, //reorder
				velocityResult.data_ptr<scalar_t>() + (component*domain.stride.w), // result
				&singularity
			);
		}));
	}
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Solve Advection");
	LOG("singularity: " << singularity);
	
	cusolverSpDestroy(solverHandle);

	return std::vector<torch::Tensor>{velocityResult};
}


std::vector<torch::Tensor> SetupPressurePISO(torch::Tensor velocity, torch::Tensor velocityPrediction,
		torch::Tensor matrixDiagonal, torch::Tensor csrMatrixValue, torch::Tensor csrMatrixRow, torch::Tensor csrMatrixIndex,
		torch::Tensor timeStepCPU, torch::Tensor boundariesCPU, torch::Tensor boundaryValuesCPU){
	// setup domain
	Domain<float> domain;
	SetDomain(domain, velocity, timeStepCPU.data_ptr<float>()[0], 0, boundariesCPU.data_ptr<int32_t>(), boundaryValuesCPU.data_ptr<float>());
	int sparseMatrixSize = getCSRSize(domain);

	// allocate temp and output
	auto valueOptions = torch::TensorOptions().dtype(velocity.dtype()).layout(torch::kStrided).device(velocity.device().type(), velocity.device().index());
	auto indexOptions = torch::TensorOptions().dtype(boundariesCPU.dtype()).layout(torch::kStrided).device(velocity.device().type(), velocity.device().index());
	auto pMatrixValue = torch::zeros(sparseMatrixSize, valueOptions);
	auto pMatrixIndex = torch::zeros(sparseMatrixSize, indexOptions);
	auto pMatrixRow = torch::zeros(domain.stride.w+1, indexOptions);
	auto pressureRHS = torch::zeros_like(velocity);
	std::vector<int64_t> velocityShape(velocity.dim(),1);
	for(int dim=0;dim<velocity.dim();++dim){
		velocityShape[dim] = velocity.size(dim);
	}
	velocityShape[1] = 1;
	auto pressureRHSdiv = torch::zeros(velocityShape, valueOptions);
	//LOG("pressure div: " << pressureRHSdiv.sizes());

	const int threads = 1024;
	const dim3 blocks((domain.stride.w + threads - 1) / threads, 1);
	const auto dispatchType = velocity.scalar_type();

	// make P matrix
	BEGIN_SAMPLE;
	AT_DISPATCH_FLOATING_TYPES(dispatchType, "PISO_build_pressure_matrix", ([&] {
		PISO_build_pressure_matrix<scalar_t><<<blocks, threads>>>(
			matrixDiagonal.data_ptr<scalar_t>(),
			pMatrixIndex.data_ptr<int32_t>(),
			pMatrixRow.data_ptr<int32_t>(),
			pMatrixValue.data_ptr<scalar_t>()
		);
	}));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Build p-Matrix");
	
	// make P rhs
	BEGIN_SAMPLE;
	AT_DISPATCH_FLOATING_TYPES(dispatchType, "PISO_pressure_RHS", ([&] {
		PISO_build_pressure_rhs<scalar_t><<<blocks, threads>>>(
			velocity.data_ptr<scalar_t>(), velocityPrediction.data_ptr<scalar_t>(),
			matrixDiagonal.data_ptr<scalar_t>(),
			csrMatrixIndex.data_ptr<int32_t>(),
			csrMatrixRow.data_ptr<int32_t>(),
			csrMatrixValue.data_ptr<scalar_t>(),
			pressureRHS.data_ptr<scalar_t>()
		);
	}));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Build p-RHS");
	
	// divergence P rhs
	BEGIN_SAMPLE;
	AT_DISPATCH_FLOATING_TYPES(dispatchType, "PISO_pressure_RHS_divergence", ([&] {
		compute_divergence<scalar_t><<<blocks, threads>>>(
			pressureRHS.data_ptr<scalar_t>(),
			pressureRHSdiv.data_ptr<scalar_t>()
		);
	}));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Divergence p-RHS"); 
	
	/* BEGIN_SAMPLE;
	AT_DISPATCH_FLOATING_TYPES(dispatchType, "PISO_build_pressure_rhs_div_v2", ([&] {
		PISO_build_pressure_rhs_div_v2<scalar_t><<<blocks, threads>>>(
			pressureRHS.data_ptr<scalar_t>(),
			matrixDiagonal.data_ptr<scalar_t>(),
			pressureRHSdiv.data_ptr<scalar_t>()
		);
	}));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Divergence p-RHS");*/


	return std::vector<torch::Tensor>{pMatrixRow, pMatrixIndex, pMatrixValue, pressureRHS, pressureRHSdiv};
}

std::vector<torch::Tensor> SolvePressurePISO(torch::Tensor pMatrixValue, torch::Tensor pMatrixRow, torch::Tensor pMatrixIndex, torch::Tensor pressureRHSdiv,
		torch::Tensor epsCPU){
	// setup domain
	Domain<float> domain;
	SetDomain(domain, pressureRHSdiv, 1, 0, nullptr, nullptr);
	int sparseMatrixSize = pMatrixValue.size(0);

	// allocate temp and output
	auto pressureResult = torch::zeros_like(pressureRHSdiv);
	float eps = epsCPU.data_ptr<float>()[0];
	int singularity = -1;
	const auto dispatchType = pressureRHSdiv.scalar_type();

	// solve
	cusolverSpHandle_t solverHandle = NULL;
	cusparseMatDescr_t descrA = NULL;
	CUSOLVER_CHECK_RETURN(cusolverSpCreate(&solverHandle));
	CUSPARSE_CHECK_RETURN(cusparseCreateMatDescr(&descrA));
	CUSPARSE_CHECK_RETURN(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
	CUSPARSE_CHECK_RETURN(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));


	BEGIN_SAMPLE;
	AT_DISPATCH_FLOATING_TYPES(dispatchType, "solvePressure", ([&] {
		cusolverSpTcsrlsvlu<scalar_t>( solverHandle,
			domain.stride.w, sparseMatrixSize,
			descrA,
			pMatrixValue.data_ptr<scalar_t>(),
			pMatrixRow.data_ptr<int32_t>(),
			pMatrixIndex.data_ptr<int32_t>(),
			pressureRHSdiv.data_ptr<scalar_t>(), //RHS
			eps,
			0, //reorder
			pressureResult.data_ptr<scalar_t>(), // result
			&singularity
		);
	}));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Solve Pressure");
	LOG("singularity: " << singularity);
	
	cusolverSpDestroy(solverHandle);

	return std::vector<torch::Tensor>{pressureResult};

}

std::vector<torch::Tensor> CorrectVelocityPISO(torch::Tensor matrixDiagonal, torch::Tensor pressureRHS, torch::Tensor pressureGuess,
		torch::Tensor timeStepCPU, torch::Tensor boundariesCPU, torch::Tensor boundaryValuesCPU){
	// setup domain
	Domain<float> domain;
	SetDomain(domain, pressureRHS, timeStepCPU.data_ptr<float>()[0], 0, boundariesCPU.data_ptr<int32_t>(), boundaryValuesCPU.data_ptr<float>());

	// allocate temp and output
	auto velocityPrediction = torch::zeros_like(pressureRHS);

	const int threads = 1024;
	const dim3 blocks((domain.stride.w + threads - 1) / threads, 1);
	const auto dispatchType = pressureRHS.scalar_type();

	
	BEGIN_SAMPLE;
	AT_DISPATCH_FLOATING_TYPES(dispatchType, "solvePressure", ([&] {
		PISO_update_velocity<scalar_t><<<blocks, threads>>>( 
			matrixDiagonal.data_ptr<scalar_t>(),
			pressureRHS.data_ptr<scalar_t>(), //RHS
			pressureGuess.data_ptr<scalar_t>(),
			velocityPrediction.data_ptr<scalar_t>()
		);
	}));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Update Velocity");

	return std::vector<torch::Tensor>{velocityPrediction};

}
