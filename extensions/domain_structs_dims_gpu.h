#pragma once

#ifndef _INCLUDE_DOMAIN_STRUCTS_DIMS_GPU
#define _INCLUDE_DOMAIN_STRUCTS_DIMS_GPU

#include "custom_types.h"

#define ALIGN_UP(addr, align) (((addr) + (align-1) ) & ~(align-1))
#define ALIGN_ADDR_UP(addr, align) ((reinterpret_cast<uintptr_t>(addr) + static_cast<uintptr_t>(align-1) ) & ~static_cast<uintptr_t>(align-1))

#define WITH_GRAD

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

#include "transformations.h"

template <typename scalar_t>
using S4 = Vector<scalar_t, 4>;
/*union S4{
	struct{
		scalar_t x;
		scalar_t y;
		scalar_t z;
		scalar_t w;
	}; // v;
	scalar_t a[4];
};*/

using I4 = S4<int32_t>;
using U4 = S4<dim_t>;
using F4 = S4<float>;
using D4 = S4<double>;

template <typename scalar_t>
inline S4<scalar_t> makeS4(const scalar_t x = 0, const scalar_t y = 0, const scalar_t z = 0, const scalar_t w = 0){
	//return {{.x=x, .y=y, .z=z, .w=w}};
	return {{x, y, z, w}};
}
const auto makeI4 = makeS4<int32_t>;
const auto makeU4 = makeS4<dim_t>;
const auto makeF4 = makeS4<float>;
const auto makeD4 = makeS4<double>;


template <index_t DIMS>
union SpatialShape{
	index_t a[DIMS+1]; //row, col
};

template <>
union SpatialShape<1>{
    struct{
		scalar_t x;
		scalar_t w;
    };
	scalar_t a[2]; //row, col
};

template <>
union SpatialShape<2>{
    struct{
		scalar_t x;
		scalar_t y;
		scalar_t w;
    };
	scalar_t a[3]; //row, col
};

template <>
union SpatialShape<3>{
    struct{
		scalar_t x;
		scalar_t y;
		scalar_t z;
		scalar_t w;
    };
	scalar_t a[4]; //row, col
};

// structs

/*template <typename scalar_t, int SIZE>
struct Matrix{
	scalar_t a[SIZE][SIZE]; //row, col
};*/

template <typename scalar_t, int DIMS>
struct TransformGPU{
	MatrixSquare<scalar_t, DIMS> M;
	MatrixSquare<scalar_t, DIMS> Minv;
	scalar_t det;
};
inline index_t TransformNumValues(const index_t nDims){
	return nDims*nDims*2+1;
}
template <typename scalar_t, int DIMS>
inline index_t TransformNumValues(){
	return sizeof(TransformGPU<scalar_t, DIMS>)/sizeof(scalar_t);
}

template <typename scalar_t>
struct CSRmatrixGPU{
	scalar_t *value;
	index_t *index;
	index_t *row;
};

template <typename scalar_t, int DIMS>
struct StaticDirichletBoundaryGPU{
	scalar_t slip;
	Vector<scalar_t, DIMS> velocity;
	scalar_t scalar;
#ifdef WITH_GRAD
	//scalar_t *slip_grad;
	scalar_t *velocity_grad;
	scalar_t *scalar_grad;
#endif //WITH_GRAD
};

template <typename scalar_t, int DIMS>
struct StaticNeumannBoundaryGPU{
	scalar_t slip;
	Vector<scalar_t, DIMS> boundaryGradient;
	scalar_t scalarGradient;
};

template <typename scalar_t, int DIMS>
struct VaryingDirichletBoundaryGPU{
	scalar_t slip;
	SpatialShape<DIMS> size;
	SpatialShape<DIMS> stride;
	scalar_t *velocity;
	scalar_t *scalar;
#ifdef WITH_GRAD
	//scalar_t *slip_grad;
	scalar_t *velocity_grad;
	scalar_t *scalar_grad;
#endif //WITH_GRAD
};

template <typename scalar_t>
struct ConnectedBoundaryGPU{
	index_t connectedGridIndex;
	// dim_t connectedFace;
	// dim_t connectedAxis1;
	// dim_t connectedAxis2;
	U4 axes;
};

template <typename scalar_t>
struct PeriodicBoundaryGPU{
};

template <typename scalar_t>
struct BoundaryGPU{
	BoundaryType type;
	union{
		StaticDirichletBoundaryGPU<scalar_t> sdb;
		VaryingDirichletBoundaryGPU<scalar_t> vdb;
		StaticNeumannBoundaryGPU<scalar_t> snb;
		ConnectedBoundaryGPU<scalar_t> cb;
		PeriodicBoundaryGPU<scalar_t> pb;
	};
};

template <typename scalar_t, int DIMS>
struct BlockGPU{
	index_t globalOffset; //offset in cells from first block start. used e.g. for CSR indices
	index_t csrOffset;
	SpatialShape<DIMS> size;
	SpatialShape<DIMS> stride;
	scalar_t *velocity;
	scalar_t *pressure;
	scalar_t *scalarData;
	//scalar_t *velocityUpdate;
#ifdef WITH_GRAD
	scalar_t *velocity_grad;
	scalar_t *pressure_grad;
	scalar_t *scalarData_grad;
	//scalar_t *velocityUpdate_grad;
#endif
	BoundaryGPU<scalar_t> boundaries[2*DIMS];
	bool hasTransform;
	scalar_t *transform;
};

template <typename scalar_t, int DIMS>
struct DomainGPU{
	index_t numBlocks;
	index_t numCells; //totalSize, globalSize
	BlockGPU<scalar_t, DIMS> *blocks;
	//scalar_t timeStep;
	const dim_t numDims = DIMS;
	
	scalar_t viscosity;
	
	CSRmatrixGPU<scalar_t> C;
	scalar_t *Adiag;
	CSRmatrixGPU<scalar_t> P;

	scalar_t *scalarRHS;
	scalar_t *scalarResult;
	
	scalar_t *velocityRHS;
	scalar_t *velocityResult;

	scalar_t *pressureRHS;
	scalar_t *pressureRHSdiv;
	scalar_t *pressureResult;
#ifdef WITH_GRAD
	CSRmatrixGPU<scalar_t> C_grad;
	scalar_t *Adiag_grad;
	
	scalar_t *scalarRHS_grad;
	scalar_t *scalarResult_grad;
	
	scalar_t *velocityRHS_grad;
	scalar_t *velocityResult_grad;

	scalar_t *pressureRHS_grad;
	scalar_t *pressureRHSdiv_grad;
	scalar_t *pressureResult_grad;
#endif
};

struct DomainAtlasSet{
	size_t sizeBytes;
	size_t blocksOffsetBytes;
	void* p_host;
	void* p_device;
};

void CopyToGPU(void *p_dst, const void *p_src, const size_t bytes);
void CopyDomainToGPU(const DomainAtlasSet &domainAtlas);


#endif //_INCLUDE_DOMAIN_STRUCTS_DIMS_GPU