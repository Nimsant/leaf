#include "cuda_math.h"
#include <stdio.h>
#include "err.h"

const int NQ = 19, Nq = NQ-1; 
const int Nx = 15, Ny = 5, Nz = 5; 
const int N_containers = Nx*Ny*Nz;
const int N_cubes = 8; 
const int Nc = 8; 
const int MaxLevel = 4; 

const int TreeSpan = (1<<4)*Nc;

struct Cell{
  float fi[NQ];
  __device__ void reset(float f=0) {  for (int iq=0; iq<NQ; iq++) fi[iq] = f; }
  __device__ double sum() { float s; for (int iq=0; iq<NQ; iq++) s += fi[iq]; return s; } 
};

struct Group {
  float fi[NQ][8];
};

struct Cube {
  Cell cell[Nc*Nc*Nc];
};

struct Container {
  Cube cubes[N_cubes];
  int next; 
};

struct Tree{
  unsigned short int nodes[4681]; 
  int Nleaves; 
  int LeafContainer;
};

struct pars {
 Container* leaf_storage;
 Tree* trees;
 __device__ Cube * leaf_spot (int leaf_container, int num_leaf) {
   Cube * cube_pointer;
   if (num_leaf >= N_cubes) {
     int next_container = leaf_storage[leaf_container].next;
     cube_pointer = leaf_spot (next_container, num_leaf - N_cubes); 
   } else {
     Cube * ptr = leaf_storage[leaf_container].cubes;
     cube_pointer = ptr + num_leaf;
   }
   return cube_pointer;
 }
};

__constant__ pars par;

struct pars_host: public pars {
  void reset() {
    printf("Tree span: %6i Nc: %i\n",  TreeSpan, Nc);
    size_t sz_store = MaxLevel*N_containers*sizeof(Container);
    size_t sz_trees = Nx*Ny*Nz*sizeof(Tree);
    printf("Size of Storage: %.2fG\n",  sz_store/(1024.*1024.*1024));
    printf("Size of all trees: %.2fG\n", sz_trees/(1024.*1024.*1024));
    if(CHECK_ERROR(cudaMalloc((void**) (&leaf_storage), sz_store))) throw(-1);
    if(CHECK_ERROR(cudaMemset(leaf_storage, 0, sz_store))) throw(-1);
    if(CHECK_ERROR(cudaMalloc((void**) (&trees), sz_trees))) throw(-1);
    if(CHECK_ERROR(cudaMemset(trees, 0, sz_trees))) throw(-1);
  }
  void clear() {cudaFree(leaf_storage); cudaFree(trees); PRINT_LAST_ERROR();}
};

pars_host Host;

const dim3 blocks2init  (Nx,Ny,Nz); 
const int threads2init = Nc*Nc*Nc; //512

__global__ void init() {
  int tree_num = blockIdx.x + gridDim.x*(blockIdx.y + gridDim.y*blockIdx.z);
  Tree & tree = par.trees[tree_num];
  tree.Nleaves = 1; 
  tree.LeafContainer = tree_num ;
  int root_leaf_num = 0;
  par.leaf_spot(tree.LeafContainer,root_leaf_num)->cell[threadIdx.x].reset(blockIdx.x*Nc+threadIdx.x%Nc);
}

__device__ uint3 morton_pos(uint _im = 0) {
  uint3 pos3 {0,0,0}; 
  if (_im<1) {}
  else if (_im<1+8) {
    uint im = _im-1;
    uint3 dpos {im &1, (im &2)>>1, (im &4)>>2}; // (_ & (2**s)) >> s gives s'th bit
    pos3 += dpos*(TreeSpan/2);
  }
  else if (_im<1+8+64) {
    uint im = _im-1-8;
    uint3 dpos {im &1, (im &2)>>1, (im &4)>>2}; // (_ & (2**s)) >> s gives s'th bit
    pos3 += dpos*(TreeSpan/4);
    dpos = make_uint3( (im &8)>>3, (im &16)>>4, (im &32)>>5); // (_ & (2**s)) >> s gives s'th bit
    pos3 += dpos*(TreeSpan/2);
  }
  return pos3; 
}

__global__ void output(int it) {
  int tree_num = blockIdx.x + gridDim.x*(blockIdx.y + gridDim.y*blockIdx.z);
  uint3 pos3 = blockIdx;
  pos3 += morton_pos(50);
  Tree & tree = par.trees[tree_num];
  int root_leaf_num = 0;
  if (threadIdx.x==0 && tree_num==0) {
    Cell & cell = par.leaf_spot(tree.LeafContainer,root_leaf_num)->cell[threadIdx.x];
    printf(" it %i tree %i sum  %f (x-y-z) (%4i %4i %4i) \n", it, tree_num, cell.sum(), pos3.x, pos3.y, pos3.z);
  }
}

int step() {
  return 0;
}

int main(int argc, char** argv) {
  try
  {
    Host.reset();
    if(CHECK_ERROR(cudaMemcpyToSymbol(par, &Host, sizeof(par)))) throw(-1);
    if(CHECK_ERROR(cudaDeviceSynchronize())) throw(-1);
    init<<<blocks2init, threads2init>>>(); //----------------INIT-------------------
    if(CHECK_ERROR(cudaDeviceSynchronize())) throw(-1);
    if(CHECK_ERROR(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared))) throw(-1);
    PRINT_LAST_ERROR();
      for(int i=0; i<10; i++) {
        output<<<blocks2init, threads2init>>>(i);
        int Ntorres=step(); 
      }
  } catch(...) {
    PRINT_LAST_ERROR(); printf("Возникла какая-то ошибка.\n");
  }
  Host.clear();
  return 0;
}
