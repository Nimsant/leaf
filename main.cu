#include "cuda_math.h"
#include <stdio.h>
#include "err.h"

const int NQ = 19, Nq = NQ-1; 
const int Nx = 15, Ny = 15, Nz = 15; 
const int N_containers = Nx*Ny*Nz;
const int N_cubes = 8; 
const int Nc = 8; 
const int MaxLevel = 4; 

struct Cell{
  float fi[NQ];
};

struct Group {
  float fi[Nq+1][8];
};

struct Cube {
  Cell cube[Nc*Nc*Nc];
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
 int save_leaf(Cube cube, int leaf_container, int num_leaf){
   if (num_leaf >= N_cubes) {
     int next_container = leaf_storage[leaf_container].next;
     save_leaf(cube, next_container, num_leaf-N_cubes); 
   } else {
     leaf_storage[leaf_container].cubes[num_leaf] = cube;
   }
   return 0;
 }
};

__constant__ pars par;

struct pars_host: public pars {
  void reset() {
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

__global__ void init() {
  int tree_num = blockIdx.x + GridDim.x*(blockIdx.y + GridDim.y*blockIdx.z);
  Tree & tree = par.trees[tree_num];
/*
  tree.Nleaves = 1; 
  tree.LeafContainer = tree_num ;
  int root_leaf_num = 0;
  par.leaf_storage[tree.LeafContainer].cubes[root_leaf_num].cube[ic].set(0);
*/
  //Tile & tile = par.get_tile(blockIdx.x, blockIdx.y, blockIdx.z);
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
    //init<<<blocks2init, threads2init>>>(); //----------------INIT-------------------
    if(CHECK_ERROR(cudaDeviceSynchronize())) throw(-1);
    if(CHECK_ERROR(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared))) throw(-1);
    PRINT_LAST_ERROR();
      for(int i=0; i<10; i++) {
        int Ntorres=step(); 
      }
  } catch(...) {
    PRINT_LAST_ERROR(); printf("Возникла какая-то ошибка.\n");
  }
  Host.clear();
  return 0;
}
