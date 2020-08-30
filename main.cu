#include "cuda_math.h"
#include <stdio.h>
#include "err.h"

const int NQ = 19, Nq = NQ-1; 
const int Nx = 15, Ny = 5, Nz = 5; 
const int N_containers = Nx*Ny*Nz;
const int N_cubes = 8; 
const int Nc = 8; 
const int MaxLevel = 4; 

const int MaxStorage = MaxLevel*N_containers; 

const int TreeSpan = (1<<MaxLevel)*Nc;

const unsigned short null_node = -1;
const int nodes_in_tree = 1 + 8 + 64 + 512 + 4096;
const unsigned int no_container = -1; //has to be >> MaxStorage

struct Cell{
  float fi[NQ];
  __device__ void reset(float f=0) {  for (int iq=0; iq<NQ; iq++) fi[iq] = f; }
  __device__ double sum() { float s; for (int iq=0; iq<NQ; iq++) s += fi[iq]; return s; } 
};

struct Cube {
  Cell cell[Nc*Nc*Nc];
  char aS; //adviseS
};

struct Container {
  Cube cubes[N_cubes];
  int next; 
};


struct Tree{
  // this is the tree: MaxLevel+1 morton cubes 
  // order from S=0 (1 element) to MaxLevel (2**(Maxlevel*D) elements) one by one;
  // children of node [inode] are at (2**D)*inode+1+ range(2**D)
  // children of 0 are 1,2,3,..8; chidren of 1 are 9,10...
  unsigned short nodes[nodes_in_tree];  //each number is less than Nleaves
  unsigned short Nleaves; //less than the number of nodes in the tree
  unsigned int LeafContainer; //less than MaxStorage
  __device__ void init() ;
  __device__ void add_leaf(int inode, Cube cube) ;
};

struct pars {
  Container* leaf_storage;
  int* free_storage;
  int* next_free;
  Tree* trees;
  __device__ Cube & leaf_spot (int leaf_container, int num_leaf) ;
  __device__  void set_free(int i) { // ----------------- not  tested
    if (*next_free>0) {
      int spot = atomicSub(next_free,1);
      free_storage[*next_free] = i;
      } else {
      printf("set_free ERROR\n");
      }
    }    
};

__constant__ pars par;


__device__ void Tree::init () {
  LeafContainer = no_container;
  Nleaves = 0; 
  for (int inode=0; inode<nodes_in_tree; inode++) {
    nodes[inode] = null_node;
  }
};

__device__ void Tree::add_leaf(int inode, Cube cube) {
  nodes[inode] = Nleaves; 
  Cube & Leaf = par.leaf_spot(LeafContainer, nodes[inode]);
  Leaf = cube;
  Nleaves ++; 
};

__device__ Cube & pars::leaf_spot (int leaf_container, int num_leaf) {
   if (num_leaf >= N_cubes) {
     int next_container = leaf_storage[leaf_container].next;
     return leaf_spot (next_container, num_leaf - N_cubes); 
   } else {
     return leaf_storage[leaf_container].cubes[num_leaf];
   }
}


struct pars_host: public pars {
  void reset() {
    printf("Tree span: %6i Nc: %i\n",  TreeSpan, Nc);
    printf("%i x %i x %i  = %i trees\n", Nx, Ny, Nz, Nx*Ny*Nz);
    printf("nulls are: %u short: %u\n",  no_container, null_node);
    size_t sz_store = MaxStorage*sizeof(Container);
    size_t sz_free_store = MaxStorage*sizeof(int);
    size_t sz_trees = Nx*Ny*Nz*sizeof(Tree);
    printf("Size of Storage: %.2fM\n",  sz_store/(1024.*1024.));
    printf("Size of Free-Storage: %.2fM\n",  sz_free_store/(1024.*1024));
    printf("Size of all trees: %.2fM\n", sz_trees/(1024.*1024));
    if(CHECK_ERROR(cudaMalloc((void**) (&leaf_storage), sz_store))) throw(-1);
    if(CHECK_ERROR(cudaMemset(leaf_storage, 0, sz_store))) throw(-1);
    if(CHECK_ERROR(cudaMalloc((void**) (&free_storage), sz_free_store))) throw(-1);
    if(CHECK_ERROR(cudaMemset(free_storage, 0, sz_free_store))) throw(-1);
    if(CHECK_ERROR(cudaMalloc((void**) (&trees), sz_trees))) throw(-1);
    if(CHECK_ERROR(cudaMemset(trees, 0, sz_trees))) throw(-1);
    if(CHECK_ERROR(cudaMalloc((void**) (&next_free), sizeof(int)))) throw(-1);
    if(CHECK_ERROR(cudaMemset(next_free, 0, sizeof(int)))) throw(-1);
  }
  void clear() {
    cudaFree(free_storage); 
    cudaFree(leaf_storage); 
    cudaFree(trees); 
    cudaFree(next_free); 
    PRINT_LAST_ERROR();}
};

pars_host Host;

const dim3 blocks2init  (Nx,Ny,Nz); 
const int threads2init = Nc*Nc*Nc; //512

__global__ void init_free() {
  int i_cntn = blockIdx.x + gridDim.x*(blockIdx.y + gridDim.y*blockIdx.z);
  for (int i=i_cntn+Nx*Ny*Nz; i<MaxStorage; i+=Nx*Ny*Nz){
    par.free_storage[i] = i;
    }
  atomicAdd(par.next_free,1);
}
__global__ void init() {
  int tree_num = blockIdx.x + gridDim.x*(blockIdx.y + gridDim.y*blockIdx.z);
  Tree & tree = par.trees[tree_num];
  tree.init();
  tree.Nleaves = 1; 
  tree.LeafContainer = tree_num ;
  tree.nodes[0] = 0;
  int root_leaf_num = 0;
  par.leaf_spot(tree.LeafContainer,root_leaf_num).cell[threadIdx.x].reset(blockIdx.x*Nc+threadIdx.x%Nc);
}

__device__ unsigned int get_parent(unsigned  int inode = 0) {
  int I = (inode-1)%8; 
  int II = inode - I;
  return (II-1)/8;
}

__device__ uint3 morton_pos(unsigned int _im = 0) {
  uint3 pos3 {0,0,0}; 
  unsigned int inode = _im; 
  unsigned int bitpos = 0;
  for (int s=0; s<MaxLevel; s++) {
    if (inode>0){
      int x = ((inode-1)%8)&1;
      pos3.x = pos3.x|x<<(bitpos);
      bitpos++;
      inode = get_parent(inode);
    } else {
      pos3.x = pos3.x<<1;
    }
    printf("s,inode,x %i %i %i\n",s,inode,pos3.x);
  }
  return pos3; 
}

__global__ void refine() {
  int tree_num = blockIdx.x + gridDim.x*(blockIdx.y + gridDim.y*blockIdx.z);
  Tree & tree = par.trees[tree_num];
  Cube cube; for (int ic=0;ic<Nc*Nc*Nc;ic++) cube.cell[ic].reset(8);
  for (int inode = 0; inode < 1+8; inode++) {
    //if (tree.nodes[inode]!=null_node) tree.add_leaf(cube);
  }

}

__global__ void output(int it) {
  int tree_num = blockIdx.x + gridDim.x*(blockIdx.y + gridDim.y*blockIdx.z);
  uint3 pos3 = blockIdx;
  pos3 += morton_pos(50);
  Tree & tree = par.trees[tree_num];
  int root_leaf_num = 0;
  if (threadIdx.x==0 && tree_num==0) {
    Cell & cell = par.leaf_spot(tree.LeafContainer,root_leaf_num).cell[threadIdx.x];
    printf(" it %i tree %i sum  %f (x-y-z) (%4i %4i %4i) next_free %i \n", it, tree_num, cell.sum(), pos3.x, pos3.y, pos3.z,*par.next_free);
  }
}

__global__ void debug() {
  for (int i=0; i<MaxStorage; i++){
   // printf("%i - %i\n", i, par.free_storage[i]); 
    }
  //printf("next_free %i \n", parnext_free); 
  //int i = nodes_in_tree-1;
  //int i = 584;
  for (int i=0; i<nodes_in_tree; i++) {
    printf("parent of %i  is %i \n", i, get_parent(i)); 
    printf("pos of %i  is %i/%i\n", i, morton_pos(i).x, 1<<MaxLevel); 
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
    init_free<<<blocks2init, 1>>>(); //----------------INIT-------------------
    init<<<blocks2init, threads2init>>>(); //----------------INIT-------------------
    if(CHECK_ERROR(cudaDeviceSynchronize())) throw(-1);
    if(CHECK_ERROR(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared))) throw(-1);
    PRINT_LAST_ERROR();
      for(int i=0; i<1; i++) {
        debug<<<1,1>>>();
        //refine<<<blocks2init, 1>>>();
        //output<<<blocks2init, threads2init>>>(i);
        int Ntorres=step(); 
      }
  } catch(...) {
    PRINT_LAST_ERROR(); printf("Возникла какая-то ошибка.\n");
  }
  Host.clear();
  return 0;
}
