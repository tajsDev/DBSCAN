//example of running the program: ./A5_similarity_search_starter 7490 135000 10000.0 bee_dataset_1D_feature_vectors.txt

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <iostream>
#include <complex.h>
#include <math.h>

//Error checking GPU calls
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//Mode 1 is the baseline kernel
#define MODE 1

//Define any constants here
//Feel free to change BLOCKSIZE
#define BLOCKSIZE 128


using namespace std;


//function prototypes
//Some of these are for debugging so I did not remove them from the starter file
void warmUpGPU();
void checkParams(unsigned int N, unsigned int DIM);

void importDataset(char * fname, unsigned int N, unsigned int DIM, float * dataset);
void printDataset(unsigned int N, unsigned int DIM, float * dataset);

void computeDistanceMatrixCPU(float * dataset, unsigned int N, unsigned int DIM);
void computeSumOfDistances(float * distanceMatrix, unsigned int N);

void outputDistanceMatrixToFile(float * distanceMatrix, unsigned int N);


//Part 1: Computing the distance matrix 

//Baseline kernel --- one thread per point/feature vector
__global__ void distanceMatrixBaseline(float * dataset, float * distanceMatrix, const unsigned int N, const unsigned int DIM);

//Other kernels that compute the distance matrix (if applicable):



//Part 2: querying the distance matrix
__global__ void queryDistanceMatrixBaseline(float * distanceMatrix, const unsigned int N, const unsigned int DIM, const float epsilon, unsigned int * resultSet);

//Other kernels that query the distance matrix (if applicable):

int main(int argc, char *argv[])
{
  printf("\nMODE: %d", MODE);
  warmUpGPU(); 



  char inputFname[500];
  unsigned int N=0;
  unsigned int DIM=0;
  float epsilon=0;


  if (argc != 5) {
    fprintf(stderr,"Please provide the following on the command line: N (number of lines in the file), dimensionality (number of coordinates per point), epsilon, dataset filename.\n");
    exit(0);
  }

  sscanf(argv[1],"%d",&N);
  sscanf(argv[2],"%d",&DIM);
  sscanf(argv[3],"%f",&epsilon);
  strcpy(inputFname,argv[4]);

  checkParams(N, DIM);

  printf("\nAllocating the following amount of memory for the dataset: %f GiB", (sizeof(float)*N*DIM)/(1024*1024*1024.0));
  printf("\nAllocating the following amount of memory for the distance matrix: %f GiB", (sizeof(float)*N*N)/(1024*1024*1024.0));
  

  float * dataset=(float*)malloc(sizeof(float*)*N*DIM);
  importDataset(inputFname, N, DIM, dataset);



  //CPU-only mode
  //It only computes the distance matrix but does not query the distance matrix
  if(MODE==0){
    computeDistanceMatrixCPU(dataset, N, DIM);
    printf("\nReturning after computing on the CPU");
    return(0);
  }

  double tstart=omp_get_wtime();

  //Allocate memory for the dataset
  float * dev_dataset;
  gpuErrchk(cudaMalloc((float**)&dev_dataset, sizeof(float)*N*DIM));
  gpuErrchk(cudaMemcpy(dev_dataset, dataset, sizeof(float)*N*DIM, cudaMemcpyHostToDevice));

  //For part 1 that computes the distance matrix
  float * dev_distanceMatrix;
  gpuErrchk(cudaMalloc((float**)&dev_distanceMatrix, sizeof(float)*N*N));
  

  //For part 2 for querying the distance matrix
  unsigned int * resultSet = (unsigned int *)calloc(N, sizeof(unsigned int));
  unsigned int * dev_resultSet;
  gpuErrchk(cudaMalloc((unsigned int**)&dev_resultSet, sizeof(unsigned int)*N));
  gpuErrchk(cudaMemcpy(dev_resultSet, resultSet, sizeof(unsigned int)*N, cudaMemcpyHostToDevice));

  
  //Baseline kernels
  if(MODE==1){
  unsigned int BLOCKDIM = BLOCKSIZE; 
  unsigned int NBLOCKS = ceil(N*1.0/BLOCKDIM);
  //Part 1: Compute distance matrix
  distanceMatrixBaseline<<<NBLOCKS, BLOCKDIM>>>(dev_dataset, dev_distanceMatrix, N, DIM);
  //Part 2: Query distance matrix
  queryDistanceMatrixBaseline<<<NBLOCKS,BLOCKDIM>>>(dev_distanceMatrix, N, DIM, epsilon, dev_resultSet);
  }

  //Note to reader: you can move querying the distance matrix outside of the mode
  //Part 2: Query distance matrix
  //queryDistanceMatrixBaseline<<<NBLOCKS,BLOCKDIM>>>(dev_distanceMatrix, N, DIM, epsilon, dev_resultSet);
  
  //Copy result set from the GPU
  gpuErrchk(cudaMemcpy(resultSet, dev_resultSet, sizeof(unsigned int)*N, cudaMemcpyDeviceToHost));

  //Compute the sum of the result set array
  unsigned int totalWithinEpsilon=0;

  //Write code here
  
  printf("\nTotal number of points within epsilon: %u", totalWithinEpsilon);

  double tend=omp_get_wtime();

  printf("\n[MODE: %d, N: %d] Total time: %f", MODE, N, tend-tstart);

  
  //For outputing the distance matrix for post processing (not needed for assignment --- feel free to remove)
  // float * distanceMatrix = (float*)calloc(N*N, sizeof(float));
  // gpuErrchk(cudaMemcpy(distanceMatrix, dev_distanceMatrix, sizeof(float)*N*N, cudaMemcpyDeviceToHost));
  // outputDistanceMatrixToFile(distanceMatrix, N);
 

  //Free memory here


  printf("\n\n");
  return 0;
}
/prints the dataset that is stored in one 1-D array
void printDataset(unsigned int N, unsigned int DIM, float * dataset)
{
    for (int i=0; i<N; i++){
      for (int j=0; j<DIM; j++){
        if(j!=(DIM-1)){
          printf("%.0f,", dataset[i*DIM+j]);
        }
        else{
          printf("%.0f\n", dataset[i*DIM+j]);
        }
      }
      
    }  
}




//Import dataset as one 1-D array with N*DIM elements
//N can be made smaller for testing purposes
//DIM must be equal to the data dimensionality of the input dataset
void importDataset(char * fname, unsigned int N, unsigned int DIM, float * dataset)
{
    
    FILE *fp = fopen(fname, "r");

    if (!fp) {
        fprintf(stderr, "Unable to open file\n");
        fprintf(stderr, "Error: dataset was not imported. Returning.");
        exit(0);
    }

    unsigned int bufferSize = DIM*10; 

    char buf[bufferSize];
    unsigned int rowCnt = 0;
    unsigned int colCnt = 0;
    while (fgets(buf, bufferSize, fp) && rowCnt<N) {
        colCnt = 0;

        char *field = strtok(buf, ",");
        double tmp;
        sscanf(field,"%lf",&tmp);
        
        dataset[rowCnt*DIM+colCnt]=tmp;

        
        while (field) {
          colCnt++;
          field = strtok(NULL, ",");
          
          if (field!=NULL)
          {
          double tmp;
          sscanf(field,"%lf",&tmp);
          dataset[rowCnt*DIM+colCnt]=tmp;
          }   

        }
        rowCnt++;
    }

    fclose(fp);

}



void warmUpGPU(){
printf("\nWarming up GPU for time trialing...\n");
cudaDeviceSynchronize();
return;
}


void checkParams(unsigned int N, unsigned int DIM)
{
  if(N<=0 || DIM<=0){
    fprintf(stderr, "\n Invalid parameters: Error, N: %u, DIM: %u", N, DIM);
    fprintf(stderr, "\nReturning");
    exit(0); 
  }
}


void computeDistanceMatrixCPU(float * dataset, unsigned int N, unsigned int DIM)
{
  float * distanceMatrix = (float*)malloc(sizeof(float)*N*N);
  double tstart = omp_get_wtime();

  //Write code here

  double tend = omp_get_wtime();

  computeSumOfDistances(distanceMatrix, N);

  printf("\nTime to compute distance matrix on the CPU: %f", tend - tstart);

  free(distanceMatrix);
}

//For testing/debugging
void computeSumOfDistances(float * distanceMatrix, unsigned int N)
{
  double computeSumOfDistances=0;
  for (unsigned int i=0; i<N; i++)
  {
    for (unsigned int j=0; j<N; j++)
    {
      computeSumOfDistances+=(double)distanceMatrix[i*N+j];
    }
  }  

  printf("\nSum of distances: %f", computeSumOfDistances);
}
//This is used to do post-processing in Python of bee statistics
//I left it in the starter file in case anyone else wants to tinker with the 
//distance matrix and the bees, but it is unnecessary for the assignment
void outputDistanceMatrixToFile(float * distanceMatrix, unsigned int N)
{

// Open file for writing
FILE * fp = fopen( "distance_matrix_output.txt", "w" ); 

 for (int i=0; i<N; i++){
    for (int j=0; j<N; j++){
      if(j!=(N-1)){
        fprintf(fp, "%.3f,", distanceMatrix[i*N+j]);
      }
      else{
        fprintf(fp, "%.3f\n", distanceMatrix[i*N+j]);
      }
    }
    
  }   

  fclose(fp);
}




//Query distance matrix with one thread per feature vector
__global__ void queryDistanceMatrixBaseline(float * distanceMatrix, const unsigned int N, const unsigned int DIM, const float epsilon, unsigned int * resultSet)
{
  //write code here
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x ;

  if(tid < N ) {

  }
}


//One thread per feature vector -- baseline kernel
__global__ void distanceMatrixBaseline(float * dataset, float * distanceMatrix, const unsigned int N, const unsigned int DIM)
{
  //write code here
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x ;
  if(tid < N ) {


  }
}

