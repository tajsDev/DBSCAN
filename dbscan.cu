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

/*
sortedD: Dataset sorted along a certain axis
sortedDim: single int that represents which axis is sorted
   - This should be the axis with the biggest range in values
eps: search radius distance
MinPts: minimum number of points in eps to define as part of cluster
DIM: number of dimensions per point

neighborFreqs: a global array which shows how many neighbors each point has 
(size N) 
EX - [2,1,3,4,4...] 
      point 1 has 2 neighbors
	  point 2 has 1 neighbors 
	  point 3 has 3 neighbors ....
	  
neighborsArr: a global array that lists all neighbors, must be used in accordance with neighborFreq
(size N*N*F) where F is ~0.05 which is a percent of how many neighbors a point should have 
EX - [34,26,200,439,23,34590,3459,239,49]
when looking at neighborFreq we can say
	  point 1 has neighbors 34 & 26
	  point 2 has neighbor 200
	  point 3 has neighbors 439, 23, & 34590 ....
	  
neighborPos: a global array which shows the start position of neighborsArr for each point
             This array is populated in getNeighbors, but utilized in expand
(size N)
EX - [0,2,3,6,10]
	  point 1 starts at index 0 of neighborsArr
	  point 2 starts at Index 2 of neighborsArr
	  point 3 starts at index 3 of neighborsArr ....
*/
getNeighbors(sortedD, eps, DIM, int min_pts, int sortedDim, int *neighborFreqs, int *neighborsArr, int *neighborPos) //called with thread per element of dataset (N threads)
{
	//assign thread ID 0,1,2,3....N
	tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	//only keep N threads 
	if (tid >= N)
	{
		return;
	}
	
	unsigned int numNeighbors=0;
	float oneDimDistance = 0;
	float fullDistance;
	unsigned int localNeighbors[N*F];
	unsigned int startIndex = 0;
	
	/////////////////////// OBTAINING LOCAL NEIGHBORS  /////////////////////////////
	
	//loop up from threadID element + 1 until difference in sorted dimension values > epsilon
	for (int pointIndex=tid+1; oneDimDistance < eps && pointIndex < N; pointIndex++)            //can optimize by having neighbors >= minpts terminate loop
	{
		float currSumOfDiff = 0;
		
		//this line breaks the loop
		//basically if the difference in the sortedDim of 2 points > eps, no more comparisons needed
		oneDimDistance = sortedD[ tid*DIM+sortedDim ] - sortedD[ pointIndex*DIM+sortedDim ];
		
		//loop through dimensions of points 
		for (int dimIndex = 0; dimIndex < DIM; dimIndex++)
		{
			currSumOfDiff += (sortedD[tid*DIM+dimIndex] - sortedD[pointIndex*DIM+dimIndex]) * 
							  (sortedD[tid*DIM+dimIndex] - sortedD[pointIndex*DIM+dimIndex]);
		}
		fullDistance = sqrt(currSumOfDiff);
		
		if (fullDistance <= eps)
		{
			//neighborArr[tid+neighborIndex] = pointIndex;
			localNeighbors[numNeighbors] = pointIndex;
			numNeighbors++;
		}
	}
			
	//loop down from threadID element - 1 until difference in x values > epsilon
	for (int pointIndex=tid-1; oneDimDistance < eps; pointIndex--)
	{
		//this line breaks the loop
		//basically if the difference in the sortedDim of 2 points > eps, no more comparisons needed
		oneDimDistance = sortedD[ tid*DIM+sortedDim ] - sortedD[ pointIndex*DIM+sortedDim ];
		
		//loop through dimensions of points 
		for (int dimIndex = 0; dimIndex < DIM; dimIndex++)
		{
			currSumOfDiff += (sortedD[tid*DIM+dimIndex] - sortedD[pointIndex*DIM+dimIndex]) * 
							  (sortedD[tid*DIM+dimIndex] - sortedD[pointIndex*DIM+dimIndex]);
		}
		fullDistance = sqrt(currSumOfDiff);
		
		if (fullDistance <= eps)
		{
			//neighborArr[tid+neighborIndex] = pointIndex;
			localNeighbors[numNeighbors] = pointIndex;
			numNeighbors++;
		}
	}

	////////////////// POPULATE NEIGHBOR FREQUENCY ARRAY ///////////////////////
	
	//give neighborFreq number of neighbors 
    neighborFreq[tid] = numNeighbors;
	
	__syncthreads();
	
	//////////////////////// POPULATE NEIGHBORS ARRAY  ////////////////////////
	
	//find starting position
	for (int i = 0; i < tid; i++)
	{
		startIndex += neighborFreq[i];             //confirm this works
	}
	
	neighborPos[ tid ] = startIndex;
	
	//transfer from registers to global 
	for (int i = startIndex, int j=0; i < startIndex + numNeighbors; i++, j++)
	{
	    neighborsArr[ i ] = localNeighbors[j];
	}
	
}
