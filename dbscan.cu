//example of running the program: ./DBSCAN 7490 135000 10000.0 250 bee_dataset_1D_feature_vectors.txt
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <iostream>
#include <complex.h>
#include <math.h>
#include "disjoint_set.h"

//factor (the max percent of dataset that can be neighbors) 
//      No point be neighbors with more than 5% of the dataset
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
#define MODE 2
//Define any constants here
//Feel free to change BLOCKSIZE
#define BLOCKSIZE 128
#define SORTED_DIM 0
#define MAX_NEIGHBORS 100
using namespace std;
//function prototypes
void warmUpGPU();
void computeDistanceMatrixCPU(float* dataset, unsigned int N, unsigned int DIM);
void checkParams(unsigned int N, unsigned int DIM, unsigned int minPts);
void importDataset(char * fname, unsigned int N, unsigned int DIM, float * dataset);
void printDataset(unsigned int N, unsigned int DIM, float * dataset);
void sortDataset(float *dataset);     //for MODE 2 optimization 
void outputResultToFile(int * resultSet, unsigned int N, double runTime);
void expandClusters(unsigned int N, int* neighborFreqs, int* neighborsArr, int* neighborPos, int minPts, int* clusterLabels);
void outputNeighbors( int *neighborsArr, int *neighborPos, unsigned int N );
void outputNeighborPos( int *neighborPos, unsigned int N );
void outputNeighborFreqs( int *neighborFreqs, unsigned int N );

//kernel prototypes
__global__ void getNeighborsSorted(float *sortedD, float eps, unsigned int N, int DIM, int min_pts, int *neighborFreqs, int *neighborsArr, int *neighborPos);
__global__ void partTwo(float *sortedD, float eps, unsigned int N, int DIM, int min_pts, int *neighborFreqs, int *neighborsArr, int *neighborPos);

int main(int argc, char *argv[])
{
  printf("\nMODE: %d", MODE);
  warmUpGPU(); 
  char inputFname[500];
  unsigned int N=0;
  unsigned int DIM=0;
  float epsilon=0;
  int minPts=0;
  if (argc != 6) {
    fprintf(stderr,"Please provide the following on the command line: \nN (number of lines in the file),\ndimensionality (number of coordinates per point), \nepsilon, \nMinPts (The number of points that defines a core point) --larger minpts = more noise = more clusters--\ndataset filename.\n");
    exit(0);
  }
  sscanf(argv[1],"%d",&N);
  sscanf(argv[2],"%d",&DIM);
  sscanf(argv[3],"%f",&epsilon);
  sscanf(argv[4],"%d",&minPts);
  strcpy(inputFname,argv[5]);

  checkParams(N, DIM, minPts);

  printf("\nAllocating the following amount of memory for the dataset: %f GiB", (sizeof(float)*N*DIM)/(1024*1024*1024.0));
  
  //float * dataset=(float*)malloc(sizeof(float*)*N*DIM);
  //importDataset(inputFname, N, DIM, dataset);

  double tstart=omp_get_wtime();

  //create, populate, allocate, and copy sortedData
  float *sortedD=(float*)malloc(sizeof(float*)*N*DIM);
  importDataset(inputFname, N, DIM, sortedD);

  //sortedD = sortDataset(dataset);
  float *dev_sortedD;
  gpuErrchk(cudaMalloc((float**)&dev_sortedD, sizeof(float)*N*DIM));
  gpuErrchk(cudaMemcpy(dev_sortedD, sortedD, sizeof(float)*N*DIM, cudaMemcpyHostToDevice));

  //create & allocate neighbors array
  int *neighborsArr=(int*)malloc(sizeof(int*)*N*MAX_NEIGHBORS);
  int *dev_neighborsArr;
  gpuErrchk(cudaMalloc((int**)&dev_neighborsArr, sizeof(int)*N*MAX_NEIGHBORS));

  //create & allocate neighbor frequency array
  int *neighborFreqs=(int*)malloc(sizeof(int*)*N);
  int *dev_neighborFreqs;
  gpuErrchk(cudaMalloc((int**)&dev_neighborFreqs, sizeof(int)*N));

 //create & allocate neighbor position array
  int *neighborPos=(int*)malloc(sizeof(int*)*N);
  int *dev_neighborPos;
  gpuErrchk(cudaMalloc((int**)&dev_neighborPos, sizeof(int)*N));


  //Result set shows each elements cluster ID
  //EX: [2,5,2,3,2,1]
  // point 1 belongs to cluster 2
  // point 2 belongs to cluster 5  ... etc.
  unsigned int * resultSet = (unsigned int *)calloc(N, sizeof(unsigned int));
  unsigned int * dev_resultSet;
  gpuErrchk(cudaMalloc((unsigned int**)&dev_resultSet, sizeof(unsigned int)*N));
  gpuErrchk(cudaMemcpy(dev_resultSet, resultSet, sizeof(unsigned int)*N, cudaMemcpyHostToDevice));

  int *clusterLabels = (int *)malloc(sizeof(int *)*N);
/*
  //Baseline kernels
  if(MODE==1){
  unsigned int BLOCKDIM = BLOCKSIZE; 
  unsigned int NBLOCKS = ceil(N*1.0/BLOCKDIM);

  }
*/
  //if(MODE==2){
  unsigned int BLOCKDIM = BLOCKSIZE; 
  unsigned int NBLOCKS = ceil(N*1.0/BLOCKDIM);

  //Part 1: get neighbors array
  getNeighborsSorted<<<NBLOCKS, BLOCKDIM>>>( dev_sortedD, epsilon, N, DIM, minPts, dev_neighborFreqs, dev_neighborsArr, dev_neighborPos);
  partTwo<<<NBLOCKS, BLOCKDIM>>>( dev_sortedD, epsilon, N, DIM, minPts, dev_neighborFreqs, dev_neighborsArr, dev_neighborPos);

  //copy neighbors arrays to CPU for expand function
  gpuErrchk(cudaMemcpy(neighborsArr, dev_neighborsArr, sizeof(unsigned int)*N*MAX_NEIGHBORS, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(neighborFreqs, dev_neighborFreqs, sizeof(unsigned int)*N, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(neighborPos, dev_neighborPos, sizeof(unsigned int)*N, cudaMemcpyDeviceToHost));

outputNeighborPos( neighborPos, N );
outputNeighborFreqs( neighborFreqs, N );
outputNeighbors( neighborsArr, neighborPos, N );

  //Part 2: assign clusters
  //create a clusterID array
  expandClusters( N, neighborFreqs, neighborsArr, neighborPos, minPts, clusterLabels);

  //}
  
  double tend=omp_get_wtime(); 
  double runTime = tend - tstart;
  
  outputResultToFile( clusterLabels, N, runTime);
  printf("\nResult written to 'result_set.txt'\n");
 
  //Free memory here
  free(neighborsArr);
  free(neighborFreqs);
  free(neighborPos);
  free(clusterLabels);
  //free(dataset);
  
  if (MODE != 1)
  {
      free(sortedD);
  }
  
  printf("\n\nMain complete\n");
  return 0;
}

//Import dataset as one 1-D array with N*DIM elements
//N can be made smaller for testing purposes
//DIM must be equal to the data dimensionality of the input dataset
//Import dataset as one 1-D array with N*DIM elements
//N can be made smaller for testing purposes
//DIM must be equal to the data dimensionality of the input dataset
void importDataset(char *fname, unsigned int N, unsigned int DIM, float* dataset){
    FILE *fp = fopen(fname, "r");
    if (!fp) {
        fprintf(stderr, "Unable to open file\n");
        fprintf(stderr, "Error: dataset was not imported. Returning.");
        exit(0);
    }

    char line[256];

    // Skip the first row containing headers
    fgets(line, sizeof(line), fp);

    unsigned int idx = 0;
    while (fgets(line, sizeof(line), fp) && idx < N*DIM) {
        char *token = strtok(line, ",");
        double value;

        while (token != NULL && idx < N*DIM) {
            sscanf(token, "%lf", &value);
            dataset[idx++] = value;
            token = strtok(NULL, ",");
        }
    }

    fclose(fp);

    // Print the imported dataset
    for (unsigned int i = 0; i < N*DIM; i += 2) {
        printf("point[%u]: %f, %f\n", i/2, dataset[i], dataset[i+1]);
    }
}

void outputNeighbors( int *neighborsArr, int *neighborPos, unsigned int N )
{
    // Open file for writing
    FILE * fp = fopen( "neighbors.txt", "w" );
	
    fprintf(fp, "pointID -> neighbors\n");

    for (int i=0; i<N; i++)
    {
        fprintf(fp, "\n%d -> ", i);

        for (int j=neighborPos[i]; j<neighborPos[i+1]; j++)
        {
             fprintf(fp, "%d, ", neighborsArr[j]);
        }
    }   

    fclose(fp);

}

void outputNeighborPos( int *neighborPos, unsigned int N )
{
    // Open file for writing
    FILE * fp = fopen( "neighborPos.txt", "w" );
	
    fprintf(fp, "pointID -> pos in neighbor array\n");

    for (int i=0; i<N; i++)
    { 
        fprintf(fp, "%d -> %d\n", i, neighborPos[i]);
    }   

    fclose(fp);

}

void outputNeighborFreqs( int *neighborFreqs, unsigned int N )
{
    // Open file for writing
    FILE * fp = fopen( "neighborFreqs.txt", "w" );
	
    fprintf(fp, "pointID -> # neighbors\n");

    for (int i=0; i<N; i++)
    { 
        fprintf(fp, "%d -> %d\n", i, neighborFreqs[i]);
    }   

    fclose(fp);

}


void printDataset(unsigned int N, unsigned int DIM, float * dataset)
{
    for (int i=0; i<N; i++){
        for (int j=0; j<DIM; j++){
		    if(j!=(DIM-1)){
			    printf("%.0f,", dataset[i*DIM+j]);
			}
			else {
			  printf("%.0f\n", dataset[i*DIM+j]);
			}
		}
		
    }  
}

void checkParams(unsigned int N, unsigned int DIM, unsigned int minPts){
  if(N<=0 || DIM<=0){
    fprintf(stderr, "\n Invalid parameters: Error, N: %u, DIM: %u", N, DIM);
    fprintf(stderr, "\nReturning");
    exit(0); 
  }
  if(minPts >= MAX_NEIGHBORS)
  {
    fprintf(stderr, "\n For more accurate clustering, please input a MinPts value smaller than %d", MAX_NEIGHBORS);
    fprintf(stderr, "\nReturning\n");
    exit(0); 
  }
}

void outputResultToFile(int * resultSet, unsigned int N, double runTime){
    // Open file for writing
    FILE * fp = fopen( "result_set.txt", "w" ); 
	
	fprintf(fp, "\n[MODE: %d, N: %d] Total time: %f", MODE, N, runTime);
    fprintf(fp, "pointID -> clusterID\n\n");

    for (int i=0; i<N; i++)
	{
        fprintf(fp, "%d -> %d\n", i, resultSet[i]);
    }   

    fclose(fp);
}

void warmUpGPU(){
  printf("\nWarming up GPU for time trialing...\n");
  cudaDeviceSynchronize();
  return;
}


void computeDistanceMatrixCPU(float * dataset, unsigned int N, unsigned int DIM)
{
    /* float * distanceMatrix = (float*)malloc(sizeof(float)*N*N); */
	/* double tstart = omp_get_wtime(); */

	/* //Write code here */

	/* double tend = omp_get_wtime(); */

	/* computeSumOfDistances(distanceMatrix, N); */

	/* printf("\nTime to compute distance matrix on the CPU: %f", tend - tstart); */

	/* free(distanceMatrix); */
}



/*
sortedD: Dataset sorted along a certain axis
SORTED_DIM: single int that represents which axis is sorted
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
__global__ void getNeighborsSorted(float *sortedD, float eps, unsigned int N, int DIM, int min_pts, int *neighborFreqs, int *neighborsArr, int *neighborPos) //called with thread per element of dataset (N threads)
{
	//assign thread ID 0,1,2,3....N
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	//only keep N threads 
	if (tid >= N)
	{
		return;
	}
	
	unsigned int numNeighbors=0;
	float oneDimDistance = 0;
	float fullDistance;
	float currSumOfDiff = 0;
	
	/////////////////////// OBTAINING LOCAL NEIGHBORS  /////////////////////////////

	//loop up from threadID element + 1 until difference in sorted dimension values > epsilon
	for (int pointIndex=tid; oneDimDistance < eps && pointIndex < N && numNeighbors < MAX_NEIGHBORS; pointIndex++)            //can optimize by having neighbors >= minpts terminate loop
	{
		float currSumOfDiff = 0;
		
		//this line breaks the loop
		//basically if the difference in the SORTED_DIM of 2 points > eps, no more comparisons needed
		oneDimDistance = sortedD[ tid*DIM+SORTED_DIM ] - sortedD[ pointIndex*DIM+SORTED_DIM ];
		
		//loop through dimensions of points 
		for (int dimIndex = 0; dimIndex < DIM; dimIndex++)
		{
			currSumOfDiff += (sortedD[tid*DIM+dimIndex] - sortedD[pointIndex*DIM+dimIndex]) * 
							  (sortedD[tid*DIM+dimIndex] - sortedD[pointIndex*DIM+dimIndex]);
		}
		fullDistance = sqrt(currSumOfDiff);
		
		if (fullDistance <= eps)
		{
			numNeighbors++;
		}
	}
			
	//loop down from threadID element - 1 until difference in x values > epsilon
	for (int pointIndex=tid-1; oneDimDistance < eps && pointIndex >= 0 && numNeighbors < MAX_NEIGHBORS; pointIndex--)
	{
		//this line breaks the loop
		//basically if the difference in the SORTED_DIM of 2 points > eps, no more comparisons needed
		oneDimDistance = sortedD[ tid*DIM+SORTED_DIM ] - sortedD[ pointIndex*DIM+SORTED_DIM ];
		
		//loop through dimensions of points 
		for (int dimIndex = 0; dimIndex < DIM; dimIndex++)
		{
			currSumOfDiff += (sortedD[tid*DIM+dimIndex] - sortedD[pointIndex*DIM+dimIndex]) * 
							  (sortedD[tid*DIM+dimIndex] - sortedD[pointIndex*DIM+dimIndex]);
		}
		fullDistance = sqrt(currSumOfDiff);
		
		if (fullDistance <= eps)
		{
			numNeighbors++;
		}
	}
	////////////////// POPULATE NEIGHBOR FREQUENCY ARRAY ///////////////////////
	
	//give neighborFreq number of neighbors 
        neighborFreqs[tid] = numNeighbors;
	
}
	
__global__ void partTwo(float *sortedD, float eps, unsigned int N, int DIM, int min_pts, int *neighborFreqs, int *neighborsArr, int *neighborPos)
{
	//assign thread ID 0,1,2,3....N
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	//only keep N threads 
	if (tid >= N)
	{
		return;
	}
	
	unsigned int numNeighbors=0;
	float oneDimDistance = 0;
	float fullDistance;
	unsigned int localNeighbors[ MAX_NEIGHBORS + 1 ];
	unsigned int startIndex = 0;
	float currSumOfDiff = 0;
	
	/////////////////////// OBTAINING LOCAL NEIGHBORS  /////////////////////////////

	//loop up from threadID element + 1 until difference in sorted dimension values > epsilon
	for (int pointIndex=tid; oneDimDistance < eps && pointIndex < N && numNeighbors < MAX_NEIGHBORS; pointIndex++)            //can optimize by having neighbors >= minpts terminate loop
	{
		float currSumOfDiff = 0;
		
		//this line breaks the loop
		//basically if the difference in the SORTED_DIM of 2 points > eps, no more comparisons needed
		oneDimDistance = sortedD[ tid*DIM+SORTED_DIM ] - sortedD[ pointIndex*DIM+SORTED_DIM ];
		
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
	for (int pointIndex=tid-1; oneDimDistance < eps && pointIndex >= 0 && numNeighbors < MAX_NEIGHBORS; pointIndex--)
	{
		//this line breaks the loop
		//basically if the difference in the SORTED_DIM of 2 points > eps, no more comparisons needed
		oneDimDistance = sortedD[ tid*DIM+SORTED_DIM ] - sortedD[ pointIndex*DIM+SORTED_DIM ];
		
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

//////////////////////// POPULATE NEIGHBORS ARRAY  ////////////////////////
	
	//find starting position
	for (int i = 0; i < tid; i++)
	{
		startIndex += neighborFreqs[i];             //confirm this works
	}
	
	neighborPos[ tid ] = startIndex;
	
	//transfer from registers to global 

	int j = 0;
	for (int i = startIndex; i < startIndex + numNeighbors; i++)
	{
	    neighborsArr[ i ] = localNeighbors[j];
		j++;
	}
}

// CPU function for expand clusters using disjoint set
void expandClusters(unsigned int N, int* neighborFreqs, int* neighborsArr, int* neighborPos, int minPts, int* clusterLabels)
{
    // Create a disjoint set data structure
    DisjointSet ds(N);

    // Iterate through each point
    for (int i = 0; i < N; i++) {
        if (neighborFreqs[i] >= minPts) {
            // Point forms a cluster
            int startPos = neighborPos[i];
            int endPos = neighborPos[i+1];

            // Merge the sets containing the point and its neighbors
            for (int j = startPos; j < endPos; j++) {
                int neighbor = neighborsArr[j];
                ds.unionSets(i, neighbor);
            }
        }
    }

    // Assign cluster labels based on the disjoint set
    for (int i = 0; i < N; i++) {
        clusterLabels[i] = ds.findSet(i);
    }
}