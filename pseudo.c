//sort dimension of data set with the most change/fluctuation  
	
/*
sortedD: Dataset with a sorted x axis
eps: search radius distance
MinPts: minimum number of points in eps to define as part of cluster
DIM: number of dimensions per point
*/
DBSCANBaseline(sortedD, eps, MinPts, DIM, int sortedDim) //called with thread per element of dataset (N threads)
{
	//assign thread ID 0,1,2,3....N
	tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	//check if tid < N
	if (tid < N)
	{
	    unsigned int neighborIndex=0;
		float oneDimDistance = 0;
		float fullDistance;
	
		//initialize neighbors array
		unsigned int neighborsArr[ N ];
		
		/////////////////////// OBTAINING NEIGHBORS  /////////////////////////////
		
		//loop up from threadID element + 1 until difference in sorted dimension values > epsilon
		for (int pointIndex=tid+1; oneDimDistance < eps; pointIndex++)
		{
			float currSumOfDiff = 0;
			
			//this line breaks the loop
			//basically if the difference in the sortedDim of 2 points > eps, no more comparisons needed
			oneDimDistance = sortedD[ tid*DIM+sortedDim ] - sortedD[ pointIndex*DIM+sortedDim ];
			
			//loop through dimensions of points 
			for (int dimIndex = 0; dimIndex < DIM; dimIndex++)
			{
			    currSumOfDiff += (sortedD[tid*DIM+dimIndex] - sortedD[beeNum2*DIM+dimIndex]) * 
                                  (sortedD[tid*DIM+dimIndex] - sortedD[beeNum2*DIM+dimIndex]);
			}
			fullDistance = sqrt(currSumOfDiff);
			
			if (fullDistance <= eps)
			{
				neighborArr[neighborIndex] = pointIndex;
				
				neighborIndex++;
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
			    currSumOfDiff += (sortedD[tid*DIM+dimIndex] - sortedD[beeNum2*DIM+dimIndex]) * 
                                  (sortedD[tid*DIM+dimIndex] - sortedD[beeNum2*DIM+dimIndex]);
			}
			fullDistance = sqrt(currSumOfDiff);
			
			if (fullDistance <= eps)
			{
				neighborArr[neighborIndex] = pointIndex;
				
				neighborIndex++;
			}
		}

		//////////////////////// END OBTAINING NEIGHBORS ///////////////////////
		
		// check if size of neighbors array < MinPts 
		if (neighborsIndex < MinPts)
		{
			//mark P as Noise (not part of cluster)
			
		}
		else
		{
			//expand to neighbors   
			
		}
	}            
}

//uses binary search algorithm to find upper and lower bounds of sorted data
DBSCANOptimized