#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define DATASET_SIZE 1864620
#define DIMENTION 2
#define ELIPSON 1.25
#define MIN_POINTS 4

struct Rect {
    double min[2];
    double max[2];
};

int searchNeighbors(const void *node, const void *rect,double **dataset) {
    int id = *(int *)node;
    struct Rect *r = (struct Rect *)rect;
    // Check if the node is within the rectangle
    if (dataset[id][0] >= r->min[0] && dataset[id][0] <= r->max[0] &&
        dataset[id][1] >= r->min[1] && dataset[id][1] <= r->max[1]) {
        return 1; // Return 1 to include the node in the search results
    }
    return 0; // Return 0 to exclude the node from the search results
}


double getDistance(int center, int neighbor, double **dataset) {
    double dist = (dataset[center][0] - dataset[neighbor][0]) *
                      (dataset[center][0] - dataset[neighbor][0]) +
                  (dataset[center][1] - dataset[neighbor][1]) *
                      (dataset[center][1] - dataset[neighbor][1]);

    return dist;
}

void importDataset(char * fname, unsigned int N, unsigned int DIM, float * dataset){
    
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

void findNeighbors(int pos,float *dataset, double elipson, int *neighbors, int *numNeighbors) {
    struct Rect searchRect = {
        {dataset[pos * DIMENTION + index] - elipson, dataset[pos][1] - elipson},
        {dataset[pos][0] + elipson, dataset[pos][1] + elipson}
    };

    *numNeighbors = 0;
    for (int i = 0; i < DATASET_SIZE; i++) {
        if (searchNeighbors(&i, &searchRect,dataset)) {
            double distance = getDistance(pos, i, dataset);
            if (distance <= elipson * elipson) {
                neighbors[(*numNeighbors)++] = i;
            }
        }
    }
}

void dbscan(float *dataset, double elipson, int minPoints, int *clusters) {
    int cluster = 0;

    for (int i = 0; i < DATASET_SIZE; i++) {
        if (clusters[i] != 0) continue;
        int neighbors[DATASET_SIZE];
        int numNeighbors;
        findNeighbors(i, dataset, elipson, neighbors, &numNeighbors);

        if (numNeighbors < minPoints) {
            clusters[i] = -1;
            continue;
        }
        cluster++;
        clusters[i] = cluster;

        for (int j = 0; j < numNeighbors; j++) {
            int dataIndex = neighbors[j];

            if (dataIndex == i) continue;

            if (clusters[dataIndex] == -1) {
                clusters[dataIndex] = cluster;
                continue;
            }
            if (clusters[dataIndex] != 0) continue;

            clusters[dataIndex] = cluster;

            int moreNeighbors[DATASET_SIZE];
            int numMoreNeighbors;
            findNeighbors(dataIndex, dataset, elipson, moreNeighbors, &numMoreNeighbors);

            if (numMoreNeighbors >= minPoints) {
                for (int x = 0; x < numMoreNeighbors; x++) {
                    int neighbor = moreNeighbors[x];
                    int found = 0;
                    for (int y = 0; y < numNeighbors; y++) {
                        if (neighbors[y] == neighbor) {
                            found = 1;
                            break;
                        }
                    }
                    if (!found) {
                        neighbors[numNeighbors++] = neighbor;
                    }
                }
            }
        }
    }
}

void results(int *clusters) {
    int cluster = 0;
    int noises = 0;
    for (int i = 0; i < DATASET_SIZE; i++) {
        if (clusters[i] > cluster) {
            cluster = clusters[i];
        } else if (clusters[i] == -1) {
            noises++;
        }
    }

    printf("Number of clusters: %d\n", cluster);
    printf("Noises: %d\n", noises);
}

int main(int argc, char **argv) {
    float *dataset = (float *)malloc(sizeof(float) * DATASET_SIZE);
       importDataset("sorted_smiley.txt", DATASET_SIZE,DIMENTION, dataset);

    clock_t totalTimeStart, totalTimeStop;
    double totalTime = 0.0;
    totalTimeStart = clock();

    int *clusters = (int *)malloc(sizeof(int) * DATASET_SIZE);
    memset(clusters, 0, sizeof(int) * DATASET_SIZE);

    dbscan(dataset, ELIPSON, MIN_POINTS, clusters);

    totalTimeStop = clock();
    totalTime = (double)(totalTimeStop - totalTimeStart) / CLOCKS_PER_SEC;
    printf("==============================================\n");
    printf("Total Time: %3.2f seconds\n", totalTime);
    printf("==============================================\n");

    results(clusters);

     free(dataset);
    free(clusters);

    return 0;
}

