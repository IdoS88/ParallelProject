#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cassert>
#include <malloc.h>
#include <mpi.h>
#include "omp.h"
#include "main.h"
#define FILENAME "input.txt"
int recievingResults(objectFound* searchRecordsArray, int* resultsReceived, MPI_Status* status);
void recievePictureFromProcess(int senderProcessID, Matrix* picture, MPI_Status* status);
int MasterDynamicWay(objectFound* searchRecordsArray, Matrix* arrP, int numOfps, int size, MPI_Status* status);
int SlaveDynamicWay(Matrix* arrO, int numOfobjs, double matchingV, int rank, MPI_Status* status);
enum typeS
{
    matrixType,
    sizeDetailsType
};
int main(int argc, char* argv[])
{
    int size, rank;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2)
    {
        printf("Run the example with at least two processes\n");
        MPI_Abort(MPI_COMM_WORLD, __LINE__);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int numOfps, numOfobjs;
    double matchingV;
    Matrix* arrP;
    Matrix* arrO;
    int strategy;
    double startTime, endTime;
    char send[5] = "send", recv[5] = "recv";
    if (rank == 0) // master
    {
        FILE* fp;
        startTime = MPI_Wtime();
        int flag;
        //objectFound* searchRecordsArray = (objectFound*)malloc(3 * numOfps * sizeof(objectFound));

//----------------------------------reading part-------------------------------------------------/
// the process 0 reads all the content from the file
        readInputFile(&arrP, &numOfps, &arrO, &numOfobjs, &matchingV);
        assert(arrP[0].id == 1);
        //assert(arrO[0].id == 1);
        assert(numOfps == 5);
        assert(numOfobjs == 8);
        printf("end reading\n");
        for (int i = 0;i < numOfps;i++)
        {
            printf("\n\nid %d size %d\n\n", arrP[i].id, arrP[i].size);
            //printMat(arrP[i]);
        }

        //-------------------------end reading------------------------------------------------//
        //------------------------sending and recieving data-----------------------------------//
        //broadcasting with mpi - array of objects
        broadcasting(&matchingV, &numOfobjs, &arrO, rank);

        // final buffer for all the findings
        objectFound* searchRecordsArray = (objectFound*)malloc(numOfps * 3 * sizeof(objectFound));


        if (!MasterDynamicWay(searchRecordsArray, arrP, numOfps, size, &status))
        {
            printf("\n\nmaster failed dynamic\n");
            MPI_Abort(MPI_COMM_WORLD, __LINE__);
        }
        printf("\n\n\nprocess 0 recievied all. done processing %d", __LINE__);

        //------------------------------final phase - writing to output file---------------------------------------//
        write_to_file("output.txt", numOfps, searchRecordsArray);
        printf("success writing output\n");

        freePos(searchRecordsArray, 3 * numOfps);
        free(searchRecordsArray);
        endTime = MPI_Wtime();
        printf("\n\nMaster ended process succefully time %lf\n\n", endTime - startTime);
    }
    else // slave
    {

        //----------------------------------------recieving process----------------------------------------------------------------
          //recieving broadcasting with mpi - array of objects
        broadcasting(&matchingV, &numOfobjs, &arrO, rank);

        printf("slave %d recieved end of package of objects line %d \n", rank, __LINE__);
        // While the master sends pictures to handle
        if (!SlaveDynamicWay(arrO, numOfobjs, matchingV, rank, &status))
        {
            printf("\n\n\nslave process failed\n\n");
            MPI_Abort(MPI_COMM_WORLD, __LINE__);
        }

    }


    //freeing allocated memory
    freeMatrix(arrO, numOfobjs);
    printf("free successfully rank %d \n", rank);
    if (rank == 0)
    {
        freeMatrix(arrP, numOfps);
        printf("free successfully rank %d \n", rank);

    }
    MPI_Finalize();
    return EXIT_SUCCESS;
}