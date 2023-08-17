#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cassert>
#include <malloc.h>
#include "mpi.h"
#include "omp.h"
#include "main.h"

void broadcasting(double* matchingV, int* numOfobjs, Matrix** arrO, int rank)
{
    char send[5] = "send", recv[5] = "recv";
    if (rank == 0)
    {
        validateBcast(MPI_Bcast(matchingV, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, send);
        validateBcast(MPI_Bcast(numOfobjs, 1, MPI_INT, 0, MPI_COMM_WORLD), __LINE__, send);
        for (int i = 0; i < *numOfobjs; i++)
        {
            validateBcast(MPI_Bcast(&((*arrO)[i].id), 1, MPI_INT, 0, MPI_COMM_WORLD), __LINE__, send);
            validateBcast(MPI_Bcast(&((*arrO)[i].size), 1, MPI_INT, 0, MPI_COMM_WORLD), __LINE__, send);
            validateBcast(MPI_Bcast((*arrO)[i].data, (*arrO)[i].size * (*arrO)[i].size, MPI_INT, 0, MPI_COMM_WORLD), __LINE__, send);
        }
    }
    else {
        validateBcast(MPI_Bcast(matchingV, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, recv);

        validateBcast(MPI_Bcast(numOfobjs, 1, MPI_INT, 0, MPI_COMM_WORLD), __LINE__, recv);
        (*arrO) = (Matrix*)malloc(*numOfobjs * sizeof(Matrix));
        if ((*arrO) == NULL)
        {
            printf("\nCannot allocate memory for slave %d Objects array\n", rank);
            MPI_Abort(MPI_COMM_WORLD, __LINE__);
            exit(EXIT_FAILURE);
        }

        // alocate memory of objects and store it for processing
        for (int i = 0; i < *numOfobjs; i++)
        {
            // Receives thetPicture's integers count from the sender process
            validateBcast(MPI_Bcast(&((*arrO)[i].id), 1, MPI_INT, 0, MPI_COMM_WORLD), __LINE__, recv);
            validateBcast(MPI_Bcast(&((*arrO)[i].size), 1, MPI_INT, 0, MPI_COMM_WORLD), __LINE__, recv);
            (*arrO)[i].data = (int*)malloc((*arrO)[i].size * (*arrO)[i].size * sizeof(int));
            if ((*arrO)[i].data == NULL)
            {
                printf("Error: failed to allocate memory for array line %d\n", __LINE__);
                MPI_Abort(MPI_COMM_WORLD, __LINE__);
                exit(EXIT_FAILURE);
            }
            validateBcast(MPI_Bcast((*arrO)[i].data, (*arrO)[i].size * (*arrO)[i].size, MPI_INT, 0, MPI_COMM_WORLD), __LINE__, recv);
        }
    }
}
int readMatrixes(Matrix* matrices, int numSize, FILE* fp)
{
    printf("---------------------\n");
    printf("Starting to Read Pictures -> Number of pictures %d:\n", numSize);
    for (int i = 0; i < numSize; i++)
    {
        // get picture i ID
        if (fscanf(fp, "%d", &matrices[i].id) != 1)
        {
            printf("Failed to read Picture ID\n");
            return 0;
        }
        // get picture i Dim
        if (fscanf(fp, "%d", &matrices[i].size) != 1)
        {
            printf("Failed to read Picture Dim\n");
            return 0;
        }

        int dim = matrices[i].size * matrices[i].size;

        // allocate memory for the mat of picture i
        matrices[i].data = (int*)malloc(dim * sizeof(int));
        if (!matrices[i].data)
        {
            printf("Failed to allocate memory for matrix\n");
            MPI_Abort(MPI_COMM_WORLD, __LINE__);
            return 0;
        }

        // start reading picture items
        printf("Reading matrix %d\n", matrices[i].id);
        for (int j = 0; j < dim; j++)
        {
            if (fscanf(fp, "%d", &matrices[i].data[j]) != 1)
            {
                printf("Failed to read Matrix item at index %d\n", j);
                MPI_Abort(MPI_COMM_WORLD, __LINE__);
                return 0;
            }
        }
        printf("Success in reading Matrix %d with Dim %d\n", matrices[i].id, matrices[i].size);
    }
    return 1;
}

int readInputFile(Matrix** pictures, int* numPictures, Matrix** objects, int* numObjects, double* matching_value)
{
    FILE* fp = fopen(FILENAME, "r");
    if (!fp)
    {
        printf("Could not open the file\n");
        return 0;
    }
    printf("---------------------\n");
    printf("Starting to Read Input File\n");

    //read the matching value first should be 0.100000 ~ 0.1
    if (fscanf(fp, "%lf", matching_value) != 1)
    {
        printf("Failed to read matching value\n");
        MPI_Abort(MPI_COMM_WORLD, __LINE__);
        return 0;
    }
    printf("Matching value: %lf\n", *matching_value);

    // read number of pictures should be 5
    if (fscanf(fp, "%d", numPictures) != 1)
    {
        printf("Failed to read number of pictures\n");
        MPI_Abort(MPI_COMM_WORLD, __LINE__);
        return 0;
    }
    // allocate memory for pictures
    *pictures = (Matrix*)malloc(*numPictures * sizeof(Matrix));
    if (!*pictures)
    {
        printf("Failed to allocate memory for pictures\n");
        MPI_Abort(MPI_COMM_WORLD, __LINE__);
        return 0;
    }
    // reading from the file the pictures: we read the ID and the dim of each picture
    if (!readMatrixes(*pictures, *numPictures, fp))
    {
        printf("Failed reading pictures from file\n");
        free(*pictures);
        fclose(fp);
        MPI_Abort(MPI_COMM_WORLD, __LINE__);
        return 0;
    }

    //read number of objects should be 8 objects 
    if (fscanf(fp, "%d", numObjects) != 1)
    {
        printf("Failed to read number of objects\n");
        free(*pictures);
        fclose(fp);
        MPI_Abort(MPI_COMM_WORLD, __LINE__);
        return 0;
    }
    // allocate memory for objects
    *objects = (Matrix*)malloc(*numObjects * sizeof(Matrix));
    if (!*objects)
    {
        printf("Failed to allocate memory for objects\n");
        free(*pictures);
        fclose(fp);
        MPI_Abort(MPI_COMM_WORLD, __LINE__);
        return 0;
    }
    //reading from the file the objects: we read the ID and the dim of each object
    if (!readMatrixes(*objects, *numObjects, fp))
    {
        printf("Failed to read objects from file\n");
        free(*pictures);
        free(*objects);
        fclose(fp);
        MPI_Abort(MPI_COMM_WORLD, __LINE__);
        return 0;
    }
    printf("---------------------\n");
    fclose(fp);
    return 1;
}
Matrix read_mat(FILE* fp, int id)
{

    Matrix mat;
    mat.id == NOTFOUND;
    mat.size = NOTFOUND;
    mat.data = NULL;
    if (fp == NULL)
    {
        printf("Error: failed to open file \n");
        exit(EXIT_FAILURE);
    }
    // Read the number of rows and columns from the first line
    fscanf(fp, "%d %d", &mat.id, &mat.size);
    if (mat.id != id)
    {
        printf("read wrong picture of %d instead of %d line %d", mat.id, id, __LINE__);
        mat.id = NOTFOUND;
        exit(EXIT_FAILURE);
    }
    else if (mat.size > 700 || mat.size < 1)
    {
        printf("read wrong picture size of %d line %d", mat.size, __LINE__);
        mat.id = NOTFOUND;
        exit(EXIT_FAILURE);
    }
    int size = mat.size * mat.size;
    // Allocate memory for the array
    mat.data = (int*)malloc(size * sizeof(int));
    if (mat.data == NULL)
    {
        mat.id = NOTFOUND;
        printf("Error: failed to allocate memory for array\n");
        exit(EXIT_FAILURE);
    }
    // Read the array elements from the file
    for (int i = 0; i < size; i++)
    {
        fscanf(fp, "%d", &mat.data[i]);
        printf("\n\n\nmat i %d %ld\n\n\n", i, (sizeof(mat.data[i]) / sizeof(int)));
        assert(malloc_usable_size((void*)&(mat.data[i])) > 0 && (sizeof(mat.data[i]) == (sizeof(int))));
    }
    printf("\n\n\nmat dimension %d mat.data size %ld\n\n\n", mat.size, (malloc_usable_size((void*)mat.data) / sizeof(int)));
    assert(malloc_usable_size((void*)mat.data) > 0 && sizeof(mat.data) == mat.size * mat.size * (sizeof(int)));
    return mat; // Return pointer to array
}
// Matrix* readMatrices(FILE* fp, int size)
// {

//     // if (size == NULL)
//     // {
//     //     printf("Error: failed to read size \n");
//     //     exit(1);
//     // }
//     // Create an array of matrices
//     Matrix* matrix_array = (Matrix*)malloc(size * sizeof(Matrix));
//     if (matrix_array == NULL)
//     {
//         printf("Error: failed to allocate memory for array line %d\n", __LINE__);
//         exit(EXIT_FAILURE);
//     }
//     int i, j, k = 0;

//     // Initialize each matrix
//     for (i = 0; i < size; i++)
//     {
//         Matrix temp = read_mat(fp, (i + 1));
//         if (temp.id == NOTFOUND)
//         {
//             printf("Error: failed to read memory of matrix for array line %d\n", __LINE__);
//             freeMatrix(matrix_array, i);
//             exit(EXIT_FAILURE);
//         }
//         // Initialize the matrix data
//         matrix_array[i] = temp;
//     }
//     return matrix_array;
// }
// void readingMatrices(FILE* fp, int* size, int* numOfps, int* numOfobjs, double* matchingV, Matrix* arrP, Matrix* *) {
//     fp = fopen(FILENAME, "r"); // Open file for reading
//     if (fp == NULL)
//     {
//         printf("Error: failed to open file '%s'\n", FILENAME);
//         MPI_Abort(MPI_COMM_WORLD, __LINE__);
//     }
//     fscanf(fp, "%lf %d", matchingV, numOfps);
//     assert(*numOfps == 5);
//     // Prevents idle processes
//     if (size > numOfps)
//     {
//         printf("Run the program with up to %d (# pictures) processes\n",
//             *
//             numOfps);
//         MPI_Abort(MPI_COMM_WORLD, 0);
//         exit(0);
//     }

//     //arrP = (Matrix**)malloc(sizeof(Matrix*));
//     arrP = readMatrices(fp, *numOfps); // sequential

//     // first i thought about reading a picture and then seek jump to atPicture to proceed quickly.
//     // i checked on the internet.
//     // and found out it's a bad idea to play with the hard drive reading process as it will actually slow down process
//     // and decided to keep it squential
//     if (arrP == NULL && arrP[0].data == NULL)
//     {
//         printf("Error: failed to read Matrix array line %d\n", __LINE__);
//         MPI_Abort(MPI_COMM_WORLD, __LINE__);
//         fclose(fp);
//         exit(EXIT_FAILURE);
//     }
//     for (int i = 0;i < *numOfps;i++)
//     {
//         printf("\n\n\npicture dimension %d p.data size %ld\n\n\n", arrP[i].size, (malloc_usable_size((void*)arrP[i].data) / sizeof(int)));
//         assert(malloc_usable_size((void*)arrP[i].data) > 0 && malloc_usable_size((void*)arrP[i].data) == (arrP[i].size * arrP[i].size * sizeof(int)));
//     }
//     fscanf(fp, "%d", numOfobjs);
//     assert(*numOfobjs == 8);
//     //* = (Matrix**)malloc(sizeof(Matrix*));
//     * = readMatrices(fp, *numOfobjs); // pipeline
//     if (* == NULL && *[0].data == NULL)
//     {
//         printf("Error: failed to read Matrix array line %d\n", __LINE__);
//         MPI_Abort(MPI_COMM_WORLD, __LINE__);
//         fclose(fp);
//         exit(EXIT_FAILURE);
//     }
//     for (int i = 0;i < *numOfobjs;i++)
//     {
//         printf("\n\n\nobject dimension %d object data size %ld\n\n\n", *[i].size, (malloc_usable_size((void*)*[i].data) / sizeof(int)));
//         assert(malloc_usable_size((void*)*[i].data) > 0 && malloc_usable_size((void*)*[i].data) == (*[i].size * *[i].size * sizeof(int)));
//     }
//     fclose(fp); // Close file
// }

int recievingResults(objectFound* searchRecordsArray, int* resultsReceived, MPI_Status* status) {        // Receives a search record from any available process
    // Receive function
    for (int i = *resultsReceived; i < 3 + *resultsReceived; i++) {
        int buffer[4];
        if (!validateRecv(MPI_Recv(buffer, 4, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, status), __LINE__))
        {
            printf("Error: failed recieving recv_data %d\n", __LINE__);
            exit(0);
        }
        searchRecordsArray[i] = deserialize_objectFound(buffer);
        printf("\nsearchRecordsArray[%d].PID %d searchRecordsArray[%d].id %d\n", i, (searchRecordsArray)[i].PID, i, (searchRecordsArray)[i].id);
        assert((searchRecordsArray)[i].PID > 0 && (searchRecordsArray)[i].PID < 6);
        assert((searchRecordsArray)[i].id >= -1 && (searchRecordsArray)[i].id < 9);
        assert(searchRecordsArray[i].pos[0] >= -1 && searchRecordsArray[i].pos[0] <= 800);
        assert(searchRecordsArray[i].pos[1] >= -1 && searchRecordsArray[i].pos[1] <= 800);
        printf("Master recieved search record for picture %d object %d pos %d %d from slave %d line %d\n", (searchRecordsArray)[i].PID, (searchRecordsArray)[i].id, (searchRecordsArray)[i].pos[0], (searchRecordsArray)[i].pos[1], (*status).MPI_SOURCE, __LINE__);

    }



    // for (int i = *resultsReceived, index = 0; i < 3 + *resultsReceived && index < 3; i++, index++) {
    //     printf("\n\n i %d line %d\n\n\n", i, __LINE__);
    //     (searchRecordsArray)[i].PID = recv_data[4 * index];
    //     (searchRecordsArray)[i].id = recv_data[4 * index + 1];
    //     printf("\nsearchRecordsArray[%d].PID %d searchRecordsArray[%d].id %d\n", i, (searchRecordsArray)[i].PID, i, (searchRecordsArray)[i].id);
    //     assert((searchRecordsArray)[i].PID > 0 && (searchRecordsArray)[i].PID < 6);
    //     assert((searchRecordsArray)[i].id >= -1 && (searchRecordsArray)[i].id < 9);
    //     (searchRecordsArray)[i].pos = (int*)malloc(2 * sizeof(int));
    //     if ((searchRecordsArray)[i].pos == NULL)
    //     {
    //         printf("Error: failed to allocate pos array\n");
    //         exit(0);
    //     }
    //     assert(recv_data[4 * index + 2] >= -1 && recv_data[4 * index + 2] <= 800);
    //     assert(recv_data[4 * index + 3] >= -1 && recv_data[4 * index + 3] <= 800);
    //     (searchRecordsArray)[i].pos[0] = recv_data[4 * index + 2];
    //     (searchRecordsArray)[i].pos[1] = recv_data[4 * index + 3];
    //     assert(searchRecordsArray[i].pos[0] >= -1 && searchRecordsArray[i].pos[0] <= 800);
    //     assert(searchRecordsArray[i].pos[1] >= -1 && searchRecordsArray[i].pos[1] <= 800);
    //     printf("Master recieved search record for picture %d object %d pos %d %d from slave %d line %d\n", (searchRecordsArray)[i].PID, (searchRecordsArray)[i].id, (searchRecordsArray)[i].pos[0], (searchRecordsArray)[i].pos[1], (*status).MPI_SOURCE, __LINE__);
    // }
    * resultsReceived += 3;
    assert(*resultsReceived > -1 && *resultsReceived <= 15);
    return 1;
}
/* ------------------------unnecessary code----------------------------------
int generalStrategy(Matrix* arrP, Matrix* *, int sizeP, int sizeO, int numOfProcesses)
{
    Strategy dyn = DYNAMIC;

    Strategy stat = STATIC;
    int res = (int)stat;
#pragma omp parallel for if (res == (int)stat) schedule(dynamic)
    for (int i = 0; i < sizeP; i++)
    {
        if (arrP[i].size % numOfProcesses > 0)
#pragma omp critical
            res = (int)dyn;
        for (int j = 0; j < sizeO; j++)
            if ((arrP[i].size / numOfProcesses) < *[j].size)
#pragma omp critical
                res = (int)dyn;
    }
    return res;
}*/
int sendPictureToProcess(Matrix p, int receiverProcessID)
{
    int err;
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    //("%d / %d\n", world_rank, world_size);
    err = validateSend(MPI_Send(&p.id, 1, MPI_INT, receiverProcessID, WORK, MPI_COMM_WORLD), __LINE__);
    assert(p.id > 0 && p.id < 6);
    if (!err)
        return 0;

    // printf("%d / %d\n", world_rank, world_size);
    err = validateSend(MPI_Send(&p.size, 1, MPI_INT, receiverProcessID, WORK, MPI_COMM_WORLD), __LINE__);
    assert(p.size > 0 && p.size < 600);
    if (!err)
        return 0;
    //printf("%d / %d\n", world_rank, world_size);
    err = validateSend(MPI_Send(p.data, p.size * p.size, MPI_INT, receiverProcessID, WORK, MPI_COMM_WORLD), __LINE__);
    if (!err)
        return 0;
    printf("\n\n\npicture dimension %d p.data size %ld line %d\n\n\n", p.size, (malloc_usable_size((void*)p.data) / sizeof(int)), __LINE__);
    //assert(malloc_usable_size((void*)p.data) > 0 && malloc_usable_size((void*)p.data) == p.size * p.size * sizeof(int));
    return 1;
}
void recievePictureFromProcess(int senderProcessID, Matrix* picture, MPI_Status* status)
{

    // Sets up a default return value (in case TERMINATETAG is met)
    picture->size = NOTFOUND;
    picture->id = NOTFOUND;
    picture->data = NULL;

    // Receives the picture's integers count from the sender process
    if (!validateRecv(MPI_Recv(&(picture->id), 1, MPI_INT, senderProcessID, MPI_ANY_TAG, MPI_COMM_WORLD, status), __LINE__))
    {
        printf("failed recieving picture id line %d", __LINE__);
        exit(0);
    }

    if ((*status).MPI_TAG == TERMINATE)
        return; // empty
    assert(picture->id > 0 && picture->id < 6);
    if (!validateRecv(MPI_Recv(&(picture->size), 1, MPI_INT, senderProcessID, MPI_ANY_TAG, MPI_COMM_WORLD, status), __LINE__))
    {
        printf("failed recieving picture size line %d", __LINE__);
        exit(0);
    }
    assert(picture->size > 0 && picture->size < 600);
    // printf("recv size %d\n", picture->size);
    picture->data = (int*)malloc(picture->size * picture->size * sizeof(int));
    if (picture->data == NULL)
    {
        printf("\nCannot allocate memory to receive packed matrix line %d\n", __LINE__);
        picture->size = NOTFOUND;
        picture->id = NOTFOUND;
        exit(0);
    }

    // Receives the packed picture from the sender process
    if (!validateRecv(MPI_Recv(picture->data, picture->size * picture->size, MPI_INT, senderProcessID, WORK, MPI_COMM_WORLD, status), __LINE__))
    {
        printf("failed recieving picture data line %d", __LINE__);
        picture->size = NOTFOUND;
        picture->id = NOTFOUND;
        exit(0);
    }
    printf("\n\n\npicture dimension %d picture->data size %ld line %d\n\n\n", picture->size, (malloc_usable_size((void*)picture->data) / sizeof(int)), __LINE__);
    for (int i = 0;i < picture->size * picture->size;i++)
        assert(picture->data[i] >= 0 && picture->data[i] <= 100);
    //assert(malloc_usable_size((void*)picture->data) > 0 && malloc_usable_size((void*)picture->data) == picture->size * picture->size * sizeof(int));
}
int searchObjectsInPicture(Matrix picture, Matrix* objects, int numObjects, double matchingV, objectFound* searchRecord)
{
    int numSet = 0;

    // Initialize all searchRecord elements
    for (int i = 0; i < 3; i++)
    {
        searchRecord[i].PID = picture.id;
        searchRecord[i].id = -1;
        searchRecord[i].pos = NULL;
    }
    int* devicePictureMatrix;
    allocateMatOnGPU(picture, &devicePictureMatrix);
    // Search for each object in the picture and update searchRecord accordingly
    for (int i = 0; i < numObjects; i++)
    {
        objectFound result;
        int searchResult = searchObjectInPicture(&result, picture, devicePictureMatrix, objects[i], matchingV);

        if (searchResult == 1 && numSet < 3)
        {
            searchRecord[numSet].id = result.id;
            searchRecord[numSet].pos = (int*)malloc(2 * sizeof(int));
            if (searchRecord[numSet].pos == NULL)
            {
                freePos(searchRecord, numSet);
                exit(0);
            }
            searchRecord[numSet].pos[0] = result.pos[0];
            searchRecord[numSet].pos[1] = result.pos[1];
            numSet++;
        }
        free(result.pos);
    }
    freeMatFromGPU(&devicePictureMatrix);
    // Fill any remaining unset searchRecord elements
    for (int i = 0; i < 3; i++)
    {
        if (searchRecord[i].pos == NULL && searchRecord[i].id == -1)
        {

            searchRecord[i].id = -1;
            searchRecord[i].pos = (int*)malloc(2 * sizeof(int));
            if (searchRecord[i].pos == NULL)
            {
                exit(0);
            }
            searchRecord[i].pos[0] = -1;
            searchRecord[i].pos[1] = -1;
        }
    }

    return 1;
}

int searchObjectInPicture(objectFound* result, Matrix picture, int* devicePictureMatrix, Matrix object, double matchingV)
{
    int actualCheckDim = picture.size - object.size + 1;
    int colorsInPicture = actualCheckDim * actualCheckDim;

    // Initializes a default result
    result->PID = picture.id;
    result->id = NOTFOUND;
    result->pos = (int*)malloc(2 * sizeof(int));
    if (result->pos == NULL)
    {
        printf("\n\n\nfailed result pos array line %d\n\n", __LINE__);
        exit(0);
    }
    result->pos[0] = -1;
    result->pos[1] = -1;
    printf("result pid %d id %d line %d\n", result->PID, result->id, __LINE__);
    assert(result->PID > 0 && result->PID <= 5);
    // Validation - Object's dimension isn't blocked by the picture's
    if (object.size > picture.size)
        return 0;

    // Searches the object within the picture using the GPU
    int* tempAddressArray = searchOnGPU(picture.size, devicePictureMatrix, object, matchingV);
    if (tempAddressArray == NULL)
    {
        printf("\nfailed searchGPU picture %d and object %d\n", picture.id, object.id);
        exit(0);
    }
    int* positionFlagsArray = (int*)malloc(colorsInPicture * sizeof(int));
    if (positionFlagsArray == NULL)
    {
        printf("\nfailed allocating positionFlags for picture %d and object %d\n", picture.id, object.id);
        exit(0);
    }
    memcpy(positionFlagsArray, tempAddressArray, colorsInPicture * sizeof(int));


    // Extracts the final search record - the first position the object was found in (if at all) or "not found"
    for (int i = 0; i < colorsInPicture; i++)
        if (positionFlagsArray[i])
        {
            printf("\n\npoisition[%d] = %d\n\n", i, positionFlagsArray[i]);
            result->id = object.id;
            assert(result->id > 0 && result->id <= 8);
            assert(i / actualCheckDim < picture.size && i % actualCheckDim < picture.size);
            result->pos[0] = i / actualCheckDim;
            result->pos[1] = i % actualCheckDim;
            break;
        }
    printf("result pid %d id %d line %d pos %d %d", result->PID, result->id, __LINE__, result->pos[0], result->pos[1]);

    free(positionFlagsArray);
    free(tempAddressArray);

    return result->pos[0] == -1 && result->pos[1] == -1 ? 0 : 1;
}
int isNotEmptyPosition(objectFound of)
{
    return (of.pos[0] != NOTFOUND) && (of.pos[1] != NOTFOUND); // 0 = row , 1 = col
}
void write_to_file(const char* filename, int num_matrices, objectFound* searchRecordsArray)
{
    FILE* fp = fopen(filename, "w");
    if (fp == NULL)
    {
        printf("Error: Failed to open file %s\n", filename);
        return;
    }
    if (searchRecordsArray == NULL)
    {
        printf("objectfound is null\n");
        MPI_Abort(MPI_COMM_WORLD, __LINE__);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < num_matrices; i++)
    {
        printf("\n\nwriting i %d\n\n", i);
        int emptyFlag = 0;
        for (int p = 0;p < 3;p++) {
            printf("\n\nwriting p %d\n\n", p);
            if (searchRecordsArray[i * 3 + p].id < -1 || searchRecordsArray[i * 3 + p].id >= 8)
            {
                printf("objectfound picture %d in result %d is %d line %d\n", searchRecordsArray[i * 3 + p].PID, p + 1, searchRecordsArray[i * 3 + p].id, __LINE__);
                MPI_Abort(MPI_COMM_WORLD, __LINE__);
                exit(EXIT_FAILURE);
            }
            else if (!isNotEmptyPosition(searchRecordsArray[i * 3 + p])) {
                fprintf(fp, "Picture %d: No three different Objects were found\n", (i + 1));
                emptyFlag = 1;
                break;
            }
        }
        if (!emptyFlag)
        {
            fprintf(fp, "Picture %d: found Objects:", i + 1);
            for (int k = 0; k < 3; k++)
                fprintf(fp, " %d Position(%d,%d) ", searchRecordsArray[i * 3 + k].id, searchRecordsArray[i * 3 + k].pos[0], searchRecordsArray[i * 3 + k].pos[1]);
            fprintf(fp, "\n");
        }

    }

}
void freePos(objectFound* searchRecordsArray, int size)
{
    for (int i = 0; i < size; i++)
    {
        free(searchRecordsArray[i].pos);
    }
}
void freeMatrix(Matrix* mt, int size)
{
    for (int i = 0; i < size; ++i)
    {
        free(mt[i].data);
    }
    free(mt);
}
// debugging
int validateSend(int res, int line)
{
    int result = 0;
    printf("\nsend ");
    switch (res)
    {
    case MPI_SUCCESS:
        printf("No error; MPI routine completed successfully. line %d \n", line);
        result = 1;
        break;

    case MPI_ERR_COMM:
        printf("Invalid communicator.A common error is to use a null communicator in a call(not even allowed in MPI_Comm_rank). line %d \n", line);
        result = 0;
        break;
    case MPI_ERR_COUNT:
        printf("Invalid count argument.Count arguments must be non - negative; a count of zero is often valid. line %d \n", line);
        result = 0;
        break;
    case MPI_ERR_TYPE:
        printf("Invalid datatype argument.Additionally, this error can occur if an uncommitted MPI_Datatype(see MPI_Type_commit) is used in a communication call. line %d \n", line);
        result = 0;
        break;

    case MPI_ERR_TAG:
        printf("Invalid tag argument.Tags must be non -negative; tags in a receive(MPI_Recv, MPI_Irecv, MPI_Sendrecv, etc.) may also be MPI_ANY_TAG.The largest tag value is available through the the attribute MPI_TAG_UB. line %d \n", line);
        result = 0;
        break;

    case MPI_ERR_RANK:
        printf("Invalid source or destination rank. Ranks must be between zero and the size of the communicator minus one; ranks in a receive (MPI_Recv, MPI_Irecv, MPI_Sendrecv, etc.) may also be MPI_ANY_SOURCE.  line %d \n", line);
        result = 0;
        break;
    default:
        printf("other... %d\n", line);

    }
    return result;
}
int validateRecv(int res, int line)
{
    int result = 0;
    printf("\nrecv ");
    switch (res)
    {
    case MPI_SUCCESS:
        printf("No error; MPI routine completed successfully. line %d \n", line);
        result = 1;
        break;
    case MPI_ERR_COMM:
        printf("Invalid communicator.A common error is to use a null communicator in a call(not even allowed in MPI_Comm_rank). line %d \n", line);
        result = 0;
        break;

    case MPI_ERR_TYPE:
        printf("Invalid communicator.A common error is to use a null communicator in a call(not even allowed in MPI_Comm_rank). line %d \n", line);
        result = 0;
        break;

    case MPI_ERR_COUNT:
        printf(" Invalid count argument.Count arguments must be non - negative; a count of zero is often valid.. line %d \n", line);
        result = 0;
        break;

    case MPI_ERR_TAG:
        printf("Invalid tag argument.Tags must be non -negative; tags in a receive(MPI_Recv, MPI_Irecv, MPI_Sendrecv, etc.) may also be MPI_ANY_TAG.The largest tag value is available through the the attribute MPI_TAG_UB. %d \n", line);
        result = 0;
        break;

    case MPI_ERR_RANK:
        printf("Invalid source or destination rank.Ranks must be between zero and the size of the communicator minus one; ranks in a receive(MPI_Recv, MPI_Irecv, MPI_Sendrecv, etc.) may also be MPI_ANY_SOURCE. line %d \n", line);
        result = 0;
        break;

    default:
        printf("other... %d \n", line);
        result = 0;
        break;
    }
    return result;
}
void validateBcast(int res, int line, char sr[5])
{
    switch (res)
    {
    case MPI_SUCCESS:
        printf("%s bcast data successfully %d\n", sr, line);
        break;
    case MPI_ERR_COMM:
        printf("%s MPI_ERR_COMM %d\n", sr, line);
        break;
    case MPI_ERR_COUNT:
        printf("%s MPI_ERR_COUNT %d\n", sr, line);
        break;
    case MPI_ERR_TYPE:
        printf("%s MPI_ERR_TYPE %d\n", sr, line);
        break;
    case MPI_ERR_BUFFER:
        printf("%s MPI_ERR_BUFFER %d\n", sr, line);
        break;
    case MPI_ERR_ROOT:
        printf("%s MPI_ERR_ROOT %d\n", sr, line);
        break;
    default:
        printf("%s other... %d\n", sr, line);
        break;
    };
}
int MasterDynamicWay(objectFound* searchRecordsArray, Matrix* arrP, int numOfps, int size, MPI_Status* status)
{
    // the main program for process 0 - the master\manager of the program
    //first it starts to send dynamically each picture to a "slave" project 
    //after all slaves recieved their first job, the process waits for them to process and sends back the finding results 
    // the finding results contained of 3 objectFound struct objects that contain the finding result with an object and a picture
    // contains metadata information about the picture and object checked and the position of the the first coordinate of the sliding window algorithm that was searched
    // if the searching process failed to find it will result in position array of (-1,-1) 
    // in case of other unrelated unexpected failure id of object in the metadata will result in -1


    //counter of results recieved
    int resultsReceived = 0;
    int sentPictureOffset;
    printf("Starting dynamic task pool with %d slaves line %d\n", size - 1, __LINE__);
    // starting to run the exec time
    // Sends each process its initial picture to handle
    for (sentPictureOffset = 0; sentPictureOffset < (size - 1); sentPictureOffset++)
    {
        printf("sendPictureToProcess line %d\n", __LINE__);
        if (!sendPictureToProcess(arrP[sentPictureOffset], sentPictureOffset + 1))
        {
            free(searchRecordsArray);
            exit(0);
        }
        printf("Master sent Picture %d to slave %d line %d\n ", sentPictureOffset + 1, sentPictureOffset + 1, __LINE__);
    }
    ////-----------------------------part 1 succeeded-----------------------------------------

    sentPictureOffset == size - 1;

    // While there are pictures to handle 
    while (sentPictureOffset < numOfps)
    {

        if (!recievingResults(searchRecordsArray, &resultsReceived, status))
        {
            printf("Error: failed receiving results %d\n", __LINE__);
            free(searchRecordsArray);
            exit(0);
        }
#pragma omp parallel for
        for (int i = resultsReceived - 3;i < resultsReceived;i++)
        {
            printf("\n\nPID %d id %d\n\n", searchRecordsArray[i].PID, searchRecordsArray[i].id);
            assert(searchRecordsArray[i].PID > 0 && searchRecordsArray[i].PID <= 5);
            assert(searchRecordsArray[i].id >= -1 && searchRecordsArray[i].id <= 8);
            assert(searchRecordsArray[i].pos[0] >= -1 && searchRecordsArray[i].pos[0] <= 800);
            assert(searchRecordsArray[i].pos[1] >= -1 && searchRecordsArray[i].pos[1] <= 800);
        }
        printf("recievingResults success line %d\n", __LINE__);
        // Sends the next picture to the available process

        if (!sendPictureToProcess(arrP[sentPictureOffset++], (*status).MPI_SOURCE))
        {
            printf("Error: failed sending picture to process %d\n", __LINE__);
            free(searchRecordsArray);
            exit(0);
        }
        printf("sendPictureToProcess success line %d\n", __LINE__);
        printf("\n\n\n sentPictureOffset %d numofps %d\n\n\n", sentPictureOffset, numOfps);
        printf("Master sent picture %d to slave %d line %d\n", sentPictureOffset - 1, (*status).MPI_SOURCE, __LINE__);
    }
    assert(sentPictureOffset == numOfps);
    // Waits for all slaves to finish their works
    for (int term = 0;term < size - 1;term++)
    {
        printf("recievingResults line %d\n", __LINE__);
        if (!recievingResults(searchRecordsArray, &resultsReceived, status))
        {
            printf("Error: failed recieving results %d\n", __LINE__);
            free(searchRecordsArray);
            exit(0);
        }
#pragma omp parallel for
        for (int i = resultsReceived - 3;i < resultsReceived;i++)
        {
            assert(searchRecordsArray[i].PID > 0 && searchRecordsArray[i].PID <= 5);
            assert(searchRecordsArray[i].id >= -1 && searchRecordsArray[i].id <= 8);
            assert(searchRecordsArray[i].pos[0] >= -1 && searchRecordsArray[i].pos[0] <= 800);
            assert(searchRecordsArray[i].pos[0] >= -1 && searchRecordsArray[i].pos[0] <= 800);
        }
        // master sends termination message
        if (!validateSend(MPI_Send(NULL, 0, MPI_INT, (*status).MPI_SOURCE, TERMINATE, MPI_COMM_WORLD), __LINE__))
        {
            printf("Error: failed sending picture to process %d\n", __LINE__);
            free(searchRecordsArray);
            exit(0);
        }
        printf("Master sent termination tag to slave %d line %d\n", (*status).MPI_SOURCE, __LINE__);
    }
    return 1;
}
int SlaveDynamicWay(Matrix* arrO, int numOfobjs, double matchingV, int rank, MPI_Status* status)
{
    // dynamic approach of workers
    // worker process waits for the master manager process to send the picture to be able to execute the job required
    // the worker recieve picture and start to check for matching results by the difference method by cuda. the search done by openMP and cuda using the sliding window algorithm approach for parallel
    // when done scanning all objects with the picture start minimizing the results for 3 different object findings
    // the master will just need to recieve results and write to file immidietly
    Matrix picture;
    objectFound* searchRecord = NULL;
    printf("recievePictureFromProcess line %d\n", __LINE__);
    recievePictureFromProcess(0, &picture, status);
    if (picture.id == NOTFOUND || picture.id < 1 || picture.id > 4)
    {
        printf("\nrecieve failed for slave %d line %d\n", rank, __LINE__);
        exit(0);
    }
    do {
        printf("Slave %d received picture %d from master line %d\n", rank, picture.id, __LINE__);

        //----------------------------------------part 1 succeeded----------------------------------------------

        // search record for 3 object findings
        searchRecord = (objectFound*)malloc(3 * sizeof(objectFound));
        if (searchRecord == NULL)
        {
            printf("\nfailed allocating searchRecord for slave %d line %d\n", rank, __LINE__);
            free(picture.data);
            exit(0);

        }
        for (int i = 0;i < 3;i++)
        {
            searchRecord[i].pos = (int*)malloc(2 * sizeof(int));
            if (searchRecord[i].pos == NULL)
            {
                printf("\nfailed allocating searchRecord for slave %d line %d\n", rank, __LINE__);
                freePos(searchRecord, i);
                free(searchRecord);
                free(picture.data);
                exit(0);
            }
        }
        printf("\n\n\nsuccess recieve picture line %d\n\n\n", __LINE__);

        //---------------------------------part 2 succeeded ----------------------------------------------

            // Searches the picture and sends the search record to the master
        printf("slave %d calculates results line %d", rank, __LINE__);
        printf("searchObjectsInPicture line %d\n", __LINE__);
        searchObjectsInPicture(picture, arrO, numOfobjs, matchingV, searchRecord);



#pragma omp parallel for
        for (int i = 0; i < 3; i++) {

            printf("\nsearchRecord[%d]= pid %d obj id %d  pos %d %d line %d\n", i, (searchRecord)[i].PID, (searchRecord)[i].id, (searchRecord)[i].pos[0], (searchRecord)[i].pos[1], __LINE__);
            assert((searchRecord)[i].PID > 0 && (searchRecord)[i].PID < 6);
            assert((searchRecord)[i].id >= -1 && (searchRecord)[i].id < 9);
            assert((searchRecord)[i].pos[0] >= -1 && (searchRecord)[i].pos[0] < 800);
            assert((searchRecord)[i].pos[1] >= -1 && (searchRecord)[i].pos[1] < 800);
        }
        printf("\n\nsuccess validating searchRecord line %d", __LINE__);

        // Send the search record to the master
        for (int i = 0; i < 3; i++) {
            int buffer[4];
            serialize_objectFound(searchRecord[i], buffer);
            if (!validateSend(MPI_Send(buffer, 4, MPI_INT, 0, WORK, MPI_COMM_WORLD), __LINE__))
            {
                printf("\nfailed sending results for slave %d line %d\n", rank, __LINE__);
                freePos(searchRecord, 3);
                free(searchRecord);
                free(picture.data);
                exit(0);
            }

        }

        printf("Slave %d sent search record for picture %d to master line %d\n", rank, searchRecord[0].PID, __LINE__);

        freePos(searchRecord, 3);
        free(searchRecord);
        //Receives the next picture to handle
        free(picture.data);

        searchRecord = NULL;


        recievePictureFromProcess(0, &picture, status);

    } while (picture.id != NOTFOUND);
    printf("Slave %d terminated successfully\n", rank);
    return 1;

}

void printMat(Matrix p)
{
    int counter = 0;
    for (int i = 0; i < p.size; i++)
    {
        for (int j = 0; j < p.size; j++)
        {
            printf("%3d ", p.data[i * p.size + j]);
            counter++;
        }
        printf("\n");
    }
    printf("\n\n\ncounter %d\n\n\n\n", counter);
}
// Serialization function to convert objectFound struct to contiguous memory block
void serialize_objectFound(objectFound obj, int* buffer) {
    int index = 0;
    buffer[index++] = obj.PID;
    buffer[index++] = obj.id;
    buffer[index++] = obj.pos[0];
    buffer[index++] = obj.pos[1];
}

// Deserialization function to convert contiguous memory block to objectFound struct
objectFound deserialize_objectFound(int* buffer) {
    objectFound obj;
    int index = 0;
    obj.PID = buffer[index++];
    obj.id = buffer[index++];
    obj.pos = (int*)malloc(2 * sizeof(int));
    obj.pos[0] = buffer[index++];
    obj.pos[1] = buffer[index++];
    return obj;
}