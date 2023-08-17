#ifndef MAIN_H
#define MAIN_H
// matrix struct for picture and object
typedef struct
{
    int id; 
    int size; //size dim 
    int* data;
} Matrix;

// object findings in picture struct
typedef struct __attribute__((packed))
{
    int PID; // picture id
    int id; // object id
    int* pos; // coordinates
} objectFound;

#define BLOCK_SIZE 256		// for CUDA calculations
#define TERMINATE 1
#define NOTFOUND -1
#define WORK 0
#define EMPTY 2

#define FILENAME "input.txt"
// ======================================= External methods =======================================
extern void allocateMatOnGPU(Matrix mat, int** deviceMat);
extern void freeMatFromGPU(int** deviceMat);
extern int* searchOnGPU(int pictureDim, int* devicePictureMatrix, Matrix object, double matchingV);
int readMatrixes(Matrix* matrices, int numSize, FILE* fp);
//void readingMatrices(FILE* fp, int* size, int* numOfps, int* numOfobjs, double* matchingV, Matrix* arrP, Matrix* arrO);
int readInputFile(Matrix** pictures, int* numPictures, Matrix** objects, int* numObjects, double* matching_value);
void broadcasting(double* matchingV, int* numOfobjs, Matrix** arrO, int rank);
int isNotEmptyPosition(objectFound of);
//matching calculations
double distance(int p, int o);
int matching(Matrix pic, int y, int x, Matrix obj, double value);
Matrix read_mat(FILE* fp, int id);
void freeMatrix(Matrix* mt, int size);
void freePos(objectFound* searchRecordsArray, int size);
void write_to_file(const char* filename, int num_matrices, objectFound* of);
void serialize_objectFound(objectFound obj, int* buffer);
objectFound deserialize_objectFound(int* buffer);

void printMat(Matrix p);
int sendPictureToProcess(Matrix p, int receiverProcessID);

int searchObjectsInPicture(Matrix picture, Matrix* objects, int numObjects, double matchingV, objectFound* searchRecord);
int searchObjectInPicture(objectFound* result, Matrix picture, int* devicePictureMatrix, Matrix object, double matchingV);

// for debugging
int validateRecv(int res, int line);
int validateSend(int res, int line);
void validateBcast(int res, int line, char sr[5]);



#endif

