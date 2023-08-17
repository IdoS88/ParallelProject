build:	

	mpicxx -fopenmp -g -c main.c -lc -o main.o	
	mpicxx -fopenmp -g -c cFunctions.c -o cFunctions.o 
	nvcc -I./Common -g -G -gencode arch=compute_61,code=sm_61 -c CudaFunctions.cu  -o CudaFunctions.o 
	mpicxx -fopenmp -o mpiCudaOpenMP main.o cFunctions.o CudaFunctions.o /usr/local/cuda/lib64/libcudart_static.a -ldl -lrt 
clean:
	rm -f *.o ./mpiCudaOpemMP

run: 
	mpiexec -n 2 ./mpiCudaOpenMP 


	
runOn2:
	mpiexec -np 2 -hostfile hosts.txt -map-by node ./mpiCudaOpenMP 
