# ParallelProject
# Distributed Object Search Project

## Overview

This project addresses a defined challenge involving P images and O objects extracted from an input file. The project orchestrates a multi-process architecture comprising a dynamic master process and multiple slave processes, with the objective of optimizing computational efficiency.

The master process dynamically allocates initial work segments (images) to each slave process. Upon completion of a task by a slave process, the master dynamically assigns new work. Upon concluding the distribution of all workloads, the master collates the results from each slave process and subsequently generates a comprehensive output file.

Each slave process undertakes a distinct set of tasks, including processing object searches against provided images. The slave process returns results to the master, encompassing object presence or absence and corresponding coordinates.

## Project Complexity

The project showcases a general time complexity of O(Log(P)) where P denotes the total number of images. This complexity is attributed to the parallel processing approach, which outperforms serial execution times.

## Task Decomposition

- **MPI (Message Passing Interface):** Employed to facilitate inter-process communication. The master process orchestrates data distribution among slave processes and collects aggregated results. This approach optimally handles data sharing and result gathering, constituting a pivotal aspect of the project's implementation.

- **OMP (Open Multi-Processing):** Leverages parallel threads within each process to conduct object searches within provided images. This fine-grained concurrency approach ensures efficiency, as the search for distinct objects within an image does not influence the search for other objects.

- **CUDA (Compute Unified Device Architecture):** Harnessed to capitalize on the computational power of GPUs. This approach is pivotal in managing the processing demands associated with large-dimensional images and objects, facilitating true parallelism of computations.

The project elegantly melds these techniques to craft an effective solution that addresses the inherent challenges of image and object analysis in a highly efficient and distributed manner.

## How to Use

1. Clone the repository.
2. Build and run the project using "make build" and "make run".
3. View the output files and results in [output directory].
**There is an option to run on 2 computers with the command specified in the make file "make runOn2" but you need to modify the hosts.txt file with both 2 computers's IP.
## Contributions

Contributions are welcome! Feel free to submit pull requests or open issues for enhancements, bug fixes, or suggestions.


---

Developed by Ido Shamir
