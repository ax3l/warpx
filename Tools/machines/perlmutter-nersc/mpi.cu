#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Allocate GPU memory
    int* d_data;
    cudaMalloc(&d_data, sizeof(int));

    if (rank == 0) {
        // Set value on GPU
        int h_data = 42;
        cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice);

        // Send directly from GPU memory
        MPI_Send(d_data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        std::cout << "Rank 0: Sent value from GPU\n";
    } else if (rank == 1) {
        // Receive directly to GPU memory
        MPI_Recv(d_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Copy back to verify
        int h_data;
        cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << "Rank 1: Received " << h_data << " on GPU\n";
    }

    cudaFree(d_data);
    MPI_Finalize();
    return 0;
}
