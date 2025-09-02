#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
  int i, idx, nprocs, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int buffer[nprocs];
  MPI_Request requests[nprocs];

  // Generate random numbers on rank 0
  srand(time(NULL) + rank);
  if (rank == 0) {
    for (i = 0; i < nprocs; i++) buffer[i] = rand();
  }

  // Broadcast all the random numbers
  for (i = 0; i < nprocs; i++)
    MPI_Ibcast(&buffer[i], 1, MPI_INT, 0, MPI_COMM_WORLD, &requests[i]);

  // Print the numbers in the order that they are received
  for (i = 0; i < nprocs; i++) {
    MPI_Waitany(nprocs, requests, &idx, MPI_STATUSES_IGNORE);
    printf("Rank %d: Random number %d is %d\n", rank, idx, buffer[idx]);
  }

  MPI_Finalize();
  return EXIT_SUCCESS;
}
