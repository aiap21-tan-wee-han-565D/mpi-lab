#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
  int i, nprocs, rank, random_number;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int buffer[nprocs];
  srand(rank * 2); random_number = rand();

  MPI_Allgather(&random_number, 1, MPI_INT, buffer, 1, MPI_INT, MPI_COMM_WORLD);
  for (i = 0; i < nprocs; i++) {
    printf("Rank %d: Rank %d has random number %d\n", rank, i, buffer[i]);
  }

  MPI_Finalize();
  return EXIT_SUCCESS;
}
