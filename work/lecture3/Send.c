#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
  int nprocs, rank;
  char message[100] = "Hello!";

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  for (int i = 0; i < nprocs; i++)
    if (i != rank) MPI_Send(message, 100, MPI_CHAR, i, 0, MPI_COMM_WORLD);

  printf("Process #%d sends \"%s\" to all other processes.\n", rank, message);

  MPI_Finalize();
  return EXIT_SUCCESS;
}
