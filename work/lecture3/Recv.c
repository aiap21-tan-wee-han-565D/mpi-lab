#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
  int nprocs, rank;
  char message[100];
  MPI_Status status;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Recv(message, 100, MPI_CHAR, MPI_ANY_SOURCE,
           MPI_ANY_TAG, MPI_COMM_WORLD, &status);

  printf("Process #%d receives \"%s\" from Process #%d.\n",
         rank, message, status.MPI_SOURCE);

  MPI_Finalize();
  return EXIT_SUCCESS;
}
