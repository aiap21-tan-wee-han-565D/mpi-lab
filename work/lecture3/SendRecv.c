#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
  int nprocs, rank, root = 0;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == root) {     // for the process that sends the message
    char message[100] = "Hello!";
    for (int i = 0; i < nprocs; i++)
      if (i != rank) MPI_Send(message, 100, MPI_CHAR, i, 0, MPI_COMM_WORLD);
    printf("Process #%d sends \"%s\" to all other processes.\n", rank, message);
  } else {                // for processes that receive the message
    MPI_Status status;
    char message[100];
    MPI_Recv(message, 100, MPI_CHAR, root, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    printf("Process #%d receives \"%s\" from Process #%d.\n", rank, message, root);
  }
  MPI_Finalize();
  return EXIT_SUCCESS;
}
