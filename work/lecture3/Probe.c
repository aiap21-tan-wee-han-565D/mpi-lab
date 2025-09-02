#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
  int nprocs, rank, root = 0;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == root) {     // for the process that sends the message
    char message1[100], message2[100];
    int len1, len2;
    MPI_Request req[2*nprocs];

    srand(time(NULL));
    len1 = sprintf(message1, "Random number #1: %d.", rand());
    len2 = sprintf(message2, "Random number #2: %d.", rand());
    for (int i = 0; i < nprocs; i++) {
      if (i != rank) {
        MPI_Isend(message1, len1 + 1, MPI_CHAR, i, 1, MPI_COMM_WORLD, &req[2*i]);
        MPI_Isend(message2, len2 + 1, MPI_CHAR, i, 2, MPI_COMM_WORLD, &req[2*i+1]);
      } else
        req[2*i] = req[2*i+1] = MPI_REQUEST_NULL;
    }
    MPI_Waitall(2*nprocs, req, MPI_STATUSES_IGNORE);
  } else {                // for processes that receives the message
    MPI_Status status;
    char *message1, *message2;
    int count;

    MPI_Probe(root, 1, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_CHAR, &count);
    message1 = (char *)malloc(sizeof(char) * count);
    MPI_Recv(message1, count, MPI_CHAR, root, 1, MPI_COMM_WORLD, &status);

    MPI_Probe(root, 2, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_CHAR, &count);
    message2 = (char *)malloc(sizeof(char) * count);
    MPI_Recv(message2, count, MPI_CHAR, root, 2, MPI_COMM_WORLD, &status);

    printf("Rank %d: The two messages are:\n\t%s\n\t%s\n", rank, message1, message2);
    free(message1);
    free(message2);
  }
  MPI_Finalize();
  return EXIT_SUCCESS;
}
