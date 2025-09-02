#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 100000

int main(int argc, char *argv[])
{
  int nprocs, rank, dest, source;
  char random_message[N], received_message[N];

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  srand(time(NULL) + rank);
  sprintf(random_message, "%d", rand());

  dest = (rank + 1) % nprocs;
  MPI_Send(random_message, N, MPI_CHAR, dest, 0, MPI_COMM_WORLD);

  source = (rank + nprocs - 1) % nprocs;
  MPI_Recv(received_message, N, MPI_CHAR, source, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  printf("Process #%d sends \"%s\" to Process #%d, and receives \"%s\" from Process #%d\n",
      rank, random_message, dest, received_message, source);

  MPI_Finalize();
  return EXIT_SUCCESS;
}

