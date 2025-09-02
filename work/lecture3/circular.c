#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 100000

int main(int argc, char *argv[])
{
  int nprocs, rank, dest, source;
  char random_message[N], received_message[N];
  MPI_Request req[2];

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  srand(time(NULL) + rank);
  sprintf(random_message, "%d", rand());

  dest = (rank + 1) % nprocs;
  MPI_Isend(&random_message, N, MPI_CHAR, dest, 0, MPI_COMM_WORLD, &req[0]);

  source = (rank + nprocs - 1) % nprocs;
  MPI_Irecv(&received_message, N, MPI_CHAR, source, MPI_ANY_TAG, MPI_COMM_WORLD, &req[1]);

  MPI_Waitall(2, req, MPI_STATUSES_IGNORE);

  printf("Process #%d sends \"%s\" to Process #%d, and receives \"%s\" from Process #%d\n",
      rank, random_message, dest, received_message, source);

  MPI_Finalize();

  return EXIT_SUCCESS;
}

