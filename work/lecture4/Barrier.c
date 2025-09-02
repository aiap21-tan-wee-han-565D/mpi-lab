#include <mpi.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000

int main(int argc, char** argv) {
  int rank, nprocs, i, n = 0;
  double begin;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  srand(time(NULL) + rank); // this is used to generate seed for rng

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) begin = MPI_Wtime();
  // n/N should be approximately pi/4. generate pi.
  // just to show we do something. 
  for (i = 0; i < N; i++) {
    /* Generate two random numbers between 0 and 1 */
    double x = 1. * rand() / RAND_MAX;
    double y = 1. * rand() / RAND_MAX;

    /* Find the number of points inside the unit circle */
    if (x*x + y*y < 1) n++;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0)
    printf("Monte Carlo takes %lf seconds.\n", MPI_Wtime() - begin);

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    begin = MPI_Wtime();
    /*source = i means it expects each specific rank in turn.
    MPI_ANY_TAG accepts any tag (senders use tag 0).
    The sum accumulates into rank 0’s own n, producing the global total.*/
    for (i = 1; i < nprocs; i++) {
      int n0;
      MPI_Recv(&n0, 1, MPI_INT, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      n += n0;
    }
  } else {
    MPI_Send(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0)
    printf("Data transfer takes %lf seconds.\nResult: pi = %lf\n",
        MPI_Wtime() - begin, 4. * n / (N * nprocs));

  MPI_Finalize();
  return 0;
}
