#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
  int nprocs, rank, name_len;
  char proc_name[100];
  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Get_processor_name(proc_name, &name_len);
  printf("Hello! I am Process #%d of %d, executed on %s.\n", rank, nprocs, proc_name);

  MPI_Finalize();

  return EXIT_SUCCESS;
}
