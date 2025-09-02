#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

/* Add the new data to the training set and the test set */
void append_data(int n_item, int n_feature,
    double **data_training, double **data_test, int *n_training, int *n_test, double *tmp)
{
  double *random_numbers = (double *)malloc(n_item * sizeof(double));
  int i, n_training_new = 0;
  for (i = 0; i < n_item; i++) {
    random_numbers[i] = (double)rand() / RAND_MAX;
    if (random_numbers[i] < 0.7) n_training_new++;
  }

  *data_training = (double *)realloc(*data_training,
      (n_training_new + *n_training) * (n_feature + 1) * sizeof(double));
  *data_test = (double *)realloc(*data_test,
      (n_item - n_training_new + *n_test) * (n_feature + 1) * sizeof(double));

  for (int i = 0; i < n_item; i++) {
    if (random_numbers[i] < 0.7) {
      memcpy((*data_training) + (*n_training) * (n_feature + 1),
          tmp + i * (n_feature + 1), (n_feature + 1) * sizeof(double));
      (*n_training)++;
    } else {
      memcpy((*data_test) + (*n_test) * (n_feature + 1),
          tmp + i * (n_feature + 1), (n_feature + 1) * sizeof(double));
      (*n_test)++;
    }
  }
  free(random_numbers);
}

/* Read data from file and distribute them to all processes */
void read_data(char *file_name, int n_feature,
    double **data_training, double **data_test, int *n_training, int *n_test)
{
  const int max_sample = 100, max_req = 50;
  double *tmp = (double *)malloc((n_feature + 1) * max_sample * sizeof(double));
  int n_item = 0, dest = 0, rank, nprocs;
  *n_training = *n_test = 0;
  *data_training = *data_test = NULL;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (rank == 0) {
    FILE *fp = fopen(file_name, "r");
    int read_success = 1, n_req = 0;
    MPI_Request *send_req = (MPI_Request *)malloc((max_req + nprocs) * sizeof(MPI_Request));
    while (read_success) {
      /* Read one row of data */
      int feature_idx = 0;
      while (read_success && feature_idx <= n_feature) {
        if (fscanf(fp, "%lf", tmp + n_item * (n_feature + 1) + feature_idx) == EOF) {
          read_success = 0;
          fclose(fp);
        }
        feature_idx++;
      }
      if (read_success) n_item++;
      
      /* Distribute data to other processes */
      if (n_item >= max_sample) {
        if (dest == rank) {
          append_data(n_item, n_feature, data_training, data_test, n_training, n_test, tmp);
        } else {
          if (n_req >= max_req) {
            MPI_Waitall(n_req, send_req, MPI_STATUSES_IGNORE);
            n_req = 0;
          }
          MPI_Isend(tmp, n_item * (n_feature + 1), MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &send_req[n_req++]);
        }
        n_item = 0;
        dest = (dest + 1) % nprocs;
      }
    }

    /* Send remaining data */
    if (n_item > 0) {
      if (dest == rank)
        append_data(n_item, n_feature, data_training, data_test, n_training, n_test, tmp);
      else
        MPI_Isend(tmp, n_item * (n_feature + 1), MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &send_req[n_req++]);
    }

    /* Send empty messages to indicate the end of data */
    for (dest = 1; dest < nprocs; dest++) {
      MPI_Isend(NULL, 0, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &send_req[n_req++]);
    }

    MPI_Waitall(n_req, send_req, MPI_STATUSES_IGNORE);
    free(send_req);
  } else {
    MPI_Status status;
    while (1) {
      MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
      MPI_Get_count(&status, MPI_DOUBLE, &n_item);
      MPI_Recv(tmp, n_item, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (n_item == 0) break;
      append_data(n_item / (n_feature + 1), n_feature, data_training, data_test, n_training, n_test, tmp);
    }
  }

  free(tmp);
}

/* Local summation */
void summation(double *data, int n_ele, int n_sample, double *sum)
{
  int i, n;
  for (i = 0; i < n_ele; i++) sum[i] = 0;
  for (n = 0; n < n_sample; n++)
    for (i = 0; i < n_ele; i++, data++)
      sum[i] = sum[i] + *data;
}

/* Global averaging */
void average(const double *sum, int n_ele, int n_sample, int nprocs, int rank, double *avg)
{
  if (rank == 0) {
    double *sums = (double *)malloc(sizeof(double) * n_ele * nprocs);
    int i, j, *n = (int *)malloc(sizeof(int) * nprocs);
    MPI_Gather(sum, n_ele, MPI_DOUBLE, sums, n_ele, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&n_sample, 1, MPI_INT, n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    for (i = 0; i < n_ele; i++) avg[i] = 0;
    for (i = 0; i < nprocs; i++) {
      for (j = 0; j < n_ele; j++)
        avg[j] += sums[i * n_ele + j];
      if (i != 0) n[0] += n[i];
    }
    for (i = 0; i < n_ele; i++) avg[i] /= n[0];

    free(sums); free(n);
  } else {
    MPI_Gather(sum, n_ele, MPI_DOUBLE, NULL, n_ele, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&n_sample, 1, MPI_INT, NULL, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }
  MPI_Bcast(avg, n_ele, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

/* Compute the local part of the coefficient matrix and the right-hand side */
void weight_coef(double *data, int n_feature, int n_sample, double *avg, double *sum)
{
  int i, j, n, len = n_feature * (n_feature + 3) / 2;
  for (i = 0; i < len; i++) sum[i] = 0;

  double *p_data = data;
  for (n = 0; n < n_sample; n++, p_data += n_feature + 1) {
    for (i = 0; i < n_feature; i++)
      sum[i] += p_data[n_feature] * (p_data[i] - avg[i]);

    for (i = 0; i < n_feature; i++) {
      len = n_feature + i*(i+1)/2;
      for (j = 0; j <= i; j++)
        sum[len + j] += (p_data[i] - avg[i]) * (p_data[j] - avg[j]);
    }
  }
}

/* Use LDL^T decomposition to solve the linear system */
void solve_spd_system(int size, double *matrix, double *rhs, double *solution)
{
  int i, j, k;

  /* LDL^T decomposition */
  for (i = 0; i < size; i++) {
    int len_i = i*(i+1)/2;
    for (j = 0; j <= i; j++) {
      int len_j = j*(j+1)/2;
      double sum = 0;
      for (k = 0; k < j; k++)
        sum += matrix[len_i + k] * matrix[len_j + k] * matrix[k*(k+3)/2];

      if (i == j)
        matrix[len_i + j] = matrix[len_i + i] - sum;
      else
        matrix[len_i + j] = (1.0 / matrix[len_j + j] * (matrix[len_i + j] - sum));
    }
  }

  /* Forward substitution */
  for (i = 1; i < size; i++) {
    int len_i = i*(i+1)/2;
    for (j = 0; j < i; j++)
      rhs[i] -= matrix[len_i + j] * rhs[j];
  }

  /* Solve the diagonal system */
  for (i = 0; i < size; i++) solution[i] = rhs[i] / matrix[i * (i+3)/2];

  /* Backward substitution */
  for (i = size-1; i > 0; i--) {
    int len_i = i*(i+1)/2;
    for (j = 0; j < i; j++)
      solution[j] -= matrix[len_i + j] * solution[i];
  }
}

/* Compute weights of the linear system */
void compute_weight(double *sum, double* avg, int n_feature, int n_sample, int nprocs, int rank, double *weight)
{
  int len = n_feature * (n_feature + 3) / 2;
  if (rank == 0) {
    int i;
    average(sum, len, n_sample, nprocs, rank, sum);

    solve_spd_system(n_feature, sum + n_feature, sum, weight);

    weight[n_feature] = avg[n_feature];
    for (i = 0; i < n_feature; i++)
      weight[n_feature] -= weight[i] * avg[i];

    MPI_Bcast(weight, n_feature + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    average(sum, len, n_sample, nprocs, rank, sum);
    MPI_Bcast(weight, n_feature + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
}

/* Compute the RMSE */
double test(double *data, int n_feature, int n_sample, int nprocs, int rank, double *weight) {
  int i, j;
  double squared_err = 0, avg;
  for (i = 0; i < n_sample; i++) {
    double diff = 0;
    for (j = 0; j < n_feature; j++, data++)
      diff += *data * weight[j];
    diff += weight[n_feature] - *(data++);
    squared_err += diff * diff;
  }
  average(&squared_err, 1, n_sample, nprocs, rank, &avg);
  return sqrt(avg);
}

int main(int argc, char *argv[])
{
  int nprocs, rank, n_training, n_test, n_feature, i;
  char file_name[100];
  double *data_training, *data_test, *sum, *avg, *weight, rmse, begin;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (argc < 3) {
    if (rank == 0)
      fprintf(stderr, "Usage: %s <Number of Features> <File Name>\n", argv[0]);
    MPI_Finalize();
    return 1;
  }
  n_feature = atoi(argv[1]);

  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  srand(rank + time(NULL));
  read_data(argv[2], n_feature, &data_training, &data_test, &n_training, &n_test);
  printf("Rank %d: Local training data size: %d\n", rank, n_training);

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) begin = MPI_Wtime();
  sum = (double *)malloc(sizeof(double) * (n_feature + 3) * n_feature / 2);
  summation(data_training, n_feature + 1, n_training, sum);

  avg = (double *)malloc(sizeof(double) * (n_feature + 1));
  average(sum, n_feature + 1, n_training, nprocs, rank, avg);

  weight = (double *)malloc(sizeof(double) * (n_feature + 1));
  weight_coef(data_training, n_feature, n_training, avg, sum);
  compute_weight(sum, avg, n_feature, n_training, nprocs, rank, weight);
  if (rank == 0) {
    printf("Weights: [%lf", weight[0]);
    for (i = 1; i <= n_feature; i++) printf(", %lf", weight[i]);
    printf("]\n");
  }

  rmse = test(data_training, n_feature, n_training, nprocs, rank, weight);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    printf("Root mean squared error for training data: %lf\n", rmse);
    printf("Elasped time: %lf seconds\n\n", MPI_Wtime() - begin);
  }

  printf("Rank %d: Local test data size: %d\n", rank, n_test);
  rmse = test(data_test, n_feature, n_test, nprocs, rank, weight);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0)
    printf("Root mean squared error for test data: %lf\n", rmse);

  free(data_training);
  free(data_test);
  free(sum);
  free(avg);
  free(weight);

  MPI_Finalize();
}
