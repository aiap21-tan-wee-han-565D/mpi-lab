#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

typedef struct {
  int rank; // Rank of the current process
  int n; // Size of the vector
  int* n_cpnt; // Number of components in process
  int* loc; // Location of the first element in each process
} size_info_t;

void create_size_info(int n, size_info_t *size)
{
  int i, nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &size->rank);

  size->n_cpnt = (int *)malloc(sizeof(int) * nprocs);
  size->loc = (int *)malloc(sizeof(int) * nprocs);
  MPI_Allgather(&n, 1, MPI_INT, size->n_cpnt, 1, MPI_INT, MPI_COMM_WORLD);

  size->loc[0] = 0; size->n = size->n_cpnt[0];
  for (i = 1; i < nprocs; i++) {
    size->loc[i] = size->loc[i-1] + size->n_cpnt[i-1];
    size->n += size->n_cpnt[i];
  }
}

void destroy_size_info(size_info_t *size)
{
  free(size->n_cpnt);
  free(size->loc);
}

/* Matrix-vector multiplication */
void mv_mul(const double *matrix, const double *vector, double *product,
    const size_info_t *size, int n_row)
{
  int i, j;
  double *whole_vector = (double *)malloc(size->n * sizeof(double));
  MPI_Allgatherv(vector, size->n_cpnt[size->rank], MPI_DOUBLE,
      whole_vector, size->n_cpnt, size->loc, MPI_DOUBLE, MPI_COMM_WORLD);
  for (i = 0; i < n_row; i++) {
    product[i] = 0;
    for (j = 0; j < size->n; j++)
      product[i] += matrix[i*size->n + j] * whole_vector[j];
  }
  free(whole_vector);
}

/* Compute the inner product of two vectors */
double inner_product(const double *v1, const double *v2, int len)
{
  double local_ip = 0, ip;
  for (int i = 0; i < len; i++)
    local_ip += v1[i] * v2[i];

  MPI_Allreduce(&local_ip, &ip, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return ip;
}

/* Compute the vector ax+b */
void axpy(double a, const double *x, const double *y, double *res, int len)
{
  for (int i = 0; i < len; i++)
    res[i] = a * x[i] + y[i];
}

/* Apply the conjugate gradient method to solve the linear system */
void conjugate_gradient(double *matrix, const double *y, double *x, double lambda, const size_info_t *size)
{
  int i;
  for (i = 0; i < size->n_cpnt[size->rank]; i++)
    matrix[i * size->n + size->loc[size->rank] + i] += lambda;

  double *r, *p, *w, se, new_se;
  r = (double *)malloc(size->n_cpnt[size->rank] * sizeof(double));
  p = (double *)malloc(size->n_cpnt[size->rank] * sizeof(double));
  w = (double *)malloc(size->n_cpnt[size->rank] * sizeof(double));

  mv_mul(matrix, x, w, size, size->n_cpnt[size->rank]);
  axpy(-1, w, y, r, size->n_cpnt[size->rank]);
  memcpy(p, r, size->n_cpnt[size->rank] * sizeof(double));

  se = inner_product(r, r, size->n_cpnt[size->rank]);
  if (size->rank == 0) {
    printf("Using CG method to solve the linear system...\n");
    printf("\tIter 0: Residual = %lf\n", sqrt(se));
  }
  i = 0;
  while (se > 1e-10) {
    double s;
    mv_mul(matrix, p, w, size, size->n_cpnt[size->rank]);
    s = se / inner_product(p, w, size->n_cpnt[size->rank]);
    axpy(s, p, x, x, size->n_cpnt[size->rank]);
    axpy(-s, w, r, r, size->n_cpnt[size->rank]);
    new_se = inner_product(r, r, size->n_cpnt[size->rank]);
    axpy(new_se / se, p, r, p, size->n_cpnt[size->rank]);
    se = new_se;
    if (size->rank == 0) printf("\tIter %d: Residual = %lf\n", ++i, sqrt(se));
  }

  free(r);
  free(p);
  free(w);
}

double gaussian_kernel(const double *x1, const double *x2, double s, int n_feature)
{
  double sum = 0;
  for (int i = 0; i < n_feature; i++)
    sum += (x1[i] - x2[i]) * (x1[i] - x2[i]);
  return exp(-sum / (2 * s * s));
}

void build_matrix(double **matrix,
    const double *training_data, const size_info_t *training_data_size,
    const double *test_data, const size_info_t *test_data_size,
    double s, int n_feature, int nprocs)
{
  MPI_Request send_req[nprocs-1];
  MPI_Request recv_req;
  const double *rcvd_data = training_data;
  double *buf, *old_buf;

  *matrix = (double *)malloc(sizeof(double) *
      training_data_size->n * test_data_size->n_cpnt[test_data_size->rank]);
  for (int k = 0; k < nprocs; k++) {
    if (k != 0) {
      MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
      rcvd_data = old_buf = buf;
    }

    /* Send the training data to the next process and receive data from the
     * previous process. */
    if (k < nprocs - 1) {
      int dest = (training_data_size->rank + k + 1) % nprocs;
      int orig = (training_data_size->rank + nprocs - k - 1) % nprocs;
      MPI_Isend(training_data, training_data_size->n_cpnt[training_data_size->rank] * n_feature,
          MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &send_req[k]);

      buf = (double *)malloc(sizeof(double) * training_data_size->n_cpnt[orig] * n_feature);
      MPI_Irecv(buf, training_data_size->n_cpnt[orig] * n_feature,
          MPI_DOUBLE, orig, 0, MPI_COMM_WORLD, &recv_req);
    }

    /* Compute one block of the coefficient matrix */
    int orig = (training_data_size->rank + nprocs - k) % nprocs;
    for (int i = 0; i < test_data_size->n_cpnt[test_data_size->rank]; i++) {
      for (int j = 0; j < training_data_size->n_cpnt[orig]; j++)
        (*matrix)[i * training_data_size->n + training_data_size->loc[orig] + j] =
          gaussian_kernel(test_data + i * n_feature, rcvd_data + j * n_feature, s, n_feature);
    }

    /* Release memory */
    if (k != 0) free(old_buf);
  }
  MPI_Waitall(nprocs-1, send_req, MPI_STATUSES_IGNORE);
}

/* Add the new data to the training set and the test set */
void append_data(int n_item, int n_feature, double **feature_training, double **label_traing,
    double **feature_test, double **label_test, int *n_training, int *n_test, double *tmp)
{
  double *random_numbers = (double *)malloc(n_item * sizeof(double));
  int i, n_training_new = 0;
  for (i = 0; i < n_item; i++) {
    random_numbers[i] = (double)rand() / RAND_MAX;
    if (random_numbers[i] < 0.7) n_training_new++;
  }

  *feature_training = (double *)realloc(*feature_training,
      (n_training_new + *n_training) * n_feature * sizeof(double));
  *label_traing = (double *)realloc(*label_traing,
      (n_training_new + *n_training) * sizeof(double));
  *feature_test = (double *)realloc(*feature_test,
      (n_item - n_training_new + *n_test) * n_feature * sizeof(double));
  *label_test = (double *)realloc(*label_test,
      (n_item - n_training_new + *n_test) * sizeof(double));

  for (int i = 0; i < n_item; i++) {
    if (random_numbers[i] < 0.7) {
      memcpy((*feature_training) + (*n_training) * n_feature,
          tmp + i * (n_feature + 1), n_feature * sizeof(double));
      (*label_traing)[*n_training] = tmp[i * (n_feature + 1) + n_feature];
      (*n_training)++;
    } else {
      memcpy((*feature_test) + (*n_test) * n_feature,
          tmp + i * (n_feature + 1), n_feature * sizeof(double));
      (*label_test)[*n_test] = tmp[i * (n_feature + 1) + n_feature];
      (*n_test)++;
    }
  }
  free(random_numbers);
}

/* Read data from file and distribute them to all processes */
void read_data(char *file_name, int n_feature,
    double **feature_training, double **label_training, double **feature_test, double ** label_test,
    size_info_t *training_data_size, size_info_t *test_data_size)
{
  const int max_sample = 100, max_req = 50;
  double *tmp = (double *)malloc((n_feature + 1) * max_sample * sizeof(double));
  int n_item = 0, dest = 0, rank, nprocs, n_training = 0, n_test = 0;

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
          append_data(n_item, n_feature, feature_training, label_training,
              feature_test, label_test, &n_training, &n_test, tmp);
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
        append_data(n_item, n_feature, feature_training, label_training,
            feature_test, label_test, &n_training, &n_test, tmp);
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
      append_data(n_item / (n_feature + 1), n_feature,
          feature_training, label_training, feature_test, label_test, &n_training, &n_test, tmp);
    }
  }

  free(tmp);
  create_size_info(n_training, training_data_size);
  create_size_info(n_test, test_data_size);
}

/* Normalize the training data to have mean value 0 and standard deviation 1 */
void normalize_data(double *training_data, const size_info_t *training_data_size,
    int n_feature, double *mean, double *std)
{
  int i, j;
  for (i = 0; i < training_data_size->n_cpnt[training_data_size->rank]; i++)
    for (j = 0; j < n_feature; j++)
      mean[j] += training_data[i * n_feature + j];
  MPI_Allreduce(MPI_IN_PLACE, mean, n_feature, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  for (j = 0; j < n_feature; j++)
    mean[j] /= training_data_size->n;

  for (i = 0; i < training_data_size->n_cpnt[training_data_size->rank]; i++)
    for (j = 0; j < n_feature; j++) {
      double diff = training_data[i * n_feature + j] - mean[j];
      std[j] += diff * diff;
    }
  MPI_Allreduce(MPI_IN_PLACE, std, n_feature, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  for (j = 0; j < n_feature; j++)
    std[j] = sqrt(std[j] / training_data_size->n);

  for (i = 0; i < training_data_size->n_cpnt[training_data_size->rank]; i++)
    for (j = 0; j < n_feature; j++)
      training_data[i * n_feature + j] =
        (training_data[i * n_feature + j] - mean[j]) / std[j];
}

/* Compute RMSE */
double test(const double *training_data, const size_info_t *training_data_size,
    const double *test_data, const double *test_data_label,
    const size_info_t *test_data_size, const double *mean, const double *std,
    const double *weight, double s, int n_feature, int nprocs)
{
  double *matrix;
  int i, j, n = test_data_size->n_cpnt[test_data_size->rank];
  double *test_data_copy = (double *)malloc(sizeof(double) * n * n_feature);

  memcpy(test_data_copy, test_data, sizeof(double) * n * n_feature);
  if (mean != NULL && std != NULL) {
    for (i = 0; i < n; i++)
      for (j = 0; j < n_feature; j++)
        test_data_copy[i * n_feature + j] =
          (test_data_copy[i * n_feature + j] - mean[j]) / std[j];
  } else if (mean != NULL) {
    for (i = 0; i < n; i++)
      for (j = 0; j < n_feature; j++)
        test_data_copy[i * n_feature + j] -= mean[j];
  }

  build_matrix(&matrix, training_data, training_data_size,
    test_data_copy, test_data_size, s, n_feature, nprocs);

  double *test_result = (double *)malloc(sizeof(double) * n);
  mv_mul(matrix, weight, test_result, training_data_size, n);
  free(matrix);

  double squared_err = 0;
  for (i = 0; i < n; i++) {
    double diff = test_data_label[i] - test_result[i];
    squared_err += diff * diff;
  }

  MPI_Allreduce(MPI_IN_PLACE, &squared_err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  return sqrt(squared_err / n);
}

int main(int argc, char *argv[])
{
  int nprocs, rank, n_feature;
  double *feature_training = NULL, *label_training = NULL;
  double *feature_test = NULL, *label_test = NULL;
  double *matrix, *weight, *mean, *std, rmse, s, lambda;
  size_info_t size_train, size_test;

  /* Read parameters */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (argc < 5) {
    if (rank == 0)
      fprintf(stderr, "Usage: %s <Number of Features> <File Name> <lambda> <s>\n", argv[0]);
    MPI_Finalize();
    return 1;
  }
  n_feature = atoi(argv[1]);
  lambda = atof(argv[3]);
  s = atof(argv[4]);

  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  /* Read data from file */
  srand(rank + time(NULL));
  read_data(argv[2], n_feature,
      &feature_training, &label_training, &feature_test, &label_test,
      &size_train, &size_test);

  if (rank == 0) {
    int r;
    printf("Training data size: %d\n\t", size_train.n);
    for (r = 0; r < nprocs; r++)
      printf("[%d] %d; ", r, size_train.n_cpnt[r]);
    printf("\n");
  }

  /* Normalize data */
  mean = (double *)calloc(n_feature, sizeof(double));
  std = (double *)calloc(n_feature, sizeof(double));
  normalize_data(feature_training, &size_train, n_feature, mean, std);

  /* Create the matrix and solve the linear system */
  weight = (double *)calloc(sizeof(double), size_train.n_cpnt[rank]);
  build_matrix(&matrix, feature_training, &size_train, feature_training, &size_train, s, n_feature, nprocs);
  conjugate_gradient(matrix, label_training, weight, lambda, &size_train);

  /* Compute the training error */
  rmse = test(feature_training, &size_train, feature_training, label_training, &size_train,
      NULL, NULL, weight, s, n_feature, nprocs);
  if (rank == 0)
    printf("Root mean squared error for training data: %lf\n", rmse);

  /* Compute the test error */
  if (rank == 0) {
    int r;
    printf("Test data size: %d\n\t", size_test.n);
    for (r = 0; r < nprocs; r++)
      printf("[%d] %d; ", r, size_test.n_cpnt[r]);
    printf("\n");
  }

  rmse = test(feature_training, &size_train, feature_test, label_test,
      &size_test, mean, std, weight, s, n_feature, nprocs);
  if (rank == 0)
    printf("Root mean squared error for test data: %lf\n", rmse);

  /* Clean up */
  free(feature_training);
  free(label_training);
  free(feature_test);
  free(label_test);
  free(matrix);
  free(weight);
  free(mean);
  free(std);
  destroy_size_info(&size_train);
  destroy_size_info(&size_test);

  MPI_Finalize();

  return EXIT_SUCCESS;
}
