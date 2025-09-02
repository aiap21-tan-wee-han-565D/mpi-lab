#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define MAX_NUM_ENTRIES 100
#define MAX_NUM_ENTRIES_PER_ROUND 5
#define WORD(i) i
#define ACK (MAX_NUM_ENTRIES+2)
#define SYNC (MAX_NUM_ENTRIES+3)
#define SAFE (MAX_NUM_ENTRIES+4)

typedef struct {
  int count;
  char *entries[MAX_NUM_ENTRIES];

  /* In the send list, this variable stores whether the acknowledgement has
   * been received. In the receive list, this variable stores the tag of the
   * message. */
  int aux_var[MAX_NUM_ENTRIES];
} list_t;

/* Receive a message from another process. */
int receive_message(MPI_Request req[], int *n_req,
    list_t *send_list, list_t *recv_list, MPI_Status *status, int safe[])
{
  if (status->MPI_TAG == ACK) {
    /* An acknowledgement is received. */
    int idx;
    MPI_Recv(&idx, 1, MPI_INT, status->MPI_SOURCE,
        status->MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    send_list->aux_var[idx] = 1;
    free(send_list->entries[idx]);
  } else if (status->MPI_TAG == SYNC) {
    /* A message requesting synchronization is received. */
    MPI_Recv(NULL, 0, MPI_CHAR, status->MPI_SOURCE,
        status->MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return 1;
  } else if (status->MPI_TAG == SAFE) {
    /* A message "I'm safe" is received. */
    MPI_Recv(NULL, 0, MPI_CHAR, status->MPI_SOURCE,
        status->MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    safe[status->MPI_SOURCE] = 1;
    return 1;
  } else {
    /* A word is received. */
    int count;
    char *str;

    MPI_Get_count(status, MPI_CHAR, &count);
    str = (char *)malloc(sizeof(char) * count);
    MPI_Recv(str, count, MPI_CHAR, status->MPI_SOURCE,
        status->MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    /* Add the word to the list. */
    recv_list->aux_var[recv_list->count] = status->MPI_TAG;
    recv_list->entries[recv_list->count] = str;
    
    /* Send an acknowlegment. */
    MPI_Isend(&(recv_list->aux_var[recv_list->count]), 1, MPI_INT,
        status->MPI_SOURCE, ACK, MPI_COMM_WORLD, &req[*n_req]);
    (*n_req)++;
    recv_list->count++;
  }
  return 0;
}

/* Check if all elements in an array are true. */
int all(int count, int arr[])
{
  int i, all = 1;
  for (i = 0; all && i < count; i++) all &= arr[i];
  return all;
}

/* Implementation of the alpha synchronizer. */
void synchronize(int *n_req, MPI_Request req[],
    int nprocs, int rank, list_t *send_list, list_t *recv_list, int safe[])
{
  int all_safe, all_acked;
  MPI_Status status;

  safe[rank] = 1;
  all_acked = all(send_list->count, send_list->aux_var);

  /* Get messages until the process is safe. */
  while (!all_acked) {
    MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    receive_message(req, n_req, send_list, recv_list, &status, safe);
    all_acked = all(send_list->count, send_list->aux_var);
  }

  /* Send "I'm safe" to all other processes. */
  for (int i = 0; i < nprocs; i++)
    if (i != rank) {
      MPI_Isend(NULL, 0, MPI_CHAR, i, SAFE, MPI_COMM_WORLD, &req[*n_req]);
      (*n_req)++;
    }

  all_safe = all(nprocs, safe);

  /* Get messages until all processes are safe. */
  while (!all_safe) {
    MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    receive_message(req, n_req, send_list, recv_list, &status, safe);
    all_safe = all(nprocs, safe);
  }

  MPI_Waitall(*n_req, req, MPI_STATUSES_IGNORE);
}

/* Determine which process should the string be sent to. */
int find_dest(int nprocs, char *str, char *first_word[])
{
  int i;
  for (i = 0; i < nprocs; i++)
    if (strcmp(str, first_word[i]) < 0)
      return i-1;
  return i-1;
}

/* Add a word to the list or send the word to the destiniation process */
int add_word(MPI_Request req[], int *n_req, int rank, int nprocs,
    char *str, int n_char, int idx,
    list_t *send_list, list_t *recv_list, char *first_word[])
{
  int dest, flag = 1;
  if (str[n_char-1] == '\n') str[--n_char] = '\0';
  dest = find_dest(nprocs, str, first_word);

  if (dest != rank) { // Send to the corresponding process
    MPI_Isend(str, n_char+1, MPI_CHAR, dest,
        WORD(send_list->count), MPI_COMM_WORLD, &req[(*n_req)++]);
    send_list->entries[send_list->count] = str;
    send_list->aux_var[send_list->count] = 0;
    send_list->count++;
  } else { // Add the word to the list
    recv_list->entries[recv_list->count] = str;
    recv_list->aux_var[recv_list->count] = idx;
    recv_list->count++;
  }
  return flag;
}

/**
 * Check incoming messages.
 */
int check_incoming_msg(MPI_Request req[], int *n_req,
    list_t *send_list, list_t *recv_list, int safe[])
{
  int start_synchronizer = 0, flag = 1;
  while (flag) {
    MPI_Status status;
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
    if (flag) {
      start_synchronizer |= receive_message(req, n_req,
          send_list, recv_list, &status, safe);
    }
  }
  return start_synchronizer;
}

/*
 * Read words from the file and send them to corresponding processes. Return
 * the number of requests.
 */
int add_entries(MPI_Request req[], int rank, int nprocs,
    list_t *send_list, list_t *recv_list, FILE *fp, int safe[], char *first_word[])
{
  int i, n_req = 0;
  for (i = 0; i < MAX_NUM_ENTRIES_PER_ROUND; i++) {
    char *str = NULL;
    size_t n = 0, n_char;
    if ((n_char = getline(&str, &n, fp)) != -1) { // Read from file
      add_word(req, &n_req, rank, nprocs, str, n_char, i, send_list, recv_list, first_word);

      int start_synchronizer = check_incoming_msg(req, &n_req, send_list, recv_list, safe);
      if (start_synchronizer)
        printf("Rank %d: Received request for starting a synchronizer!\n", rank);
      if (start_synchronizer) return n_req;
    } else
      break;
  }

  if (!feof(fp)) {
    /*
     * Initiate the synchronizer. This part is reached only if the maximum number
     * of entries per round is reached.
     */
    for (i = 0; i < nprocs; i++)
      if (i != rank)
        MPI_Isend(NULL, 0, MPI_CHAR, i, SYNC, MPI_COMM_WORLD, &req[n_req++]);
  }

  return n_req;
}

void destroy_list(list_t *list)
{
  int i;
  for (i = 0; i < list->count; i++)
    if (list->entries[i] != NULL) free(list->entries[i]);
  list->count = 0;
}

/* Combine the strings from the receive list into a single string. */
char *join_strings(list_t *recv_list, int first, int last, int *size)
{
  int i;
  *size = 0;
  for (i = first; i < last; i++)
    *size += strlen(recv_list->entries[i]) + 1;

  char *str = (char *)malloc(*size * sizeof(char));
  int l = 0;
  for (i = first; i < last; i++) {
    int len = strlen(recv_list->entries[i]) + 1;
    memcpy(str + l, recv_list->entries[i], len);
    l += len;
  }
  
  return str;
}

/* Decompose the received string into words and add them to the list. */
void separate_words(const char *str, int len, list_t *list)
{
  int start = 0;
  for (int i = 0; i < len; i++)
    if (str[i] == '\0') {
      list->entries[list->count] = (char *)malloc((i - start + 1) * sizeof(char));
      memcpy(list->entries[list->count], str + start, i - start + 1);
      start = i + 1;
      list->count++;
    }
}

/*
 * Redistribute the words such that each process has roughly the same number of
 * words.
 */
void redistribute_words(list_t *recv_list, int rank, int nprocs)
{
  /* Gather the counts of words from each process. */
  int *count = (int *)malloc((nprocs+1) * sizeof(int));
  count[0] = 0;
  MPI_Allgather(&recv_list->count, 1, MPI_INT, count+1, 1, MPI_INT, MPI_COMM_WORLD);
  for (int i = 2; i <= nprocs; i++) count[i] += count[i-1];

  /* Create a new list to hold the redistributed words. */
  list_t new_list;
  new_list.count = 0;

  /* Distribute the words among processes. */
  int quotient = count[nprocs] / nprocs, remainder = count[nprocs] % nprocs;
  int start = 0, end;
  int *recv = (int *)calloc(nprocs, sizeof(int)); // indicates if we need to receive words from process i
  char **str = (char **)malloc(nprocs * sizeof(char *)); // Buffer for words to be sent
  MPI_Request *send_req = (MPI_Request *)malloc(nprocs * sizeof(MPI_Request)); // requests for sending words

  for (int i = 0; i < nprocs; i++) {
    send_req[i] = MPI_REQUEST_NULL;
    str[i] = NULL;
    end = start + quotient + (i < remainder ? 1 : 0);
    if (i != rank) {
      /* Check if we need to send any words to process i. If so, send the words. */
      int first = start > count[rank] ? start : count[rank];
      int last = end < count[rank+1] ? end : count[rank+1];
      if (last > first) {
        int size;
        str[i] = join_strings(recv_list, first - count[rank], last - count[rank], &size);
        MPI_Isend(str[i], size, MPI_CHAR, i, 0, MPI_COMM_WORLD, &send_req[i]);
      }
    } else {
      /* Copy existing words */
      int first = start > count[i] ? start : count[i];
      int last = end < count[i+1] ? end : count[i+1];
      for (int j = first; j < last; j++) {
        new_list.entries[new_list.count++] = recv_list->entries[j-count[i]];
        recv_list->entries[j-count[i]] = NULL;
      }

      /* Check if we need to receive any words from other processes. */
      for (int j = 0; j < nprocs; j++) {
        if (j != rank) {
          int first = start > count[j] ? start : count[j];
          int last = end < count[j+1] ? end : count[j+1];
          if (last > first) recv[j] = 1;
        }
      }
    }
    start = end;
  }

  /* Receive messages from other processes. */
  for (int j = 0; j < nprocs; j++) {
    if (recv[j]) {
      char *str;
      int count;
      MPI_Status status;
      MPI_Probe(j, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      MPI_Get_count(&status, MPI_CHAR, &count);
      str = (char *)malloc(sizeof(char) * count);
      MPI_Recv(str, count, MPI_CHAR, j, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      separate_words(str, count, &new_list);
      free(str);
    }
  }

  /* Clean up */
  free(count);
  MPI_Waitall(nprocs, send_req, MPI_STATUSES_IGNORE);
  for (int i = 0; i < nprocs; i++)
    if (str[i] != NULL) free(str[i]);
  free(send_req);

  /* Copy the new list back to the receive list. */
  destroy_list(recv_list);
  recv_list->count = new_list.count;
  for (int i = 0; i < new_list.count; i++)
    recv_list->entries[i] = new_list.entries[i];
}

/* Write the list of words into a file. */
void output_list(const list_t *recv_list, FILE *fp)
{
  int i;
  for (i = 0; i < recv_list->count; i++)
    fprintf(fp, "%s\n", recv_list->entries[i]);
}

int strcmp_for_qsort(const void *str1, const void *str2)
{
  return strcmp(*(char **)(str1), *(char **)(str2));
}

int all_eof(FILE *fp)
{
  int eof = feof(fp);
  MPI_Allreduce(MPI_IN_PLACE, &eof, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
  return eof;
}

void gather_first_words(char *first_word[], int nprocs, int rank)
{
  MPI_Request *req = (MPI_Request *)malloc(nprocs * sizeof(MPI_Request));
  for (int i = 0; i < nprocs; i++) {
    if (i != rank)
      MPI_Isend(first_word[rank], strlen(first_word[rank]) + 1, MPI_CHAR, i, 0, MPI_COMM_WORLD, &req[i]);
    else
      req[i] = MPI_REQUEST_NULL;
  }

  MPI_Status status;
  int count;
  for (int i = 0; i < nprocs; i++) {
    if (i != rank) {
      MPI_Probe(i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      MPI_Get_count(&status, MPI_CHAR, &count);
      first_word[i] = (char *)malloc(sizeof(char) * count);
      MPI_Recv(first_word[i], count, MPI_CHAR, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
  MPI_Waitall(nprocs, req, MPI_STATUSES_IGNORE);

  free(req);
}

int main(int argc, char *argv[])
{
  FILE *fp;
  char file_name[100], **first_word;
  int nprocs, rank, n_req, i, *safe;
  MPI_Request req[4 * MAX_NUM_ENTRIES];
  list_t send_list, recv_list;
  typedef int (*cmp_t)(const void *, const void *);

  /* Initialization */
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  send_list.count = recv_list.count = 0;

  /* Task distribution */
  first_word = (char **)malloc(nprocs * sizeof(char *));
  int quotient = 26 / nprocs, remainder = 26 % nprocs;
  for (i = 0; i < remainder; i++) {
    first_word[i] = (char *)calloc(2, sizeof(char));
    first_word[i][0] = 'a' + i * (quotient + 1);
  }

  for (; i < nprocs; i++) {
    first_word[i] = (char *)calloc(2, sizeof(char));
    first_word[i][0] = 'a' + i * quotient + remainder;
  }

  /* Open a text file */
  sprintf(file_name, "words_%d.txt", rank);
  fp = fopen(file_name, "r");

  do {
    printf("Rank %d: Store words from '%s'...\n", rank, first_word[rank]);

    /* Read words from the file */
    printf("Rank %d: Read words from file...\n", rank);
    send_list.count = 0;
    safe = (int *)calloc(nprocs, sizeof(int));
    n_req = add_entries(req, rank, nprocs, &send_list, &recv_list, fp, safe, first_word);
    printf("Rank %d: Waiting for synchronization...\n", rank);

    /* Synchronization */
    printf("Rank %d: Start synchronizer...\n", rank);
    synchronize(&n_req, req, nprocs, rank, &send_list, &recv_list, safe);
    printf("Rank %d: Synchronization completed.\n", rank);
    free(safe);

    /* Sort and redistribute words */
    qsort(recv_list.entries, recv_list.count, sizeof(char *), &strcmp_for_qsort);
    printf("Rank %d: The list contains %d words before redistribution.\n", rank, recv_list.count);
    redistribute_words(&recv_list, rank, nprocs);
    printf("Rank %d: The list contains %d words after redistribution.\n", rank, recv_list.count);
    qsort(recv_list.entries, recv_list.count, sizeof(char *), &strcmp_for_qsort);
    if (rank == 0)
      for (i = 1; i < nprocs; i++) free(first_word[i]);
    else {
      for (i = 0; i < nprocs; i++) free(first_word[i]);
      first_word[rank] = strdup(recv_list.entries[0]);
    }
    gather_first_words(first_word, nprocs, rank);
  } while (!all_eof(fp));

  /* Output the list */
  sprintf(file_name, "list_%d.txt", rank);
  fp = fopen(file_name, "w");
  output_list(&recv_list, fp);

  /* Clean up */
  fclose(fp);
  destroy_list(&recv_list);
  for (i = 0; i < nprocs; i++)
    free(first_word[i]);
  free(first_word);

  MPI_Finalize();

  return EXIT_SUCCESS;
}
