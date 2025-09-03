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
  /*
entries[]: array of char* to the actual word strings.
In send_list: outgoing buffers to keep alive until ACK → then free.
In recv_list: words you own (received or local) → later sorted/printed/freed.

aux_var[]: per-slot integer metadata.
In send_list: ack flags (0/1) used to detect when all sends are acknowledged.
In recv_list: stores the message id/tag (or local idx) and doubles as the stable buffer for the nonblocking ACK send payload.*/
  int count;
  char *entries[MAX_NUM_ENTRIES];

  int aux_var[MAX_NUM_ENTRIES];
} list_t;

/* Receive a message from another process. 
MPI_Request req[] — This rank’s array where we store any new nonblocking sends we initiate here (e.g., ACKs).
int *n_req — Counter (by pointer) of how many requests we’ve appended into req[] so far on this rank. We write the next MPI_Request at req[*n_req] and then do (*n_req)++.
list_t *send_list — Bookkeeping for words this rank sent earlier this round.
send_list->aux_var[i] = 0/1 → whether the i-th send has been ACKed.
send_list->entries[i] → pointer to the sent buffer (freed on ACK).
list_t *recv_list — Where we store words this rank owns locally (received from others or self).
MPI_Status *status — Metadata produced by a prior MPI_Probe/MPI_Iprobe (contains MPI_SOURCE and MPI_TAG) for the message we’re about to consume.
int safe[] — Per-rank flags used by the α-synchronizer; safe[src]=1 when we receive a SAFE control message from src.
*/
int receive_message(MPI_Request req[], int *n_req,
    list_t *send_list, list_t *recv_list, MPI_Status *status, int safe[])
{
  if (status->MPI_TAG == ACK) {
    /* An acknowledgement is received. */
    int idx;
    //idx is WORD(send_list->count) from ISend
    // if target node receives an ack, means the idx is its own send list idx meaning send_list->entries[idx] is being acknolwedged i.e., sendlist->aux_var[idx] = 1
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
  all_acked = all(send_list->count, send_list->aux_var); // ) returns the logical AND of the first count integers in arr (treating non-zero as true)

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

/* Add a word to the list or send the word to the destination process */
/*Given a freshly read line str of length n_char, decide which MPI rank should own it.
If the destination is a different rank, post a non-blocking send (MPI_Isend) and track that send so you can free the buffer when an ACK arrives.
If the destination is this rank, just append the word locally to your receive list.
It always returns 1 (the flag), which in this snippet isn’t used for control flow.

MPI_Request req[]
The caller’s array where this function will append new nonblocking send requests (MPI_Isend). Each new request handle is written into the next free slot.
int *n_req: A counter (by pointer) for how many requests have been appended to req[] so far by this rank in this round. The call uses &req[(*n_req)++], which: stores the new MPI_Request at req[*n_req], then increments *n_req by 1.
int rank:  This process’s MPI rank. Used to tell whether we’re the destination and thus should keep the word locally. int nprocs Total number of MPI processes. Passed to find_dest (via first_word) to figure out which rank owns the word.
char *str:  The line read from the file (a dynamically allocated, NUL-terminated C string). Ownership rules: If we send it: keep the pointer in send_list so we can free it later when we get an ACK (the code frees it in receive_message upon ACK). If we keep it locally: store the pointer in recv_list (freed later when lists are destroyed/redistributed).
int n_char:  The length returned by getline (bytes read). It usually includes the trailing '\n' if present. The function trims that newline (if (str[n_char-1] == '\n') ...) and then uses n_char+1 in MPI_Isend so the NUL terminator is sent too.
int idx:  The index of this word within the current read batch (0…MAX_NUM_ENTRIES_PER_ROUND-1). If the word stays local, the code stashes idx into recv_list->aux_var[...]. (For received-from-remote words, aux_var stores the sender’s tag instead.) There’s no ACK for local words; storing idx just keeps a consistent “origin index” field.
list_t *send_list:  Metadata for words we sent this round:
entries[i] holds the pointer we sent,
aux_var[i] is 0/1 for “ACK not received / received”.
The index i is also encoded in the MPI tag for the send (via WORD(send_list->count)), so the receiver can return it in an ACK and we know which slot to mark as acked.
list_t *recv_list : Where we accumulate locally owned words (either self-kept or received from others).
char *first_word[] : An array (length = nprocs) of threshold strings that define the alphabetical bins per rank. find_dest(nprocs, str, first_word) uses this to pick the destination rank for str.*/
int add_word(MPI_Request req[], int *n_req, int rank, int nprocs,
    char *str, int n_char, int idx,
    list_t *send_list, list_t *recv_list, char *first_word[])
{
  int dest, flag = 1;
  if (str[n_char-1] == '\n') str[--n_char] = '\0';
  dest = find_dest(nprocs, str, first_word);

  if (dest != rank) { // Send to the corresponding process
    // &req[(*n_req)++] means Take the address of req[k] (so MPI can write into it), where k is the current value of *n_req. 
    // After using that index, increment *n_req by 1 (post-increment), so the next request will go into the next slot.
    /*str: pointer to the send buffer (a C string for one word).
n_char+1: number of elements to send. getline gave you n_char bytes read (usually ends with '\n'). The code sends +1 so the NUL terminator '\0' is transmitted too (receiver can treat it as a proper C string).
MPI_CHAR: datatype for each element in the buffer.
dest: destination rank.
WORD(send_list->count): the tag. Here WORD(i) is just i, so the tag encodes the index of this send in the sender’s send_list. The receiver stores this tag and sends it back in an ACK so the sender knows which entry got acknowledged.
MPI_COMM_WORLD: communicator.
&req[(*n_req)++]: where to store the MPI request handle for this nonblocking send, then post-increment *n_req.
The request handle lands in req[*n_req].
After storing it, *n_req is incremented by 1.*/
    MPI_Isend(str, n_char+1, MPI_CHAR, dest,
        WORD(send_list->count), MPI_COMM_WORLD, &req[(*n_req)++]);
/*You stash the pointer you just sent in send_list->entries[...] and mark aux_var[...] = 0 (meaning “not yet acknowledged”).
This is critical: because it’s an Isend, the buffer must remain valid until the request completes. 
Keeping the pointer lets you free(...) it on ACK (your receive_message does exactly that).*/
    send_list->entries[send_list->count] = str;
    send_list->aux_var[send_list->count] = 0;
    send_list->count++;
  } else { // Add the word to the list because it is your own word
    recv_list->entries[recv_list->count] = str;
    recv_list->aux_var[recv_list->count] = idx;
    recv_list->count++;
  }
  return flag;
}

/**
 * It drains all currently pending MPI messages for this rank without blocking.

Uses MPI_Iprobe in a loop:
flag is set to 1 if at least one message is waiting (from any source / any tag), else 0.
When flag==1, status is filled with the metadata (source, tag, count-able later).
For each pending message, it calls receive_message(...), which:
If it’s a WORD: does a matching MPI_Recv, appends the word to recv_list, and sends an ACK with MPI_Isend (so a new request gets appended to req[], and *n_req increments).
If it’s an ACK: marks the corresponding send as acknowledged in send_list and frees the sent buffer.
If it’s SYNC or SAFE: consumes the control message; for SAFE, it also sets safe[src]=1. In both cases it returns 1 to signal “start/participate in synchronizer.”
start_synchronizer OR-accumulates those return values so if any control message (SYNC/SAFE) was seen during this drain, the function returns 1; otherwise 0.
Net effect: after check_incoming_msg returns, your process has:
Responded to all waiting messages (acked words, recorded acks, absorbed control msgs),
Possibly queued some ACK sends (tracked in req[], counted by *n_req),
Learned whether it should stop its current work and enter the synchronization phase (return 1).
 */
int check_incoming_msg(MPI_Request req[], int *n_req,
    list_t *send_list, list_t *recv_list, int safe[])
{
  int start_synchronizer = 0, flag = 1;
  while (flag) {
    MPI_Status status;
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status); // flag gets modifed here. flag = 1 → at least one pending message (from any source/tag). flag = 0 → no pending messages right now.
    if (flag) {
      start_synchronizer |= receive_message(req, n_req,
          send_list, recv_list, &status, safe);
    }
  }
  return start_synchronizer;
}

/*
 * Read words from the file and send them to corresponding processes. Return the number of requests.
 * add_entries reads up to MAX_NUM_ENTRIES_PER_ROUND lines (“words”) from a file and, for each line, hands the work off to the right MPI process (non-blocking).
 *  While it’s doing that, it also checks if any peer has asked to “synchronize” the round. If a sync request is seen, 
 * it stops early and returns the number of in-flight MPI requests so the caller can MPI_Wait* on them. 
 * If it finishes the per-round budget without hitting EOF, it proactively asks everyone else to synchronize by sending a zero-byte “SYNC” control message.
 * 
 * Concretely, across the whole program there are three ways you’ll end up starting the synchronizer:
 * You reached the per-round cap (read 5 lines by default) and not EOF
 * → you proactively broadcast SYNC to everyone at the end of the loop body in add_entries and then return to run synchronize(...).
 * A peer reached the cap first and sent SYNC while you were still reading
 * → your immediate check_incoming_msg detects it, sets start_synchronizer=1, prints the message, returns early from add_entries, and you call synchronize(...).
 * A peer already entered/finished synchronization and sent SAFE
 * → same as above: you detect SAFE in check_incoming_msg, set start_synchronizer=1, and return to start your synchronizer.
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

  /* Task distribution 
  *and first_word[i][0] = 'a' + start[i].

Concrete examples

nprocs = 4
quotient=6, remainder=2 → sizes: 7, 7, 6, 6
Starts (as letters):

i=0: 'a' + 0*(7) = 'a'

i=1: 'a' + 1*(7) = 'h'

i=2: 'a' + 2*6 + 2 = 'o'

i=3: 'a' + 3*6 + 2 = 'u'
Bins: [a..g], [h..n], [o..t], [u..z].*/
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
