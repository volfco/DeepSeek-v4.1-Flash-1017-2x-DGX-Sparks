// Stream from DRAM or L1 on the same host cores. End when the controlling pipe closes.
#define _GNU_SOURCE
#include <errno.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef uint64_t v2 __attribute__((vector_size(16)));
struct worker {
    _Alignas(64) atomic_uint_fast64_t bytes;
    atomic_bool ready;
    int cpu;
    size_t size;
    uint64_t checksum;
};
static atomic_bool stop;
static uint64_t timestamp(void) {
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (uint64_t)t.tv_sec * 1000000000 + t.tv_nsec;
}
static void *read_loop(void *arg) {
    struct worker *w = arg;
    cpu_set_t cores;
    CPU_ZERO(&cores);
    CPU_SET(w->cpu, &cores);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cores), &cores)) exit(2);
    v2 *data = aligned_alloc(64, w->size);
    if (!data) exit(3);
    size_t n = w->size / sizeof(v2);
    for (size_t i = 0; i < n; ++i) data[i] = (v2){i + 1, i + w->cpu};
    v2 a = {0}, b = {0}, c = {0}, d = {0};
    // Publish counters at the same byte interval in each arm.
    const size_t group = 1u << 18;  // four MiB
    size_t cursor = 0;
    atomic_store(&w->ready, 1);
    while (!atomic_load(&stop)) {
        size_t consumed = 0;
        while (consumed < group) {
            size_t length = n - cursor < group - consumed ? n - cursor : group - consumed;
            __asm__ volatile ("" ::: "memory");
            for (size_t i = cursor; i < cursor + length; i += 4) {
                a += data[i]; b += data[i+1]; c += data[i+2]; d += data[i+3];
            }
            cursor = (cursor + length) % n;
            consumed += length;
        }
        atomic_fetch_add_explicit(&w->bytes, group * sizeof(v2), memory_order_relaxed);
    }
    v2 total = a+b+c+d;
    w->checksum = total[0] + total[1];
    free(data);
    return NULL;
}
int main(int argc, char **argv) {
    if (argc != 3 || (strcmp(argv[1], "cache") && strcmp(argv[1], "stream"))) return 1;
    int n = atoi(argv[2]);
    if (n < 1 || n > 3) return 1;
    struct worker workers[3] = {0};
    pthread_t threads[3];
    for (int i=0; i<n; ++i) {
        workers[i].cpu = i;
        workers[i].size = !strcmp(argv[1], "cache") ? 16384 : 64u<<20;
        if (pthread_create(&threads[i], NULL, read_loop, &workers[i])) return 4;
    }
    for (int i=0; i<n; ++i) while (!atomic_load(&workers[i].ready)) usleep(1000);
    setvbuf(stdout, NULL, _IOLBF, 0);
    puts("utc_ns,bytes");
    struct pollfd ctrl = {.fd=STDIN_FILENO,.events=POLLIN|POLLHUP};
    for (int round=0; round<600; ++round) {
        uint64_t bytes = 0;
        for (int i=0; i<n; ++i) bytes += atomic_load(&workers[i].bytes);
        printf("%llu,%llu\n",(unsigned long long)timestamp(),(unsigned long long)bytes);
        int rc = poll(&ctrl, 1, 500);
        if (rc > 0 || (rc < 0 && errno != EINTR)) break;
    }
    atomic_store(&stop, 1);
    for (int i=0; i<n; ++i) {
        pthread_join(threads[i], NULL);
        fprintf(stderr,"cpu=%d size=%zu checksum=%llu\n", workers[i].cpu, workers[i].size,
                (unsigned long long)workers[i].checksum);
    }
    return 0;
}
