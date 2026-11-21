// Direct random reads at a steady offered depth, limited to allocated ranges of an existing file.
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <linux/aio_abi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

enum { max_depth = 1024, max_extents = 2048, block = 4096 };
struct extent { uint64_t first, cumulative; };
static struct extent spans[max_extents];
static int span_count;
static uint64_t random_state = 817231;
static double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}
static void die(const char *s) { perror(s); exit(2); }
static void collect(int fd) {
    struct stat st;
    if (fstat(fd, &st)) die("stat");
    off_t at = 0;
    uint64_t blocks = 0;
    while (at < st.st_size && span_count < max_extents) {
        off_t data = lseek(fd, at, SEEK_DATA);
        if (data < 0 && errno == ENXIO) break;
        if (data < 0) die("SEEK_DATA");
        at = lseek(fd, data, SEEK_HOLE);
        if (at <= data) die("SEEK_HOLE");
        uint64_t lo = (data+block-1)/block, hi = at/block;
        if (hi > lo) {
            blocks += hi-lo;
            spans[span_count++] = (struct extent){lo, blocks};
        }
    }
    if (!blocks) { fputs("No allocated input blocks\n",stderr); exit(2); }
}
static uint64_t next_offset(void) {
    random_state ^= random_state >> 12; random_state ^= random_state << 25; random_state ^= random_state >> 27;
    uint64_t q = (random_state * UINT64_C(2685821657736338717)) % spans[span_count-1].cumulative;
    int lo = 0, hi = span_count-1;
    while (lo < hi) {
        int mid = (lo+hi)/2;
        if (q >= spans[mid].cumulative) lo = mid+1; else hi = mid;
    }
    return block*(spans[lo].first+q-(lo ? spans[lo-1].cumulative : 0));
}
static void measure(int fd, int depth) {
    struct iocb jobs[max_depth] = {0}, *ready[max_depth];
    struct io_event completed[max_depth];
    unsigned char *buffers = aligned_alloc(block, (size_t)block*depth);
    if (!buffers) die("allocation");
    aio_context_t ctx = 0;
    if (syscall(SYS_io_setup, depth, &ctx)) die("io_setup");
    for (int i=0; i<depth; ++i) {
        jobs[i].aio_data=i; jobs[i].aio_fildes=fd; jobs[i].aio_lio_opcode=IOCB_CMD_PREAD;
        jobs[i].aio_buf=(uintptr_t)(buffers+(size_t)i*block); jobs[i].aio_nbytes=block;
        jobs[i].aio_offset=next_offset(); ready[i]=&jobs[i];
    }
    int slots=depth, outstanding=0;
    uint64_t operations=0;
    double start=seconds();
    while (seconds()-start < 4 || outstanding) {
        if (seconds()-start < 4 && slots) {
            long n=syscall(SYS_io_submit,ctx,slots,ready);
            if (n<0 && errno!=EAGAIN && errno!=EINTR) die("io_submit");
            if (n>0) {
                slots-=(int)n; outstanding+=(int)n;
                memmove(ready,ready+n,(size_t)slots*sizeof(*ready));
            }
        }
        if (!outstanding) continue;
        long n=syscall(SYS_io_getevents,ctx,1,outstanding,completed,NULL);
        if (n<0 && errno==EINTR) continue;
        if (n<0) die("io_getevents");
        for (int j=0;j<n;++j) {
            if (completed[j].res!=block || completed[j].res2 || completed[j].data>=(uint64_t)depth) {
                fprintf(stderr,"Invalid read completion: %lld\n",(long long)completed[j].res);exit(3);
            }
            struct iocb *job=&jobs[completed[j].data];
            job->aio_offset=next_offset(); ready[slots++]=job;
        }
        outstanding-=(int)n; operations+=(uint64_t)n;
    }
    double elapsed=seconds()-start;
    printf("%d,%d,%llu,%.6f,%.2f,%.6f\n",depth,block,(unsigned long long)operations,elapsed,
           operations/elapsed,operations*block/elapsed/1e9);
    if (syscall(SYS_io_destroy,ctx)) die("io_destroy");
    free(buffers);
}
int main(int argc,char **argv) {
    if (argc!=2) return 1;
    int fd=open(argv[1],O_RDONLY|O_DIRECT|O_CLOEXEC);
    if (fd<0) die("open existing checkpoint");
    collect(fd);
    puts("offered_depth,read_bytes,completed,seconds,IOPS,GB_s");
    const int depths[]={64,256,1024};
    for (int i=0;i<3;++i) measure(fd,depths[i]);
    close(fd);
    return 0;
}
