#pragma once
#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <process.h>
#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#define PROT_READ  1
#define MAP_SHARED 1
#define MAP_FAILED ((void*)-1)

static inline void* win_mmap_fd(void* addr, size_t length, int prot, int flags, int fd, uint64_t offset) {
    (void)addr; (void)prot; (void)flags;
    HANDLE hFile = (HANDLE)_get_osfhandle(fd);
    if (hFile == INVALID_HANDLE_VALUE) return MAP_FAILED;
    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMap) return MAP_FAILED;
    DWORD off_hi = (DWORD)(offset >> 32);
    DWORD off_lo = (DWORD)(offset & 0xFFFFFFFFu);
    void* ptr = MapViewOfFile(hMap, FILE_MAP_READ, off_hi, off_lo, length);
    CloseHandle(hMap);
    return ptr ? ptr : MAP_FAILED;
}
#define mmap win_mmap_fd

// Use 64-bit stat on Windows for >4GB unified model files (6.74 GiB bonsai2-27b.npubin)
#define stat _stat64
#define fstat _fstat64

static inline int win_open_binary(const char* path, int flags) {
    return _open(path, flags | _O_BINARY);
}
#define open win_open_binary
#define close _close

typedef HANDLE sem_t;
static inline int sem_init(sem_t* sem, int pshared, unsigned int value) {
    (void)pshared;
    *sem = CreateSemaphoreA(NULL, (LONG)value, 32767, NULL);
    return (*sem != NULL) ? 0 : -1;
}
static inline int sem_wait(sem_t* sem) {
    return (WaitForSingleObject(*sem, INFINITE) == WAIT_OBJECT_0) ? 0 : -1;
}
static inline int sem_post(sem_t* sem) {
    return ReleaseSemaphore(*sem, 1, NULL) ? 0 : -1;
}

typedef HANDLE pthread_t;
typedef struct { void* (*fn)(void*); void* arg; } win_thread_ctx_t;
static unsigned __stdcall win_thread_trampoline(void* p) {
    win_thread_ctx_t ctx = *(win_thread_ctx_t*)p;
    free(p);
    ctx.fn(ctx.arg);
    return 0;
}
static inline int pthread_create(pthread_t* t, const void* attr, void* (*fn)(void*), void* arg) {
    (void)attr;
    win_thread_ctx_t* ctx = (win_thread_ctx_t*)malloc(sizeof(win_thread_ctx_t));
    ctx->fn = fn; ctx->arg = arg;
    uintptr_t h = _beginthreadex(NULL, 0, win_thread_trampoline, ctx, 0, NULL);
    if (!h) { free(ctx); return -1; }
    *t = (HANDLE)h;
    return 0;
}

#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif
static inline int win_clock_gettime(int clk_id, struct timespec* tp) {
    (void)clk_id;
    static LARGE_INTEGER freq = {0};
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER cnt;
    QueryPerformanceCounter(&cnt);
    tp->tv_sec = cnt.QuadPart / freq.QuadPart;
    tp->tv_nsec = (long)(((cnt.QuadPart % freq.QuadPart) * 1000000000ULL) / freq.QuadPart);
    return 0;
}
#define clock_gettime win_clock_gettime

static inline int setenv(const char* name, const char* value, int overwrite) {
    (void)overwrite;
    return _putenv_s(name, value);
}

#endif // _WIN32
