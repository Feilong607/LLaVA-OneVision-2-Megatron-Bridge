// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// cumem_hook: driver-level ledger of CUDA VMM / multicast allocations, via CUPTI driver-API callbacks.
//
// Why: on the merged48 production run, per-GPU memory grows in steps on every pod's local GPU0/GPU3 only,
// outside torch's allocator and outside any process's NVML account (see OV2-QWEN35-BRINGUP §13.22). Memory
// that no process owns can only come from cuMem*/cuMulticast* calls (multicast physical pages, imported
// fabric memory). NCCL reaches those through cuGetProcAddress function pointers, so an LD_PRELOAD of the
// symbols would not see them; CUPTI's driver-API callback domain intercepts every entry point regardless of
// how the caller obtained it. This library subscribes to the ~14 relevant driver calls and appends one line
// per call (EXIT site, with return code, sizes, handles, device, and a raw backtrace) to
//   $CUMEM_HOOK_DIR/cumem_<host>_<pid>.log          (default dir: /tmp)
// and dumps /proc/self/maps once to cumem_<host>_<pid>.maps so the backtrace can be symbolized offline
// (cumem_ledger_report.py). Activation: LD_PRELOAD this .so with CUMEM_HOOK=1; it arms itself only inside
// processes whose executable name starts with "python" (torchrun ranks), so bash/nvidia-smi/tee in the same
// environment are untouched. Any CUPTI failure is logged once to stderr and the process runs unhooked.
//
// Build (inside the training image; needs cuda.h and cupti.h at compile time only -- libcupti is dlopen'ed at
// run time by absolute path, CUMEM_HOOK_CUPTI overrides the search list): see build.sh, or
//   gcc -O2 -fPIC -shared -o libcumemhook.so cumem_hook.c -I/usr/local/cuda/include -I/usr/local/cuda/extras/CUPTI/include -ldl -lpthread
//
// Ledger line format (space separated key=value):
//   t=<epoch.us> lt=<HH:MM:SS.us> pid= tid= api=<name> site=exit ret=<CUresult> size= dev= handle=0x..
//   mc=0x.. ptr=0x.. off= flags= bt=0xa,0xb,...   (fields absent when not applicable)
// Overhead: one callback per intercepted driver call (rare, allocation-time only); nothing on kernel launch.

#define _GNU_SOURCE
#include <cuda.h>
#include <cupti.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define BT_DEPTH 12

// ---- parameter structs: CUPTI passes the driver call's arguments as a struct in declaration order ----
// (identical to the definitions in generated_cuda_meta.h; declared here so the build needs no extra header)
typedef struct { CUmemGenericAllocationHandle *handle; size_t size; const CUmemAllocationProp *prop; unsigned long long flags; } p_cuMemCreate;
typedef struct { CUmemGenericAllocationHandle handle; } p_cuMemRelease;
typedef struct { CUdeviceptr ptr; size_t size; size_t offset; CUmemGenericAllocationHandle handle; unsigned long long flags; } p_cuMemMap;
typedef struct { CUdeviceptr ptr; size_t size; } p_cuMemUnmap;
typedef struct { CUdeviceptr *ptr; size_t size; size_t alignment; CUdeviceptr addr; unsigned long long flags; } p_cuMemAddressReserve;
typedef struct { CUdeviceptr ptr; size_t size; } p_cuMemAddressFree;
typedef struct { CUdeviceptr ptr; size_t size; const CUmemAccessDesc *desc; size_t count; } p_cuMemSetAccess;
typedef struct { CUmemGenericAllocationHandle *handle; void *osHandle; CUmemAllocationHandleType shHandleType; } p_cuMemImportFromShareableHandle;
typedef struct { void *shareableHandle; CUmemGenericAllocationHandle handle; CUmemAllocationHandleType handleType; unsigned long long flags; } p_cuMemExportToShareableHandle;
#if CUDA_VERSION >= 12010
typedef struct { CUmemGenericAllocationHandle *mcHandle; const CUmulticastObjectProp *prop; } p_cuMulticastCreate;
typedef struct { CUmemGenericAllocationHandle mcHandle; CUdevice dev; } p_cuMulticastAddDevice;
typedef struct { CUmemGenericAllocationHandle mcHandle; size_t mcOffset; CUmemGenericAllocationHandle memHandle; size_t memOffset; size_t size; unsigned long long flags; } p_cuMulticastBindMem;
typedef struct { CUmemGenericAllocationHandle mcHandle; size_t mcOffset; CUdeviceptr memptr; size_t size; unsigned long long flags; } p_cuMulticastBindAddr;
typedef struct { CUmemGenericAllocationHandle mcHandle; CUdevice dev; size_t mcOffset; size_t size; } p_cuMulticastUnbind;
#endif

static int g_fd = -1;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static int g_maps_dumped = 0;
static char g_host[64] = "host";
static CUpti_SubscriberHandle g_sub;
static CUptiResult (*p_subscribe)(CUpti_SubscriberHandle *, CUpti_CallbackFunc, void *);
static CUptiResult (*p_enable)(uint32_t, CUpti_SubscriberHandle, CUpti_CallbackDomain, CUpti_CallbackId);
static CUptiResult (*p_resstr)(CUptiResult, const char **);

static void hk_dump_maps(void) {
    if (g_maps_dumped) return;
    g_maps_dumped = 1;
    const char *dir = getenv("CUMEM_HOOK_DIR");
    char path[512];
    snprintf(path, sizeof path, "%s/cumem_%s_%d.maps", dir && *dir ? dir : "/tmp", g_host, (int)getpid());
    int in = open("/proc/self/maps", O_RDONLY);
    int out = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (in >= 0 && out >= 0) {
        char buf[8192];
        ssize_t n;
        while ((n = read(in, buf, sizeof buf)) > 0) {
            if (write(out, buf, (size_t)n) != n) break;
        }
    }
    if (in >= 0) close(in);
    if (out >= 0) close(out);
}

static void hk_emit(const char *line, size_t len) {
    pthread_mutex_lock(&g_mu);
    if (g_fd >= 0) {
        ssize_t r = write(g_fd, line, len);
        (void)r;
    }
    pthread_mutex_unlock(&g_mu);
}

static void CUPTIAPI hk_cb(void *ud, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void *cbdata) {
    (void)ud;
    if (domain != CUPTI_CB_DOMAIN_DRIVER_API) return;
    const CUpti_CallbackData *cb = (const CUpti_CallbackData *)cbdata;
    if (cb->callbackSite != CUPTI_API_EXIT) return;
    hk_dump_maps();

    struct timeval tv;
    gettimeofday(&tv, NULL);
    struct tm lt;
    localtime_r(&tv.tv_sec, &lt);
    char line[2048];
    int n = snprintf(line, sizeof line, "t=%ld.%06ld lt=%02d:%02d:%02d.%06ld pid=%d tid=%ld api=%s site=exit ret=%d",
                     (long)tv.tv_sec, (long)tv.tv_usec, lt.tm_hour, lt.tm_min, lt.tm_sec, (long)tv.tv_usec,
                     (int)getpid(), (long)syscall(SYS_gettid), cb->functionName,
                     cb->functionReturnValue ? *(const int *)cb->functionReturnValue : -1);
    const void *P = cb->functionParams;
#define APP(...) n += snprintf(line + n, sizeof line - (size_t)n, __VA_ARGS__)
    switch (cbid) {
    case CUPTI_DRIVER_TRACE_CBID_cuMemCreate: {
        const p_cuMemCreate *p = P;
        APP(" size=%zu", p->size);
        if (p->prop) APP(" dev=%d loctype=%d htype=%d", p->prop->location.id, (int)p->prop->location.type, (int)p->prop->requestedHandleTypes);
        if (p->handle) APP(" handle=0x%llx", *p->handle);
        break; }
    case CUPTI_DRIVER_TRACE_CBID_cuMemRelease: { const p_cuMemRelease *p = P; APP(" handle=0x%llx", p->handle); break; }
    case CUPTI_DRIVER_TRACE_CBID_cuMemMap: { const p_cuMemMap *p = P; APP(" ptr=0x%llx size=%zu off=%zu handle=0x%llx", p->ptr, p->size, p->offset, p->handle); break; }
    case CUPTI_DRIVER_TRACE_CBID_cuMemUnmap: { const p_cuMemUnmap *p = P; APP(" ptr=0x%llx size=%zu", p->ptr, p->size); break; }
    case CUPTI_DRIVER_TRACE_CBID_cuMemAddressReserve: { const p_cuMemAddressReserve *p = P; APP(" size=%zu align=%zu", p->size, p->alignment); if (p->ptr) APP(" ptr=0x%llx", *p->ptr); break; }
    case CUPTI_DRIVER_TRACE_CBID_cuMemAddressFree: { const p_cuMemAddressFree *p = P; APP(" ptr=0x%llx size=%zu", p->ptr, p->size); break; }
    case CUPTI_DRIVER_TRACE_CBID_cuMemSetAccess: { const p_cuMemSetAccess *p = P; APP(" ptr=0x%llx size=%zu count=%zu", p->ptr, p->size, p->count); if (p->desc && p->count) APP(" dev=%d", p->desc[0].location.id); break; }
    case CUPTI_DRIVER_TRACE_CBID_cuMemImportFromShareableHandle: { const p_cuMemImportFromShareableHandle *p = P; APP(" htype=%d", (int)p->shHandleType); if (p->handle) APP(" handle=0x%llx", *p->handle); break; }
    case CUPTI_DRIVER_TRACE_CBID_cuMemExportToShareableHandle: { const p_cuMemExportToShareableHandle *p = P; APP(" handle=0x%llx htype=%d", p->handle, (int)p->handleType); break; }
#if CUDA_VERSION >= 12010
    case CUPTI_DRIVER_TRACE_CBID_cuMulticastCreate: { const p_cuMulticastCreate *p = P; if (p->prop) APP(" size=%zu ndev=%u", p->prop->size, p->prop->numDevices); if (p->mcHandle) APP(" mc=0x%llx", *p->mcHandle); break; }
    case CUPTI_DRIVER_TRACE_CBID_cuMulticastAddDevice: { const p_cuMulticastAddDevice *p = P; APP(" mc=0x%llx dev=%d", p->mcHandle, (int)p->dev); break; }
    case CUPTI_DRIVER_TRACE_CBID_cuMulticastBindMem: { const p_cuMulticastBindMem *p = P; APP(" mc=0x%llx mcoff=%zu handle=0x%llx off=%zu size=%zu", p->mcHandle, p->mcOffset, p->memHandle, p->memOffset, p->size); break; }
    case CUPTI_DRIVER_TRACE_CBID_cuMulticastBindAddr: { const p_cuMulticastBindAddr *p = P; APP(" mc=0x%llx mcoff=%zu ptr=0x%llx size=%zu", p->mcHandle, p->mcOffset, p->memptr, p->size); break; }
    case CUPTI_DRIVER_TRACE_CBID_cuMulticastUnbind: { const p_cuMulticastUnbind *p = P; APP(" mc=0x%llx dev=%d mcoff=%zu size=%zu", p->mcHandle, (int)p->dev, p->mcOffset, p->size); break; }
#endif
    default: break;
    }
    void *bt[BT_DEPTH];
    int depth = backtrace(bt, BT_DEPTH);
    APP(" bt=");
    for (int i = 0; i < depth; i++) APP("%s%p", i ? "," : "", bt[i]);
    APP("\n");
#undef APP
    if (n > 0) hk_emit(line, (size_t)n < sizeof line ? (size_t)n : sizeof line - 1);
}

static int hk_is_python(void) {
    char exe[512];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
    if (n <= 0) return 0;
    exe[n] = 0;
    const char *b = strrchr(exe, '/');
    b = b ? b + 1 : exe;
    return strncmp(b, "python", 6) == 0;
}

__attribute__((constructor)) static void hk_init(void) {
    const char *on = getenv("CUMEM_HOOK");
    if (!on || strcmp(on, "1") != 0) return;
    if (!hk_is_python()) return;
    gethostname(g_host, sizeof g_host - 1);
    for (char *c = g_host; *c; c++) if (*c == '.') { *c = 0; break; }

    const char *dir = getenv("CUMEM_HOOK_DIR");
    char path[512];
    snprintf(path, sizeof path, "%s/cumem_%s_%d.log", dir && *dir ? dir : "/tmp", g_host, (int)getpid());
    g_fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (g_fd < 0) { fprintf(stderr, "[cumem_hook] cannot open %s; hook disabled\n", path); return; }

    // libcupti is dlopen'ed by absolute path (CUMEM_HOOK_CUPTI, else the usual locations) so the workload needs
    // no LD_LIBRARY_PATH change; the .so itself links only libdl/libpthread.
    const char *cands[] = { getenv("CUMEM_HOOK_CUPTI"),
        "/usr/local/cuda/extras/CUPTI/lib64/libcupti.so", "/usr/local/cuda/lib64/libcupti.so",
        "/usr/local/lib/python3.12/dist-packages/nvidia/cuda_cupti/lib/libcupti.so.12",
        "libcupti.so.12", "libcupti.so", NULL };
    void *lib = NULL;
    for (size_t i = 0; i < sizeof cands / sizeof cands[0] && !lib; i++) if (cands[i] && *cands[i]) lib = dlopen(cands[i], RTLD_NOW | RTLD_GLOBAL);
    if (!lib) { fprintf(stderr, "[cumem_hook] libcupti not found (set CUMEM_HOOK_CUPTI=/path/libcupti.so.12); hook disabled\n"); return; }
    p_subscribe = (CUptiResult (*)(CUpti_SubscriberHandle *, CUpti_CallbackFunc, void *))dlsym(lib, "cuptiSubscribe");
    p_enable = (CUptiResult (*)(uint32_t, CUpti_SubscriberHandle, CUpti_CallbackDomain, CUpti_CallbackId))dlsym(lib, "cuptiEnableCallback");
    p_resstr = (CUptiResult (*)(CUptiResult, const char **))dlsym(lib, "cuptiGetResultString");
    if (!p_subscribe || !p_enable) { fprintf(stderr, "[cumem_hook] cupti symbols missing; hook disabled\n"); return; }
    CUptiResult r = p_subscribe(&g_sub, (CUpti_CallbackFunc)hk_cb, NULL);
    if (r != CUPTI_SUCCESS) {
        const char *s = NULL;
        if (p_resstr) p_resstr(r, &s);
        fprintf(stderr, "[cumem_hook] cuptiSubscribe failed: %s; hook disabled\n", s ? s : "?");
        return;
    }
    const CUpti_CallbackId ids[] = {
        CUPTI_DRIVER_TRACE_CBID_cuMemCreate, CUPTI_DRIVER_TRACE_CBID_cuMemRelease,
        CUPTI_DRIVER_TRACE_CBID_cuMemMap, CUPTI_DRIVER_TRACE_CBID_cuMemUnmap,
        CUPTI_DRIVER_TRACE_CBID_cuMemAddressReserve, CUPTI_DRIVER_TRACE_CBID_cuMemAddressFree,
        CUPTI_DRIVER_TRACE_CBID_cuMemSetAccess,
        CUPTI_DRIVER_TRACE_CBID_cuMemImportFromShareableHandle, CUPTI_DRIVER_TRACE_CBID_cuMemExportToShareableHandle,
#if CUDA_VERSION >= 12010
        CUPTI_DRIVER_TRACE_CBID_cuMulticastCreate, CUPTI_DRIVER_TRACE_CBID_cuMulticastAddDevice,
        CUPTI_DRIVER_TRACE_CBID_cuMulticastBindMem, CUPTI_DRIVER_TRACE_CBID_cuMulticastBindAddr,
        CUPTI_DRIVER_TRACE_CBID_cuMulticastUnbind,
#endif
    };
    int armed = 0;
    for (size_t i = 0; i < sizeof ids / sizeof ids[0]; i++) {
        if (p_enable(1, g_sub, CUPTI_CB_DOMAIN_DRIVER_API, ids[i]) == CUPTI_SUCCESS) armed++;
    }
    char line[256];
    int n = snprintf(line, sizeof line, "# cumem_hook armed pid=%d host=%s callbacks=%d cuda_version_hdr=%d\n",
                     (int)getpid(), g_host, armed, (int)CUDA_VERSION);
    hk_emit(line, (size_t)n);
    fprintf(stderr, "[cumem_hook] armed (%d callbacks) -> %s\n", armed, path);
}
