/* =====================================
 *
 * Copyright (c) 2023, AUSTRAC Australian Government
 * All rights reserved.
 *
 * Licensed under BSD 3 clause license
 *
 */

#pragma once

#include <cstdio>
#include <cstdlib>

/*
 * Convenience wrappers around some CUDA library functions
 */
static inline void
cuda_print_errmsg(cudaError err, const char *msg, const char *file, const int line)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "Fatal CUDA error at %s:%d : %s : %s\n",
                file, line, msg, cudaGetErrorString(err));
        if (cudaDeviceReset() != cudaSuccess)
            fprintf(stderr, "   ...and failed to reset the device!\n");
        exit(EXIT_FAILURE);
    }
}

#define cuda_check(err, msg)                            \
    cuda_print_errmsg(err, msg, __FILE__, __LINE__)

#define cuda_get_device(iptr)                           \
    cuda_check(cudaGetDevice(iptr), "get device")
#define cuda_device_get_attribute(iptr, attr, dev_id)           \
    cuda_check(cudaDeviceGetAttribute(iptr, attr, dev_id),      \
            "device get attribute")

#define cuda_malloc(ptr, size)                                  \
    cuda_check(cudaMalloc(ptr, size), "memory allocation")
#define cuda_malloc_host(ptr, size)                                     \
    cuda_check(cudaMallocHost(ptr, size), "pinned memory allocation")
#define cuda_malloc_managed(ptr, size)                                  \
    cuda_check(cudaMallocManaged(ptr, size),                            \
            "unified memory allocation (default attach)")
#define cuda_malloc_managed_host(ptr, size)                             \
    cuda_check(cudaMallocManaged(ptr, size, cudaMemAttachHost),         \
            "unified memory allocation (host attach)")
#define cuda_stream_attach_mem(stream, ptr)                             \
    cuda_check(cudaStreamAttachMemAsync(stream, ptr), "attach unified memory to stream")
#define cuda_mem_advise(ptr, nbytes, advice, id)           \
    cuda_check(cudaMemAdvise(ptr, nbytes, advice, id), "memory advice")
#define cuda_free(ptr)                                  \
    cuda_check(cudaFree(ptr), "memory deallocation")

#define cuda_free_host(ptr)                                  \
    cuda_check(cudaFreeHost(ptr), "pinned memory deallocation")
#define cuda_memcpy_to_device(dest, src, size)                          \
    cuda_check(cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice), "copy to device")
#define cuda_memcpy_async_to_device(dest, src, size, strm)                    \
    cuda_check(cudaMemcpyAsync(dest, src, size, cudaMemcpyHostToDevice, strm), "async copy to device")
#define cuda_memcpy_from_device(dest, src, size)                        \
    cuda_check(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost), "copy from device")
#define cuda_memcpy_async_from_device(dest, src, size, strm)                    \
    cuda_check(cudaMemcpyAsync(dest, src, size, cudaMemcpyDeviceToHost, strm), "async copy from device")
#define cuda_memcpy_on_device(dest, src, size)                        \
    cuda_check(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToDevice), "copy on device")
#define cuda_memcpy_async_on_device(dest, src, size, strm)                   \
    cuda_check(cudaMemcpyAsync(dest, src, size, cudaMemcpyDeviceToDevice, strm), "async copy on device")

#define cuda_memset(dest, val, size)                        \
    cuda_check(cudaMemset(dest, val, size), "memset on device")
#define cuda_device_synchronize() \
    cuda_check(cudaDeviceSynchronize(), "device synchronize")

#define cuda_stream_create(stream) \
    cuda_check(cudaStreamCreate(stream), "stream create")
#define cuda_stream_destroy(stream) \
    cuda_check(cudaStreamDestroy(stream), "stream destroy")
#define cuda_stream_synchronize(stream) \
    cuda_check(cudaStreamSynchronize(stream), "stream synchronize")
#define cuda_stream_wait_event(stream, event)                           \
    cuda_check(cudaStreamWaitEvent(stream, event, 0), "stream wait event")

#define cuda_event_create(event) \
    cuda_check(cudaEventCreate(event), "event create")
#define cuda_event_destroy(event) \
    cuda_check(cudaEventDestroy(event), "event destroy")
#define cuda_event_synchronize(event)                                  \
    cuda_check(cudaEventSynchronize(event), "event synchronize")
#define cuda_event_record(event, stream)                                       \
    cuda_check(cudaEventRecord(event, stream), "event record")
#define cuda_event_elapsed_time(time, start, end)                           \
    cuda_check(cudaEventElapsedTime(time, start, end), "event elapsed time")


namespace cuda {

class event {
    cudaEvent_t ev;

public:
    event() {
        cuda_event_create(&ev);
    }

    void synchronize() {
        cuda_event_synchronize(ev);
    }

    float elapsed_time(event &other) {
        float t;
        other.synchronize();
        cuda_event_elapsed_time(&t, ev, other.ev);
        return t;
    }

    void record(cudaStream_t &strm) {
        cuda_event_record(ev, strm);
    }

    ~event() {
        cuda_event_destroy(ev);
    }
};

}
