/* =====================================
 *
 * Copyright (c) 2023, AUSTRAC Australian Government
 * All rights reserved.
 *
 * Licensed under BSD 3 clause license
 *
 */

#include <algorithm>
#include <numeric>
#include <cooperative_groups.h>

extern "C" {
    #include "ftcrypto.h"
}
#include "cuda_wrap.h"
#include "sodium_fe.cu"
#include "sodium_ge.cu"

struct ft_ge25519_array_s {
    ge25519_p3 *elts;  // pointer to the array of Ed25519 points
    size_t n_elts;     // number of points in the array
    size_t n_reserved; // number of points currently allocated; must have n_elts <= n_reserved.
    uint8_t *scratch;  // scratch space of FT_ED25519_SCALARBYTES * n_elts bytes
                       // for operations on the array
};


// This is set in ft_crypto_init().
static int SM_COUNT = 1;


const char *ft_error_str(ft_error err) {
    static const char *errors[] = {
        "no error",
        "out of memory",
        "parameter sizes did not mmatch",
        "bad size parameter",
        "index out of bounds",
        "invalid ed25519 point",
        "failed to initialise libsodium",
        "insufficient space in destination",
        "ciphertext corrupted",
        "internal library error",
        "device or driver does not support a required feature"
    };
    // Ensure that the number of error strings is the same as the
    // number of error codes:
    static_assert(
        sizeof(errors)/sizeof(errors[0]) == FT_ERR_MAX_ERR + 1,
        "error string array length mismatch");

    if (err < 0 || err > FT_ERR_MAX_ERR)
        return "bad error value";
    return errors[err];
}


// Return good choices for grid and block size when executing kernel
// KFn. Adapted from standard Cuda code published on the Cuda blog.
// NB: This is currently called before every single kernel call, which
// might be wasteful depending on how expensive those Cuda runtime API
// calls are.
template<typename KFn>
static inline void
find_optimal_geometry(
    int &blockSize, int &gridSize,
    KFn func, size_t smem_bytes = 0)
{
    int maxActiveBlocks;
    int minGridSize;

    cuda_check(
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func, smem_bytes, 0),
        "find_optimal_geometry");
    cuda_check(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, func, blockSize, smem_bytes),
        "find_optimal_geometry");
    gridSize = maxActiveBlocks * SM_COUNT;
}


template<typename KFn>
static inline int
find_gridsz_for_fixed_blocksz(int blockSize, KFn func, size_t smem_bytes)
{
    int maxActiveBlocks;
    cuda_check(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, func, blockSize, smem_bytes),
        "find_gridsz_for_fixed_blocksz");
    return maxActiveBlocks * SM_COUNT;
}


// This is the folded encoding of curve identity element.
//
// The curve zero/identity element is the pair (0, 1), and the
// representation below is the encoding of the y-coordinate 1.
static const __device__ uint8_t GE25519_ZERO[FT_FOLDED_POINT_BYTES] = {
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};


// This is the folded encoding of the chosen curve generator.
//
// The generator is the unique point (x, 4/5) on Ed25519 for which x
// is positive. The encoding below is the y-coordinate 4/5 in the
// base field.
//
// \\ PARI/GP script for verification:
// > yB = Mod(4/5, 2^255 - 19); \\ This is 4/5 modulo 2^255-19
// > lift(yB)                   \\ Pretend the value is an integer
// 46316835694926478169428394003475163141307993866256225615783033603165251855960
// > printf("%Px", Vecrev(digits(lift(yB), 256)))  \\ Print in hex in LSByte first
// [58,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66]
static const __device__ uint8_t GE25519_GEN[FT_FOLDED_POINT_BYTES] = {
    0x58, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
    0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
    0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
    0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66
};


static __host__ __device__ void ge25519_zero(ge25519_p3 *zero)
{
    int r = ge25519_frombytes(zero, GE25519_ZERO);

    // TODO: work out how to handle this error properly

    // GE25519_ZERO is a constant defined above and should never
    // fail to convert to a ge25519_p3.
    // if (r)
    //     return FT_ERR_INTERNAL_LIBRARY_ERROR;

    // return FT_ERR_NO_ERROR;
}


static __device__ void ge25519_addto(ge25519_p3 *dest, const ge25519_p3 *src)
{
    ge25519_p1p1 sum_p1p1;
    ge25519_cached p_cached;
    ge25519_p3_to_cached(&p_cached, src);
    ge25519_add(&sum_p1p1, dest, &p_cached);
    ge25519_p1p1_to_p3(dest, &sum_p1p1);
}


static __device__ void ge25519_subfrom(ge25519_p3 *dest, const ge25519_p3 *src)
{
    ge25519_p1p1 diff_p1p1;
    ge25519_cached p_cached;
    ge25519_p3_to_cached(&p_cached, src);
    ge25519_sub(&diff_p1p1, dest, &p_cached);
    ge25519_p1p1_to_p3(dest, &diff_p1p1);
}


static __device__ void ge25519_neg(ge25519_p3 *pt)
{
    fe25519_neg(pt->Y, pt->Y);
    fe25519_neg(pt->T, pt->T);
}


static __device__ int ge25519_iszero(const ge25519_p3 *elt)
{
    // The zero point on the curve is (x, y) = (0, 1) in affine
    // coordinates, which is (X, Y, Z) = (0, t, t) for any non-zero t
    // in projective coordinates. So elt is zero if both X and Y-Z are
    // zero.
    ge25519_p2 elt_p2;
    ge25519_p3_to_p2(&elt_p2, elt);
    fe25519 Y_cmp_Z;
    fe25519_sub(Y_cmp_Z, elt_p2.Y, elt_p2.Z);
    return fe25519_iszero(elt_p2.X) && fe25519_iszero(Y_cmp_Z);
}


static __device__ int ge25519_equal(const ge25519_p3 *elt1, const ge25519_p3 *elt2)
{
    ge25519_p3 diff = *elt1;
    ge25519_subfrom(&diff, elt2);

    return ge25519_iszero(&diff);
}


ft_error ft_crypto_init() {
    int device;
    cudaDeviceProp props;
    cuda_check(cudaGetDevice(&device), "get device");
    cuda_check(cudaGetDeviceProperties(&props, device), "get device properties");
    SM_COUNT = props.multiProcessorCount;

    if ( ! props.managedMemory || props.major < 6)
        return FT_ERR_INSUFFICIENT_DEVICE_OR_DRIVER;

    return FT_ERR_NO_ERROR;
}


ft_error ft_crypto_device_memory(size_t *free_mem, size_t *total_mem)
{
    cuda_check(cudaMemGetInfo(free_mem, total_mem), "mem get info");

    return FT_ERR_NO_ERROR;
}


static ft_error ft_allocate_array(ft_ge25519_array *array, size_t n_elts)
{
    ft_ge25519_array new_array;
    ge25519_p3 *new_array_elts;
    size_t pt_array_bytes = sizeof(ge25519_p3) * n_elts;
    size_t scratch_bytes = FT_ED25519_SCALARBYTES * n_elts;

    // new_array_elts and scratch are on the GPU, but they are
    // 'managed', which means it can be accessed on the CPU if
    // necessary, and they can be bigger than the available space on
    // the GPU.
    cuda_malloc_managed(&new_array_elts, pt_array_bytes + scratch_bytes);
    if ( ! new_array_elts)
        return FT_ERR_OUT_OF_MEMORY;

    // new_array is on the CPU
    new_array = (ft_ge25519_array) malloc(sizeof(*new_array));
    if ( ! new_array) {
        cuda_free(new_array_elts);
        return FT_ERR_OUT_OF_MEMORY;
    }
    new_array->elts = new_array_elts;
    new_array->n_elts = n_elts;
    new_array->n_reserved = n_elts;
    new_array->scratch = (uint8_t *)(new_array_elts + n_elts);
    *array = new_array;
    return FT_ERR_NO_ERROR;
}


void ft_array_free(ft_ge25519_array array)
{
    cuda_free(array->elts);
    free(array);
}


__global__ void k_ft_array_set_zero(ge25519_p3 *pts, size_t n_pts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for ( ; i < n_pts; i += blockDim.x * gridDim.x) {
        ge25519_zero(pts + i);
    }
}


__global__ void k_ft_array_set_scalar(
    ge25519_p3 *pts, size_t n_elts, const uint8_t *value)
{
    ge25519_p3 base;
    ge25519_frombytes(&base, GE25519_GEN);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for ( ; i < n_elts; i += blockDim.x * gridDim.x) {
        // TODO: Fix and use ge25519_scalarmult_base instead
        ge25519_scalarmult(pts + i, value, &base);
    }
}


__global__ void k_ft_array_set_point(
    ge25519_p3 *pts, size_t n_elts, const ge25519_p3 *pt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for ( ; i < n_elts; i += blockDim.x * gridDim.x) {
        pts[i] = *pt;
    }
}


size_t ft_array_get_length(const_ft_ge25519_array array)
{
    return array->n_elts;
}


ft_error ft_array_set_length(ft_ge25519_array array, size_t newlen)
{
    size_t len = array->n_elts;

    int blocksz, gridsz;
    find_optimal_geometry(blocksz, gridsz, k_ft_array_set_zero);

    // truncate array
    if (newlen <= len/2) {
        // If newlen is "much less" than len, then we reallocate
        // to a smaller array rather than saving the difference in
        // reserve. This is particularly important for reduction
        // to a single element for example (e.g. ft_array_reduce()).
        ge25519_p3 *new_array_elts;
        size_t pt_array_bytes = sizeof(ge25519_p3) * newlen;
        size_t scratch_bytes = FT_ED25519_SCALARBYTES * newlen;
        cuda_malloc_managed(&new_array_elts, pt_array_bytes + scratch_bytes);
        cuda_memcpy_on_device(new_array_elts, array->elts, newlen * sizeof(ge25519_p3));
        cuda_free(array->elts);
        array->elts = new_array_elts;
        array->n_elts = newlen;
        array->n_reserved = newlen;
        array->scratch = (uint8_t *)(new_array_elts + newlen);
    } else if (newlen <= len) {
        array->n_elts = newlen;
    } else if (newlen <= array->n_reserved) {
        // extend array into reserved space
        array->n_elts = newlen;
        k_ft_array_set_zero<<<gridsz, blocksz>>>(array->elts + len, newlen - len);
    } else {
        // not enough reserved space, copy to new array
        // TODO: Refactor with the first if-else clause.
        ge25519_p3 *new_array_elts;
        size_t pt_array_bytes = sizeof(ge25519_p3) * newlen;
        size_t scratch_bytes = FT_ED25519_SCALARBYTES * newlen;
        cuda_malloc_managed(&new_array_elts, pt_array_bytes + scratch_bytes);
        cuda_memcpy_on_device(new_array_elts, array->elts, len * sizeof(ge25519_p3));
        k_ft_array_set_zero<<<gridsz, blocksz>>>(new_array_elts + len, newlen - len);
        cuda_free(array->elts);
        array->elts = new_array_elts;
        array->n_elts = newlen;
        array->n_reserved = newlen;
        array->scratch = (uint8_t *)(new_array_elts + newlen);
    }
    return FT_ERR_NO_ERROR;
}


ft_error ft_array_init_scalar(
    ft_ge25519_array *array, size_t n_elts,
    const uint8_t *scalar)
{
    // TODO: Accept a size hint to allocate more space than needed.

    ft_ge25519_array new_array;
    ft_error err;

    err = ft_allocate_array(&new_array, n_elts);
    if (err) return err;

    int blocksz, gridsz;

    if (n_elts > 0) {
        if (scalar) {
            find_optimal_geometry(blocksz, gridsz, k_ft_array_set_scalar);
            cuda_memcpy_to_device(new_array->scratch, scalar, FT_ED25519_SCALARBYTES);
            k_ft_array_set_scalar<<<gridsz, blocksz>>>(new_array->elts, n_elts, (const uint8_t *)new_array->scratch);
        } else {
            find_optimal_geometry(blocksz, gridsz, k_ft_array_set_zero);
            k_ft_array_set_zero<<<gridsz, blocksz>>>(new_array->elts, n_elts);
        }
    }
    *array = new_array;
    return FT_ERR_NO_ERROR;
}


ft_error ft_array_init_point(
    ft_ge25519_array *array, size_t n_elts,
    const_ft_ge25519_array src, size_t idx)
{
    // TODO: Accept a size hint to allocate more space than needed.

    ft_ge25519_array new_array;
    ft_error err;

    err = ft_allocate_array(&new_array, n_elts);
    if (err) return err;

    int blocksz, gridsz;

    find_optimal_geometry(blocksz, gridsz, k_ft_array_set_point);
    k_ft_array_set_point<<<gridsz, blocksz>>>(new_array->elts, n_elts, src->elts + idx);

    *array = new_array;
    return FT_ERR_NO_ERROR;
}


__global__ void k_ft_array_assign_to_subset(
    ge25519_p3 *dest, const size_t *idxs, size_t n_idxs, const ge25519_p3 *src)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for ( ; i < n_idxs; i += blockDim.x * gridDim.x) {
        dest[idxs[i]] = src[i];
    }
}


ft_error ft_array_assign_to_subset(
    ft_ge25519_array array,
    const size_t *idxs, size_t n_idxs,
    const_ft_ge25519_array src)
{
    if (n_idxs != src->n_elts)
        return FT_ERR_SIZE_MISMATCH;

    int blocksz, gridsz;
    find_optimal_geometry(blocksz, gridsz, k_ft_array_assign_to_subset);

    cuda_memcpy_to_device(array->scratch, idxs, n_idxs * sizeof(size_t));
    k_ft_array_assign_to_subset<<<gridsz, blocksz>>>(array->elts, (const size_t *)array->scratch, n_idxs, src->elts);
    return FT_ERR_NO_ERROR;
}

__global__ void k_ft_array_assign_zero_to_subset(
    ge25519_p3 *dest, const size_t *idxs, size_t n_idxs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for ( ; i < n_idxs; i += blockDim.x * gridDim.x) {
        ge25519_zero(dest + idxs[i]);
    }
}


ft_error ft_array_assign_zero_to_subset(
    ft_ge25519_array array,
    const size_t *idxs, size_t n_idxs)
{
    int blocksz, gridsz;
    find_optimal_geometry(blocksz, gridsz, k_ft_array_assign_zero_to_subset);

    cuda_memcpy_to_device(array->scratch, idxs, n_idxs * sizeof(size_t));
    k_ft_array_assign_zero_to_subset<<<gridsz, blocksz>>>(array->elts, (const size_t *)array->scratch, n_idxs);
    return FT_ERR_NO_ERROR;
}

__global__ void k_ft_array_get_subset(
    ge25519_p3 *dest, const size_t *idxs, size_t n_idxs, const ge25519_p3 *src)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for ( ; i < n_idxs; i += blockDim.x * gridDim.x) {
        dest[i] = src[idxs[i]];
    }
}


ft_error ft_array_get_subset(
    ft_ge25519_array *array,
    const size_t *idxs, size_t n_idxs,
    const_ft_ge25519_array src)
{
    ft_ge25519_array new_array;
    int err = ft_allocate_array(&new_array, n_idxs);
    if (err) return err;

    int blocksz, gridsz;
    find_optimal_geometry(blocksz, gridsz, k_ft_array_get_subset);

    cuda_memcpy_to_device(new_array->scratch, idxs, n_idxs * sizeof(size_t));
    k_ft_array_get_subset<<<gridsz, blocksz>>>(new_array->elts, (const size_t *)new_array->scratch, n_idxs, src->elts);

    *array = new_array;
    return FT_ERR_NO_ERROR;
}


__global__ void k_ft_array_assign_to_range(
    ge25519_p3 *dest,
    size_t start, size_t stop, size_t step,
    const ge25519_p3 *src, size_t n_elts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for ( ; i < n_elts; i += blockDim.x * gridDim.x) {
        size_t idx = start + i * step;
        if (idx < stop)
            dest[idx] = src[i];
    }
}


ft_error ft_array_assign_to_range(
    ft_ge25519_array array,
    size_t start, size_t stop, size_t step,
    const_ft_ge25519_array src)
{
    // Empty range
    if (stop <= start)
        return FT_ERR_NO_ERROR;

    // Check whether the number of elements in the range exceeds the
    // length of src.
    if ((stop - start) / step > src->n_elts)
        return FT_ERR_SIZE_MISMATCH;

    int blocksz, gridsz;
    find_optimal_geometry(blocksz, gridsz, k_ft_array_assign_to_range);

    k_ft_array_assign_to_range<<<gridsz, blocksz>>>(array->elts, start, stop, step, src->elts, src->n_elts);
    return FT_ERR_NO_ERROR;
}


__global__ void k_ft_array_get_range(
    ge25519_p3 *dest,
    size_t start, size_t stop, size_t step,
    const ge25519_p3 *src, size_t n_elts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for ( ; i < n_elts; i += blockDim.x * gridDim.x) {
        size_t idx = start + i * step;
        if (idx < stop)
            dest[i] = src[idx];
    }
}


ft_error ft_array_get_range(
    ft_ge25519_array *array,
    size_t start, size_t stop, size_t step,
    const_ft_ge25519_array src)
{
    size_t n_elts = (stop <= start) ? 0 : (stop - start) / step;

    // Check whether the number of elements in the range exceeds the
    // length of src.
    if (n_elts > src->n_elts)
        return FT_ERR_SIZE_MISMATCH;

    ft_ge25519_array new_array;
    int err = ft_allocate_array(&new_array, n_elts);
    if (err) return err;

    int blocksz, gridsz;
    find_optimal_geometry(blocksz, gridsz, k_ft_array_get_range);

    k_ft_array_get_range<<<gridsz, blocksz>>>(new_array->elts, start, stop, step, src->elts, n_elts);

    *array = new_array;
    return FT_ERR_NO_ERROR;
}


// subset idxs are in array->scratch
static ft_error select_subset(ft_ge25519_array array, size_t n_idxs)
{
    ft_ge25519_array subarray;
    ft_error err = ft_array_get_subset(&subarray, (const size_t *)array->scratch, n_idxs, array);
    if (err) return err;
    // We could just switch the arrays' elts pointers around, but this
    // is more straightforward
    cuda_memcpy_on_device(array->elts, subarray->elts, n_idxs * sizeof(ge25519_p3));
    err = ft_array_set_length(array, n_idxs);
    ft_array_free(subarray);
    return err;
}


ft_error ft_array_delete_range(
    ft_ge25519_array array,
    size_t start, size_t stop, size_t step)
{
    size_t n_elts = (stop - start) / step;
    size_t *remaining_idxs = (size_t *)malloc(n_elts * sizeof(size_t));
    if (remaining_idxs == NULL)
        return FT_ERR_OUT_OF_MEMORY;

    size_t n_remaining = 0;
    for (size_t i = 0; i < array->n_elts; ++i) {
        if (i >= start && i < stop && (i - start) % step == 0) {
            // Skip if  start <= i < stop  and  i == start + m*step  for some m
            continue;
        }
        if (n_remaining == n_elts) {
            // Somehow the number of idxs is more than we calculated?
            // Signal a "Hamish can't count error".
            free(remaining_idxs);
            return FT_ERR_INTERNAL_LIBRARY_ERROR;
        }

        remaining_idxs[n_remaining++] = i;
    }

    cuda_memcpy_to_device(array->scratch, remaining_idxs, n_remaining * sizeof(size_t));
    free(remaining_idxs);

    return select_subset(array, n_remaining);
}


ft_error ft_array_delete_subset(
    ft_ge25519_array array,
    const size_t *idxs, size_t n_idxs)
{
    size_t *delete_idxs = (size_t *) malloc(n_idxs * sizeof(size_t));
    if (delete_idxs == NULL)
        return FT_ERR_OUT_OF_MEMORY;

    std::copy(idxs, idxs + n_idxs, delete_idxs);
    std::sort(delete_idxs, delete_idxs + n_idxs);
    // input can have dups; remove them
    std::unique(delete_idxs, delete_idxs + n_idxs);

    size_t *remaining_idxs = (size_t *)malloc(array->n_elts * sizeof(size_t));
    if (remaining_idxs == NULL) {
        free(delete_idxs);
        return FT_ERR_OUT_OF_MEMORY;
    }

    size_t n_remaining = 0;
    size_t delete_cnt = 0;
    for (size_t i = 0; i < array->n_elts; ++i) {
        if (i == delete_idxs[delete_cnt]) {
            // Hit an index to be removed
            delete_cnt++;
        } else {
            remaining_idxs[n_remaining++] = i;
        }
    }
    free(delete_idxs);

    cuda_memcpy_to_device(array->scratch, remaining_idxs, n_remaining * sizeof(size_t));
    free(remaining_idxs);

    return select_subset(array, n_remaining);
}


__global__ void k_ft_array_to_bytes_folded(
    uint8_t *d_mem, size_t n_bytes,
    const ge25519_p3 *elts, size_t n_elts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for ( ; i < n_elts; i += blockDim.x * gridDim.x) {
        uint8_t *dest = d_mem + i * FT_FOLDED_POINT_BYTES;
        ge25519_p3_tobytes(dest, elts + i);
    }
}


ft_error ft_array_to_bytes_folded(
    uint8_t *mem, size_t n_bytes,
    const_ft_ge25519_array array)
{
    size_t max_elts = n_bytes / FT_FOLDED_POINT_BYTES;
    if (max_elts < array->n_elts)
        return FT_ERR_INSUFFICIENT_SPACE;

    int blocksz, gridsz;
    find_optimal_geometry(blocksz, gridsz, k_ft_array_to_bytes_folded);

    k_ft_array_to_bytes_folded<<<gridsz, blocksz>>>(array->scratch, n_bytes, array->elts, array->n_elts);
    cuda_memcpy_from_device(mem, array->scratch, n_bytes);

    return FT_ERR_NO_ERROR;
}


__global__ void k_ft_array_to_bytes_affine(
    uint8_t *d_mem, size_t n_bytes,
    const ge25519_p3 *elts, size_t n_elts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for ( ; i < n_elts; i += blockDim.x * gridDim.x) {
        uint8_t *dest = d_mem + i * FT_AFFINE_POINT_BYTES;
        ge25519_p2 elt_p2;
        fe25519 Zinv, x, y;

        // elt_p2 is (X : Y : Z). We store the affine coordinates (x, y) = (X/Z, Y/Z).
        ge25519_p3_to_p2(&elt_p2, elts + i);
        fe25519_invert(Zinv, elt_p2.Z);
        fe25519_mul(x, elt_p2.X, Zinv);
        fe25519_mul(y, elt_p2.Y, Zinv);
        fe25519_tobytes(dest, x);
        fe25519_tobytes(dest + FT_AFFINE_POINT_BYTES / 2, y);
    }
}


ft_error ft_array_to_bytes_affine(
    uint8_t *mem, size_t n_bytes,
    const_ft_ge25519_array array)
{
    size_t max_elts = n_bytes / FT_AFFINE_POINT_BYTES;
    if (max_elts < array->n_elts)
        return FT_ERR_INSUFFICIENT_SPACE;

    int blocksz, gridsz;
    find_optimal_geometry(blocksz, gridsz, k_ft_array_to_bytes_affine);

    uint8_t *res;
    cuda_malloc_managed(&res, n_bytes);
    if (res == NULL)
        return FT_ERR_OUT_OF_MEMORY;

    k_ft_array_to_bytes_affine<<<gridsz, blocksz>>>(res, n_bytes, array->elts, array->n_elts);
    cuda_memcpy_from_device(mem, res, n_bytes);
    cuda_free(res);

    return FT_ERR_NO_ERROR;
}


ft_error ft_array_to_bytes(
    uint8_t *mem, size_t n_bytes,
    const_ft_ge25519_array array)
{
    size_t max_elts = n_bytes / FT_EXTENDED_POINT_BYTES;
    if (max_elts < array->n_elts)
        return FT_ERR_INSUFFICIENT_SPACE;

    cuda_memcpy_from_device(mem, array->elts, array->n_elts * sizeof(ge25519_p3));

    return FT_ERR_NO_ERROR;
}


__device__ int reduce_max(cooperative_groups::thread_group g, int *temp, int val)
{
    int lane = g.thread_rank();

    // Each iteration halves the number of active threads
    // Each thread adds its partial max[i] to max[lane+i]
    for (int i = g.size() / 2; i > 0; i /= 2) {
        temp[lane] = val;
        g.sync(); // wait for all threads to store
        if (lane < i)
            val = max(val, temp[lane + i]);
        g.sync(); // wait for all threads to load
    }
    return val; // note: only thread 0 will return actual max
}


__global__ void k_ft_array_from_bytes_folded(
    int *res,
    ge25519_p3 *elts, size_t n_elts,
    const uint8_t *d_mem)
{
    int idx_of_failure = -1;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for ( ; i < n_elts; i += blockDim.x * gridDim.x) {
        const uint8_t *src = d_mem + i * FT_FOLDED_POINT_BYTES;
        int r = ge25519_frombytes(elts + i, src);
        if (r)
            idx_of_failure = i;
    }

    extern __shared__ int temp[];
    auto g = cooperative_groups::this_thread_block();
    int block_max = reduce_max(g, temp, idx_of_failure);

    if (g.thread_rank() == 0)
        atomicMax(res, block_max);
}


ft_error ft_array_from_bytes_folded(
    int64_t *res,
    ft_ge25519_array *array,
    const uint8_t *mem, size_t n_bytes)
{
    ft_ge25519_array new_array;
    size_t n_elts;
    ft_error err;

    if (n_bytes % FT_FOLDED_POINT_BYTES != 0)
        return FT_ERR_BAD_SIZE;
    n_elts = n_bytes / FT_FOLDED_POINT_BYTES;
    err = ft_allocate_array(&new_array, n_elts);
    if (err) return err;

    int blocksz, gridsz;
    find_optimal_geometry(blocksz, gridsz, k_ft_array_from_bytes_folded);

    size_t shared_bytes = blocksz * sizeof(int);

    int *d_res;
    cuda_malloc_managed(&d_res, sizeof(int));
    *d_res = -1;
    cuda_memcpy_to_device(new_array->scratch, mem, n_bytes);
    k_ft_array_from_bytes_folded<<<gridsz, blocksz, shared_bytes >>>(
        d_res, new_array->elts, new_array->n_elts, new_array->scratch);

    *res = (int64_t) *d_res;
    cuda_free(d_res);

    *array = new_array;
    return FT_ERR_NO_ERROR;
}


__global__ void k_ft_array_from_bytes_affine(
    ge25519_p3 *elts, size_t n_elts,
    const uint8_t *d_mem)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for ( ; i < n_elts; i += blockDim.x * gridDim.x) {
        const uint8_t *src = d_mem + i * FT_AFFINE_POINT_BYTES;
        fe25519_frombytes(elts[i].X, src);
        fe25519_frombytes(elts[i].Y, src + FT_AFFINE_POINT_BYTES / 2);
        fe25519_1(elts[i].Z);
        fe25519_mul(elts[i].T, elts[i].X, elts[i].Y);
    }
}


ft_error ft_array_from_bytes_affine(
    ft_ge25519_array *array,
    const uint8_t *mem, size_t n_bytes)
{
    ft_ge25519_array new_array;
    size_t n_elts;
    ft_error err;

    if (n_bytes % FT_AFFINE_POINT_BYTES != 0)
        return FT_ERR_BAD_SIZE;
    n_elts = n_bytes / FT_AFFINE_POINT_BYTES;
    err = ft_allocate_array(&new_array, n_elts);
    if (err) return err;

    int blocksz, gridsz;
    find_optimal_geometry(blocksz, gridsz, k_ft_array_from_bytes_affine);

    uint8_t *input;
    cuda_malloc_managed(&input, n_bytes);
    if (input == NULL)
        return FT_ERR_OUT_OF_MEMORY;
    cuda_memcpy_to_device(input, mem, n_bytes);
    k_ft_array_from_bytes_affine<<<gridsz, blocksz>>>(new_array->elts, new_array->n_elts, input);
    cuda_free(input);

    *array = new_array;
    return FT_ERR_NO_ERROR;
}


ft_error ft_array_from_bytes(
    ft_ge25519_array *array,
    const uint8_t *mem, size_t n_bytes)
{
    ft_ge25519_array new_array;
    size_t n_elts;
    ft_error err;

    if (n_bytes % FT_EXTENDED_POINT_BYTES != 0)
        return FT_ERR_BAD_SIZE;
    n_elts = n_bytes / FT_EXTENDED_POINT_BYTES;
    err = ft_allocate_array(&new_array, n_elts);
    if (err) return err;

    cuda_memcpy_to_device(new_array->elts, mem, n_bytes);

    *array = new_array;
    return FT_ERR_NO_ERROR;
}


__global__ void k_ft_array_validate(int *res, const ge25519_p3 *elts, size_t n_elts)
{
    int idx_of_failure = -1;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for ( ; i < n_elts; i += blockDim.x * gridDim.x) {
        int r = ! (ge25519_is_on_curve(elts + i) && ge25519_is_on_main_subgroup(elts + i));
        if (r)
            idx_of_failure = i;
    }
    extern __shared__ int temp[];
    auto g = cooperative_groups::this_thread_block();
    int block_max = reduce_max(g, temp, idx_of_failure);

    if (g.thread_rank() == 0)
        atomicMax(res, block_max);
}


ft_error ft_array_validate(int64_t *res, const_ft_ge25519_array array)
{
    int blocksz, gridsz;
    find_optimal_geometry(blocksz, gridsz, k_ft_array_validate);

    size_t shared_bytes = blocksz * sizeof(int);
    int r = -1;
    cuda_memcpy_to_device(array->scratch, &r, sizeof(int));
    k_ft_array_validate<<<gridsz, blocksz, shared_bytes>>>(
        (int *)array->scratch, array->elts, array->n_elts);

    cuda_memcpy_from_device(&r, array->scratch, sizeof(int));
    *res = (int64_t)r;

    return FT_ERR_NO_ERROR;
}


__global__ void k_ft_array_from_scalars(
    ge25519_p3 *elts, size_t n_elts,
    const uint8_t *d_scalars)
{
    ge25519_p3 base;
    ge25519_frombytes(&base, GE25519_GEN);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for ( ; i < n_elts; i += blockDim.x * gridDim.x) {
        const uint8_t *d_scalar = d_scalars + i * FT_ED25519_SCALARBYTES;

        // TODO: The call to scalarmult_base should be equivalent to
        // (and faster than) the code below it, but it fails for
        // scalars >= 8 for some reason.
        //
        //ge25519_scalarmult_base(elts + i, d_scalar);
        ge25519_scalarmult(elts + i, d_scalar, &base);
    }
}


ft_error ft_array_from_scalars(
    ft_ge25519_array *array,
    const uint8_t *scalars, size_t n_scalars)
{
    ft_ge25519_array new_array;
    ft_error err;

    err = ft_allocate_array(&new_array, n_scalars);
    if (err) return err;

    int blocksz, gridsz;
    find_optimal_geometry(blocksz, gridsz, k_ft_array_from_scalars);

    cuda_memcpy_to_device(new_array->scratch, scalars, n_scalars * FT_ED25519_SCALARBYTES);
    k_ft_array_from_scalars<<<gridsz, blocksz>>>(new_array->elts, new_array->n_elts, new_array->scratch);

    *array = new_array;
    return FT_ERR_NO_ERROR;
}


ft_error ft_array_from_small_scalars(
    ft_ge25519_array *array,
    const uint64_t *small_scalars, size_t n_scalars)
{
    // TODO: Could do this faster using the knowledge that the scalars
    // are small.

    size_t n_bytes = n_scalars * FT_ED25519_SCALARBYTES;
    uint8_t *scalars = (uint8_t *) malloc(n_bytes);
    if (scalars == NULL)
        return FT_ERR_OUT_OF_MEMORY;

    memset(scalars, 0, n_bytes);
    for (int i = 0; i < n_scalars; ++i) {
        uint8_t *off = scalars + i * FT_ED25519_SCALARBYTES;
        uint64_t sc = small_scalars[i];

        // Convert to byte array manually, just in case host is
        // big-endian...
        for (int j = 0; j < sizeof(uint64_t); ++j)
            off[j] = (sc >> (j*8)) & 255u;
    }

    ft_error err = ft_array_from_scalars(array, scalars, n_scalars);
    free(scalars);
    return err;
}


__global__ void k_ft_scale(
    ge25519_p3 *elts, size_t n_elts, const uint8_t *d_scalar)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for ( ; i < n_elts; i += blockDim.x * gridDim.x) {
        ge25519_scalarmult(elts + i, d_scalar, elts + i);
    }
}


void ft_scale(ft_ge25519_array array, const uint8_t *scalar)
{
    int blocksz, gridsz;
    find_optimal_geometry(blocksz, gridsz, k_ft_scale);

    cuda_memcpy_to_device(array->scratch, scalar, FT_ED25519_SCALARBYTES);
    k_ft_scale<<<gridsz, blocksz>>>(array->elts, array->n_elts, array->scratch);
}


__global__ void k_ft_mul(
    ge25519_p3 *elts, size_t n_elts, const uint8_t *d_scalars)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for ( ; i < n_elts; i += blockDim.x * gridDim.x) {
        const uint8_t *d_scalar = d_scalars + i * FT_ED25519_SCALARBYTES;
        ge25519_scalarmult(elts + i, d_scalar, elts + i);
    }
}


ft_error ft_mul(
    ft_ge25519_array array, const uint8_t *scalars, size_t n_scalars)
{
    if (n_scalars != array->n_elts)
        return FT_ERR_SIZE_MISMATCH;

    size_t n_bytes = n_scalars * FT_ED25519_SCALARBYTES;
    int blocksz, gridsz;
    find_optimal_geometry(blocksz, gridsz, k_ft_mul);

    cuda_memcpy_to_device(array->scratch, scalars, n_bytes);
    k_ft_mul<<<gridsz, blocksz>>>(array->elts, array->n_elts, array->scratch);

    return FT_ERR_NO_ERROR;
}


__global__ void k_ft_array_mux(
    ge25519_p3 *iftrue,
    const int64_t *cond,
    const ge25519_p3 *iffalse, size_t n_elts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for ( ; i < n_elts; i += blockDim.x * gridDim.x) {
        if (cond[i] == 0)
            iftrue[i] = iffalse[i];
    }
}


ft_error ft_array_mux(
    ft_ge25519_array iftrue,
    const int64_t *cond,
    const_ft_ge25519_array iffalse)
{
    if (iftrue->n_elts != iffalse->n_elts)
        return FT_ERR_SIZE_MISMATCH;

    int blocksz, gridsz;
    find_optimal_geometry(blocksz, gridsz, k_ft_array_mux);

    cuda_memcpy_to_device(iftrue->scratch, cond, sizeof(int64_t)*iftrue->n_elts);
    k_ft_array_mux<<<gridsz, blocksz>>>(
        iftrue->elts, (const int64_t *)iftrue->scratch, iffalse->elts, iftrue->n_elts);

    return FT_ERR_NO_ERROR;
}


__global__ void k_ft_array_contains(
    int64_t *res,
    const ge25519_p3 *array1, size_t array1_elts,
    const ge25519_p3 *array2, size_t array2_elts)
{
    // Initialise res
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    for ( ; j < array1_elts; j += blockDim.x * gridDim.x) {
        res[j] = (int64_t)0;
    }

    // Look for array2[i] in array1
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for ( ; i < array2_elts; i += blockDim.x * gridDim.x) {
        ge25519_p3 target = array2[i];
        for (int j = 0; j < array1_elts; ++j) {
            // eq is 0 or 1
            int eq = ge25519_equal(&target, array1 + j);
            // atomicOr only takes unsigned int arguments on compute
            // capability < 6; the hacky casting will work though,
            // since we're only OR-ing together values that are 0 or 1.
            atomicOr((unsigned int *)(res + j), (unsigned int)eq);
        }
    }
}


ft_error ft_array_contains(
    int64_t *res,
    const_ft_ge25519_array array1,
    const_ft_ge25519_array array2)
{
    int blocksz, gridsz;
    find_optimal_geometry(blocksz, gridsz, k_ft_array_contains);

    k_ft_array_contains<<<gridsz, blocksz>>>(
        (int64_t *)array1->scratch, array1->elts, array1->n_elts, array2->elts, array2->n_elts);
    cuda_memcpy_from_device(res, array1->scratch, array1->n_elts * sizeof(int64_t));

    return FT_ERR_NO_ERROR;
}


__global__ void k_ft_add(
    ge25519_p3 *res, const ge25519_p3 *summand, size_t n_elts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for ( ; i < n_elts; i += blockDim.x * gridDim.x) {
        ge25519_addto(res + i, summand + i);
    }
}


ft_error ft_add(ft_ge25519_array res, const_ft_ge25519_array summand)
{
    if (summand->n_elts != res->n_elts)
        return FT_ERR_SIZE_MISMATCH;

    int blocksz, gridsz;
    find_optimal_geometry(blocksz, gridsz, k_ft_add);

    k_ft_add<<<gridsz, blocksz>>>(res->elts, summand->elts, res->n_elts);

    return FT_ERR_NO_ERROR;
}


using pair = std::pair<size_t, size_t>;


__global__ void k_ft_reduce_isum(
    ge25519_p3 *dest, const pair *dest_idxs, size_t n_idxs, const size_t *offsets, const size_t *src_idxs, const ge25519_p3 *src)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for ( ; i < n_idxs; i += blockDim.x * gridDim.x) {
        ge25519_p3 *dest_i = dest + dest_idxs[i].first;
        const size_t *src_offs = src_idxs + offsets[i];
        for (size_t j = 0; j < dest_idxs[i].second; ++j) {
            ge25519_addto(dest_i, src + src_offs[j]);
        }
    }
}


static size_t run_length_encode_inplace(pair *v, size_t len)
{
    if (len == 0)
        return 0;

    size_t curr = 0;
    v[curr].second = 1;
    for (size_t i = 1; i < len; ++i) {
        if (v[curr].first == v[i].first) {
            v[curr].second++;
        } else {
            curr++;
            v[curr] = std::make_pair(v[i].first, 1);
        }
    }
    // return the new length
    return curr + 1;
}


ft_error ft_reduce_isum(
    ft_ge25519_array array,
    const size_t *idxs, size_t n_idxs,
    const_ft_ge25519_array src)
{
    // TODO: This implementation would probably be better if done with
    // the help of the Thrust/CUB library on the GPU.

    if (n_idxs != src->n_elts)
        return FT_ERR_SIZE_MISMATCH;

    pair *idx_pairs;
    cuda_malloc_managed(&idx_pairs, n_idxs * sizeof(pair));

    // Create the array of pairs (idx[i], i) where the second value is
    // an index into `src`.
    for (size_t i = 0; i < n_idxs; ++i)
        idx_pairs[i] = std::make_pair(idxs[i], i);

    // Sort according to first element of the pair
    std::sort(
        idx_pairs, idx_pairs + n_idxs,
        [] (const pair &a, const pair &b) { return a.first < b.first; });

    // Split off the permuted src idxs
    size_t *src_idxs;
    cuda_malloc_managed(&src_idxs, n_idxs * sizeof(size_t));
    std::transform(
        idx_pairs, idx_pairs + n_idxs, src_idxs,
        [] (const pair &p) { return p.second; });

    // Repurpose the array to do run length encoding of first elements
    size_t pairs_len = run_length_encode_inplace(idx_pairs, n_idxs);

    // TODO: Performance would be improved if, at this point, we "sort
    // according to run length" (including sorting associated chunks
    // of value_idxs), so that most blocks will be dealing with runs
    // of the same length.

    // Determine the offsets for each bucket
    size_t *offsets;
    cuda_malloc_managed(&offsets, pairs_len * sizeof(size_t));

    // In C++ 17 this simplifies with `std::transform_exclusive_scan`.
    // select the number of src idxs for each dest idx (i.e. p.second):
    std::transform(
        idx_pairs, idx_pairs + pairs_len, offsets,
        [] (const pair &p) { return p.second; });
    // inclusive scan
    std::partial_sum(offsets, offsets + pairs_len, offsets);
    // exclusive scan: right shift by one elt and prepend a zero
    for (ssize_t i = pairs_len - 1; i > 0; --i)
        offsets[i] = offsets[i - 1];
    offsets[0] = 0;

    int blocksz, gridsz;
    find_optimal_geometry(blocksz, gridsz, k_ft_reduce_isum);

    k_ft_reduce_isum<<<gridsz, blocksz>>>(
        array->elts,
        idx_pairs, pairs_len, offsets,
        src_idxs, src->elts);

    cuda_free(offsets);
    cuda_free(src_idxs);
    cuda_free(idx_pairs);
    return FT_ERR_NO_ERROR;
}


__global__ void k_ft_sub(
    ge25519_p3 *minuend, const ge25519_p3 *subtrahend, size_t n_elts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for ( ; i < n_elts; i += blockDim.x * gridDim.x) {
        ge25519_subfrom(minuend + i, subtrahend + i);
    }
}


ft_error ft_sub(ft_ge25519_array minuend, const_ft_ge25519_array subtrahend)
{
    if (subtrahend->n_elts != minuend->n_elts)
        return FT_ERR_SIZE_MISMATCH;

    int blocksz, gridsz;
    find_optimal_geometry(blocksz, gridsz, k_ft_sub);

    k_ft_sub<<<gridsz, blocksz>>>(minuend->elts, subtrahend->elts, minuend->n_elts);

    return FT_ERR_NO_ERROR;
}


__global__ void k_ft_neg(ge25519_p3 *elts, size_t n_elts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for ( ; i < n_elts; i += blockDim.x * gridDim.x) {
        ge25519_neg(elts + i);
    }
}


void ft_neg(ft_ge25519_array array)
{
    int blocksz, gridsz;
    find_optimal_geometry(blocksz, gridsz, k_ft_neg);

    k_ft_neg<<<gridsz, blocksz>>>(array->elts, array->n_elts);
}


__global__ void k_ft_index(int64_t *res, ge25519_p3 *elts, size_t n_elts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for ( ; i < n_elts; i += blockDim.x * gridDim.x) {
        res[i] = ! ge25519_iszero(elts + i);
    }
}


void ft_index(int64_t *res, const_ft_ge25519_array array)
{
    int blocksz, gridsz;
    find_optimal_geometry(blocksz, gridsz, k_ft_index);

    k_ft_index<<<gridsz, blocksz>>>((int64_t *)array->scratch, array->elts, array->n_elts);
    cuda_memcpy_from_device(res, array->scratch, array->n_elts * sizeof(int64_t));
}


__global__ void k_ft_equal(
    int64_t *res,
    ge25519_p3 *elts1, ge25519_p3 *elts2, size_t n_elts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for ( ; i < n_elts; i += blockDim.x * gridDim.x) {
        res[i] = ge25519_equal(elts1 + i, elts2 + i);
    }
}


ft_error ft_equal(
    int64_t *res,
    const_ft_ge25519_array array1,
    const_ft_ge25519_array array2)
{
    if (array1->n_elts != array2->n_elts)
        return FT_ERR_SIZE_MISMATCH;

    int blocksz, gridsz;
    find_optimal_geometry(blocksz, gridsz, k_ft_equal);

    k_ft_equal<<<gridsz, blocksz>>>(
        (int64_t *)array1->scratch, array1->elts, array2->elts, array1->n_elts);
    cuda_memcpy_from_device(res, array1->scratch, array1->n_elts * sizeof(int64_t));
    return FT_ERR_NO_ERROR;
}


ft_error ft_not_equal(
    int64_t *res,
    const_ft_ge25519_array array1,
    const_ft_ge25519_array array2)
{
    if (array1->n_elts != array2->n_elts)
        return FT_ERR_SIZE_MISMATCH;

    int err = ft_equal(res, array1, array2);
    if (err) return err;

    for (size_t i = 0; i < array1->n_elts; ++i)
        res[i] = !res[i];

    return FT_ERR_NO_ERROR;
}

static __device__ void print_tree(const ge25519_p3 *tree, int n_elts, int idx = 0) {
    if (threadIdx.x == idx) {
        uint8_t buf[32];
        for (int i = 0; i < n_elts; ++i) {
            ge25519_p3_tobytes(buf, tree + i);
            printf("  %d:    0x", i);
            for (ssize_t j = 31; j >= 0; --j) {
                printf("%02hX", buf[j]);
            }
            printf("\n");
        }
    }
}


// n_elts must be a power of two
static __device__ void block_prescan_upsweep(
    ge25519_p3 *elts, int idx, size_t n_elts)
{
    size_t offset = 1;
    for (size_t d = n_elts >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (idx < d) {
            size_t ai = offset * (2 * idx + 1) - 1;
            size_t bi = offset * (2 * idx + 2) - 1;
            ge25519_addto(elts + bi, elts + ai);
        }
        offset *= 2;
    }
    __syncthreads();
}


// n_elts must be a power of two
static __device__ void block_prescan_downsweep(
    ge25519_p3 *elts, int idx, size_t n_elts)
{
    size_t offset = n_elts;
    for (size_t d = 1; d < n_elts; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (idx < d) {
            size_t ai = offset * (2 * idx + 1) - 1;
            size_t bi = offset * (2 * idx + 2) - 1;

            ge25519_p3 tmp = elts[ai];
            elts[ai] = elts[bi];
            ge25519_addto(elts + bi, &tmp);
        }
    }
    __syncthreads();
}

static constexpr bool FILL_WITH_ZEROS = true;
static constexpr bool DONT_FILL_WITH_ZEROS = false;

// n_elts must be a power of two
template< bool fill_with_zeros >
static __device__ void block_copy(
    ge25519_p3 *dest, const ge25519_p3 *src, int idx, size_t n_elts_remaining, int spread_log2)
{
    int idx1 = 2 * idx;
    int idx2 = 2 * idx + 1;

    // Only the very last block will ever satisfy either of the
    // first two conditions in this if-else. The normal path for
    // all blocks is via the 'else'.
    if (idx1 >= n_elts_remaining && idx2 >= n_elts_remaining) {
        // zero load
        if (fill_with_zeros == FILL_WITH_ZEROS) {
            ge25519_zero(dest + 2 * idx);
            ge25519_zero(dest + 2 * idx + 1);
        }
    } else if (idx1 < n_elts_remaining && idx2 >= n_elts_remaining) {
        // half load
        dest[2 * idx] = src[2 * idx << spread_log2];
        if (fill_with_zeros == FILL_WITH_ZEROS) {
            ge25519_zero(dest + 2 * idx + 1);
        }
    } else {
        // normal load
        dest[2 * idx] = src[2 * idx << spread_log2];
        dest[2 * idx + 1] = src[(2 * idx + 1) << spread_log2];
    }
}


__global__ void k_ft_array_reduce(
    ge25519_p3 *elts, size_t n_active_elts, int spread_log2)
{
    extern __shared__ ge25519_p3 tree[];

    size_t elts_per_blk = 2 * blockDim.x;

    // n_active_elts_rounded_up is the smallest value >= n_active_elts
    // that is divisible by elts_per_blk.
    size_t rem = n_active_elts % elts_per_blk;
    size_t adjustment = (rem == 0) ? 0 : (elts_per_blk - rem);
    size_t n_active_elts_rounded_up = n_active_elts + adjustment;

    int i = elts_per_blk * blockIdx.x + threadIdx.x;
    int blk = elts_per_blk * blockIdx.x;

    for ( ; i < n_active_elts_rounded_up; i += elts_per_blk * gridDim.x, blk += elts_per_blk * gridDim.x) {
        int idx = threadIdx.x;
        size_t off = blk << spread_log2;
        ge25519_p3 *elts_offset = elts + off;

        block_copy<FILL_WITH_ZEROS>(tree, elts_offset, idx, n_active_elts - blk, spread_log2);

        block_prescan_upsweep(tree, idx, elts_per_blk);

        // copy result to elts[0]
        if (idx == 0) {
            elts_offset[0] = tree[elts_per_blk - 1];
        }
    }
}


ft_error ft_array_reduce(ft_ge25519_array array)
{
    // NB: blocksz must be at most 128 in order for smem_bytes =
    // 160*2*blocksz = 40960 < 49152 (maximum for testing device).
    int blocksz_log2 = 7; // must be log2(blocksz)
    int blocksz = 1 << blocksz_log2;
    int smem_bytes = sizeof(ge25519_p3) * 2 * blocksz;
    int gridsz = find_gridsz_for_fixed_blocksz(blocksz, k_ft_array_reduce, smem_bytes);

    size_t n_elts_remaining = array->n_elts;
    int spread_log2 = 0;

    while (n_elts_remaining > 1) {
        k_ft_array_reduce<<< gridsz, blocksz, smem_bytes >>>(array->elts, n_elts_remaining, spread_log2);

        spread_log2 += blocksz_log2 + 1;
        // n_elts_remaining = ceiling(n_elts_remaining / (2 * blocksz))
        n_elts_remaining = (n_elts_remaining + (2 * blocksz) - 1) / (2 * blocksz);
    }

    // Result is already in position 0.
    ft_array_set_length(array, 1);

    return FT_ERR_NO_ERROR;
}


__global__ void k_ft_array_prescan(ge25519_p3 *elts, size_t n_elts, ge25519_p3 *top_sums)
{
    extern __shared__ ge25519_p3 tree[];

    size_t elts_per_blk = 2 * blockDim.x;

    // n_active_elts_rounded_up is the smallest value >= n_active_elts
    // that is divisible by elts_per_blk.
    size_t rem = n_elts % elts_per_blk;
    size_t adjustment = (rem == 0) ? 0 : (elts_per_blk - rem);
    size_t n_elts_rounded_up = n_elts + adjustment;

    int i = elts_per_blk * blockIdx.x + threadIdx.x;
    int blk = elts_per_blk * blockIdx.x;

    int b = blockIdx.x;

    for ( ; i < n_elts_rounded_up; i += elts_per_blk * gridDim.x) {
        int idx = threadIdx.x;
        ge25519_p3 *elts_offset = elts + blk;

        // load data from global memory
        block_copy<FILL_WITH_ZEROS>(tree, elts_offset, idx, n_elts - blk, 0);

        block_prescan_upsweep(tree, idx, elts_per_blk);

        // save last element elsewhere then set to zero
        if (idx == 0) {
            top_sums[b] = tree[elts_per_blk - 1];
            ge25519_zero(tree + elts_per_blk - 1);
        }

        block_prescan_downsweep(tree, idx, elts_per_blk);

        // store results back into global memory
        block_copy<DONT_FILL_WITH_ZEROS>(elts_offset, tree, idx, n_elts - blk, 0);

        blk += elts_per_blk * gridDim.x;
        b += blockDim.x * gridDim.x;
    }
}


__global__ void k_ft_array_update_prescan_inplace(ge25519_p3 *elts, size_t n_elts, const ge25519_p3 *updates)
{
    int elts_per_blk = 2 * blockDim.x;

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int blk = blockDim.x * blockIdx.x;

    // Indices add update[b] to elt[i*b], ..., elt[i*b + blockDim.x]
    // for each b = 0, ..., blockDim.x.
    for ( ; i < n_elts; i += blockDim.x * gridDim.x, blk += blockDim.x * gridDim.x) {
        const ge25519_p3 *update = updates + i / elts_per_blk;
        ge25519_addto(elts + i, update);
    }
}


ft_error ft_array_prescan(ft_ge25519_array array)
{
    // NB: blocksz must be at most 128 in order for smem_bytes =
    // 160*2*blocksz = 40960 < 49152 (maximum for testing device).
    int blocksz_log2 = 7; // must be log2(blocksz)
    int blocksz = 1 << blocksz_log2;
    int smem_bytes = sizeof(ge25519_p3) * 2 * blocksz;
    int gridsz = find_gridsz_for_fixed_blocksz(blocksz, k_ft_array_prescan, smem_bytes);

    size_t n_elts = array->n_elts;

    // n_top_sums = ceiling(n_elts / (2 * blocksz))
    size_t n_top_sums = (n_elts + (2 * blocksz) - 1) / (2 * blocksz);

    ft_ge25519_array top_sums;
    ft_array_init_scalar(&top_sums, n_top_sums, NULL);

    k_ft_array_prescan<<< gridsz, blocksz, smem_bytes >>>(array->elts, n_elts, top_sums->elts);

    if (top_sums->n_elts > 1) {
        ft_array_prescan(top_sums);

        find_optimal_geometry(blocksz, gridsz, k_ft_array_update_prescan_inplace);
        k_ft_array_update_prescan_inplace<<< gridsz, blocksz >>>(array->elts, n_elts, top_sums->elts);
    }

    ft_array_free(top_sums);

    return FT_ERR_NO_ERROR;
}


ft_error ft_array_scan(ft_ge25519_array array)
{
    ft_ge25519_array prescan;
    ft_error err;

    err = ft_allocate_array(&prescan, array->n_elts + 1);
    if (err) return err;

    // copy array to prescan and set the last element to zero.
    cuda_memcpy_on_device(prescan->elts, array->elts, array->n_elts * sizeof(ge25519_p3));
    k_ft_array_set_zero<<<1, 1>>>(prescan->elts + array->n_elts, 1);

    err = ft_array_prescan(prescan);
    if (err) goto cleanup;

    // copy all but the first element (which is necessarily zero) back to array
    cuda_memcpy_on_device(array->elts, prescan->elts + 1, array->n_elts * sizeof(ge25519_p3));

cleanup:
    ft_array_free(prescan);
    return err;
}
