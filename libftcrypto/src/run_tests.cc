/* =====================================
 *
 * Copyright (c) 2023, AUSTRAC Australian Government
 * All rights reserved.
 *
 * Licensed under BSD 3 clause license
 *
 */

// -*- compile-command: "nvcc -g -o run_tests run_tests.cc ftcrypto.cu" -*-


#include <stdio.h>
#include <string.h>
#include <assert.h>

extern "C" {
    #include "ftcrypto.h"
}

void print_ge25519_array(const ft_ge25519_array arr) {
    size_t len = ft_array_get_length(arr);
    size_t n_bytes = len * FT_FOLDED_POINT_BYTES;
    uint8_t *mem = (uint8_t *)malloc(n_bytes);
    ft_error err = ft_array_to_bytes_folded(mem, n_bytes, arr);
    assert(err == FT_ERR_NO_ERROR);
    printf("[\n");
    for (size_t i = 0; i < len; ++i) {
        const uint8_t *elt = mem + i * FT_FOLDED_POINT_BYTES;
        printf("%2zu:   0x", i);
        for (ssize_t j = FT_FOLDED_POINT_BYTES - 1; j >= 0; --j) {
            printf("%02hhX", elt[j]);
        }
        printf("\n");
    }
    printf("]\n");
    free(mem);
}


int main()
{
    ft_error err;

    size_t idxs[] = { 1, 2, 3, 10 };
    size_t n_idxs = sizeof(idxs)/sizeof(idxs[0]);

    // This must be called before any library functions are used.
    err = ft_crypto_init();
    assert(err == FT_ERR_NO_ERROR);

    size_t free_mem, total_mem;
    ft_crypto_device_memory(&free_mem, &total_mem);
    printf("Device memory: using %zu / %zu (MiB)\n",
           (total_mem - free_mem) >> 20, total_mem >> 20);

    // Allocate array of 64 million elements, so about 12GiB, that
    // can't fit on the GPU (my unclass GPU has 2GiB of memory, of
    // which only about 500MiB is usually free after Xorg and friends
    // have loaded).
    ft_ge25519_array arr;
    err = ft_array_init_scalar(&arr, 1 << 26, NULL);
    assert(err == FT_ERR_NO_ERROR);
    assert(ft_array_get_length(arr) == (size_t)1 << 26);
    ft_array_free(arr);

    err = ft_array_init_scalar(&arr, 16, NULL);
    assert(err == FT_ERR_NO_ERROR);
    assert(ft_array_get_length(arr) == (size_t)16);

    print_ge25519_array(arr);

    ft_ge25519_array arr2;
    err = ft_array_get_subset(&arr2, idxs, n_idxs, arr);
    assert(err == FT_ERR_NO_ERROR);
    assert(ft_array_get_length(arr2) == n_idxs);

    print_ge25519_array(arr2);

    ft_array_free(arr2);

    size_t idxs2[] = { 3, 7, 8, 11 };
    size_t n_idxs2 = 3;
    err = ft_array_get_subset(&arr2, idxs2, n_idxs2, arr);
    assert(err == FT_ERR_NO_ERROR);

    err = ft_add(arr2, arr2);
    assert(err == FT_ERR_NO_ERROR);

    print_ge25519_array(arr);

    static const size_t buflen = 16*FT_AFFINE_POINT_BYTES;
    uint8_t buf[buflen];
    err = ft_array_to_bytes_affine(buf, buflen, arr);
    assert(err == FT_ERR_NO_ERROR);

    ft_ge25519_array arr3;
    err = ft_array_from_bytes_affine(&arr3, buf, buflen);
    assert(err == FT_ERR_NO_ERROR);
    int64_t bad_idx;
    err = ft_array_validate(&bad_idx, arr3);
    assert(err == FT_ERR_NO_ERROR);
    assert(bad_idx == -1);

    // Should print same as above.
    print_ge25519_array(arr);

    // Create "int array" of values [0, 1, ..., n_ints-1].
    size_t n_ints = 32;
    size_t n_bytes = n_ints * FT_ED25519_SCALARBYTES;
    uint8_t *intarray = (uint8_t *)malloc(n_bytes);
    for (int i = 0; i < n_bytes; ++i)
        intarray[i] = 0;
    for (int i = 0; i < n_ints; ++i)
        intarray[i * FT_ED25519_SCALARBYTES] = i;

    ft_array_free(arr);
    ft_array_from_scalars(&arr, intarray, 32);

    ft_array_free(arr);
    printf("Values 0..8:\n");
    ft_array_from_scalars(&arr, intarray, 9);
    print_ge25519_array(arr);
    printf("reduce:\n");
    ft_array_reduce(arr);
    print_ge25519_array(arr);

    ft_array_free(arr);
    ft_array_from_scalars(&arr, intarray, 7);
    printf("Values 0..6:\n");
    print_ge25519_array(arr);
    ft_array_scan(arr);
    printf("scan:\n");
    print_ge25519_array(arr);

    ft_array_free(arr3);
    ft_array_free(arr2);
    ft_array_free(arr);

    // Check ft_equal
    uint64_t five = 5;
    ft_array_from_small_scalars(&arr, &five, 1);
    ft_array_from_small_scalars(&arr2, &five, 1);
    // both arrays contain the point 5*G, so they should be equal
    int64_t res = 0;
    ft_equal(&res, arr, arr2);
    assert(res == 1);

    ft_add(arr2, arr);
    // arr2 now contains 10*G, so the arrays should not be equal
    ft_equal(&res, arr, arr2);
    assert(res == 0);

    ft_array_free(arr);
    ft_array_free(arr2);

    uint64_t range[7] = { 1, 2, 3, 4, 5, 6, 7 };
    ft_array_from_small_scalars(&arr, range, 7);

    print_ge25519_array(arr);
    uint64_t init[] = { 3, 5, 8, 2 };
    ft_array_from_small_scalars(&arr2, init, 4);
    print_ge25519_array(arr2);

    int64_t c[4];
    ft_array_contains(c, arr2, arr);
    assert(c[0] == 1);
    assert(c[1] == 1);
    assert(c[2] == 0);
    assert(c[3] == 1);

    ft_array_free(arr);
    ft_array_free(arr2);

    // a = ft_array_from_scalars(5,8,20,30,50)
    // b = ft_array_from_scalars(10,20,30)
    // keys = [2,2,3]
    uint64_t init1[] = { 5, 8, 20, 30, 50 };
    ft_array_from_small_scalars(&arr, init1, 5);
    print_ge25519_array(arr);

    uint64_t init2[] = { 10, 20, 30 };
    ft_array_from_small_scalars(&arr2, init2, 3);
    print_ge25519_array(arr2);

    size_t keys[] = { 2, 2, 3 };
    err = ft_reduce_isum(arr, keys, 3, arr2);
    assert(err == FT_ERR_NO_ERROR);
    print_ge25519_array(arr);

    ft_array_free(arr);
    ft_array_free(arr2);

    free(intarray);

    return 0;
}
