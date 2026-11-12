/* =====================================
 *
 * Copyright (c) 2023, AUSTRAC Australian Government
 * All rights reserved.
 *
 * Licensed under BSD 3 clause license
 *
 */

/*
 * Regression test for ft_array_to_bytes in ftcrypto.c.
 *
 * The bounds check used to compare the wrong way (max_elts >
 * array->n_elts instead of max_elts < array->n_elts), so a caller
 * who passed a buffer too small for the array got no error back and
 * the function wrote past the end of it. This builds a 100 element
 * array and a buffer only sized for 10, and checks that
 * ft_array_to_bytes refuses instead of writing out of bounds.
 *
 * Build with: cc -fsanitize=address -I. test_array_to_bytes_bounds.c
 *             ftcrypto.c -lsodium -o test_array_to_bytes_bounds
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "ftcrypto.h"

int main(void)
{
    ft_error err = ft_crypto_init();
    assert(err == FT_ERR_NO_ERROR);

    ft_ge25519_array arr;
    size_t n_elts = 100;
    err = ft_array_init(&arr, n_elts, NULL, 0);
    assert(err == FT_ERR_NO_ERROR);

    size_t undersized_n_bytes = 10 * FT_FOLDED_POINT_BYTES;
    uint8_t *mem = malloc(undersized_n_bytes);
    assert(mem != NULL);

    err = ft_array_to_bytes(mem, undersized_n_bytes, arr);
    assert(err == FT_ERR_INSUFFICIENT_SPACE);

    size_t exact_n_bytes = n_elts * FT_FOLDED_POINT_BYTES;
    uint8_t *mem2 = malloc(exact_n_bytes);
    assert(mem2 != NULL);
    err = ft_array_to_bytes(mem2, exact_n_bytes, arr);
    assert(err == FT_ERR_NO_ERROR);

    free(mem);
    free(mem2);
    ft_array_free(arr);

    printf("ok\n");
    return 0;
}
