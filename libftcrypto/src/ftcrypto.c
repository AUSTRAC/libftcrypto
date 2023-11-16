/* =====================================
 *
 * Copyright (c) 2023, AUSTRAC Australian Government
 * All rights reserved.
 *
 * Licensed under BSD 3 clause license
 *
 */

#include <string.h>

#define HAVE_TI_MODE
#include <sodium.h>
#include <sodium/private/ed25519_ref10.h>

#include "ftcrypto.h"

#define FT_ED25519_SCALARBYTES crypto_core_ed25519_SCALARBYTES

/*
 * ElGamal ciphertexts consist of two points on the Ed25519 curve. A
 * point on this curve can be represented with either 4, 2, or 1
 * elements of the curve's base field, which for Ed25519 is the finite
 * field with p = 2^255 - 19 elements in it. In principle, one such
 * element requires 256 bits, or 32 bytes, of storage; in practice an
 * extra 8 bytes is allocated in order to speed up handling carry
 * propagation, so a base field element takes 40 bytes. The choice of
 * point representation is a classic time space trade-off, as
 * operations on larger representations take less time than operations
 * on smaller representations. The represenations are:
 *
 * Representation  | # elements | size of point | size of ciphertext
 * ----------------+------------+---------------+-------------------
 * projective      | 4          | 160           | 320
 * affine          | 2          | 80            | 160
 * folded          | 1          | 32            | 64
 *
 * In the first two cases, the size of a point in the given
 * representation is #elements * 40 bytes, and the size of the
 * ciphertext in that representation is 2 * size of point (because a
 * ciphertext consists of two points). For the folded representation,
 * the extra 8 bytes of space is removed since points in folded
 * representation can't be used for arithmetic anyway.
 */

struct ft_ge25519_array {
    ge25519_p3 *elts;
    size_t n_elts;
};


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
        "internal library error"
    };
    // Ensure that the number of error strings is the same as the
    // number of error codes:
    _Static_assert(
        sizeof(errors)/sizeof(errors[0]) == FT_ERR_MAX_ERR + 1,
        "error string array length mismatch");

    if (err < 0 || err > FT_ERR_MAX_ERR)
        return "bad error value";
    return errors[err];
}


/*
 * "Private" objects and functions
 */

// This is the folded encoding of curve identity element.
//
// The curve zero/identity element is the pair (0, 1), and the
// representation below is the encoding of the y-coordinate 1.
static const uint8_t GE25519_ZERO[FT_FOLDED_POINT_BYTES] = {
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
static const uint8_t GE25519_GEN[FT_FOLDED_POINT_BYTES] = {
    0x58, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
    0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
    0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
    0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66
};


static inline ge25519_p3 *ptr_at_idx(ft_ge25519_array array, size_t idx)
{
    return array->elts + idx;
}


static inline const ge25519_p3 *cptr_at_idx(const_ft_ge25519_array array, size_t idx)
{
    return array->elts + idx;
}


static inline ft_error ptr_at_idx_chkd(ge25519_p3 **res, ft_ge25519_array array, size_t idx)
{
    if (idx >= array->n_elts)
        return FT_ERR_OUT_OF_BOUNDS;
    *res = ptr_at_idx(array, idx);
    return FT_ERR_NO_ERROR;
}


static inline ft_error cptr_at_idx_chkd(const ge25519_p3 **res, const_ft_ge25519_array array, size_t idx)
{
    if (idx >= array->n_elts)
        return FT_ERR_OUT_OF_BOUNDS;
    *res = cptr_at_idx(array, idx);
    return FT_ERR_NO_ERROR;
}


static inline void ge25519_addto(ge25519_p3 *dest, const ge25519_p3 *src)
{
    ge25519_p1p1 sum_p1p1;
    ge25519_cached p_cached;
    ge25519_p3_to_cached(&p_cached, src);
    ge25519_add(&sum_p1p1, dest, &p_cached);
    ge25519_p1p1_to_p3(dest, &sum_p1p1);
}


static inline void ge25519_subfrom(ge25519_p3 *dest, const ge25519_p3 *src)
{
    ge25519_p1p1 diff_p1p1;
    ge25519_cached p_cached;
    ge25519_p3_to_cached(&p_cached, src);
    ge25519_sub(&diff_p1p1, dest, &p_cached);
    ge25519_p1p1_to_p3(dest, &diff_p1p1);
}


static inline ft_error ge25519_zero(ge25519_p3 *zero)
{
    int r = ge25519_frombytes(zero, GE25519_ZERO);
    // GE25519_ZERO is a constant defined above and should never
    // fail to convert to a ge25519_p3.
    if (r)
        return FT_ERR_INTERNAL_LIBRARY_ERROR;

    return FT_ERR_NO_ERROR;
}


static inline ft_error ge25519_one(ge25519_p3 *one)
{
    int r = ge25519_frombytes(one, GE25519_GEN);
    // GE25519_GEN is a constant defined above and should never
    // fail to convert to a ge25519_p3.
    if (r)
        return FT_ERR_INTERNAL_LIBRARY_ERROR;

    return FT_ERR_NO_ERROR;
}


/*
 * "Public" functions
 */

ft_error ft_crypto_init() {
    int r = sodium_init();
    return r ? FT_ERR_SODIUM_INIT : FT_ERR_NO_ERROR;
}


static ft_error ft_allocate_array(ft_ge25519_array *array, size_t n_elts)
{
    ft_ge25519_array new_array;
    ge25519_p3 *new_array_elts;
    size_t total_bytes = sizeof(ge25519_p3) * n_elts;

    new_array_elts = malloc(total_bytes);
    if ( ! new_array_elts)
        return FT_ERR_OUT_OF_MEMORY;
    new_array = malloc(sizeof(*new_array));
    if ( ! new_array) {
        free(new_array_elts);
        return FT_ERR_OUT_OF_MEMORY;
    }
    new_array->elts = new_array_elts;
    new_array->n_elts = n_elts;
    *array = new_array;
    return FT_ERR_NO_ERROR;
}


void ft_array_free(ft_ge25519_array array)
{
    free(array->elts);
    free(array);
}


size_t ft_array_length(const_ft_ge25519_array array)
{
    return array->n_elts;
}


ft_error ft_array_init(
    ft_ge25519_array *array, size_t n_elts,
    const size_t *idxs, size_t n_idxs)
{
    ft_ge25519_array new_array;
    ft_error err;

    // Can't have the subset of one's larger than the whole array.
    if (n_idxs > n_elts)
        return FT_ERR_SIZE_MISMATCH;

    err = ft_allocate_array(&new_array, n_elts);
    if (err) return err;

    ge25519_p3 plaintext_zero;
    ge25519_zero(&plaintext_zero);

    // Set every element to zero
    for (size_t i = 0; i < n_elts; ++i)
        memcpy(ptr_at_idx(new_array, i), &plaintext_zero, sizeof(plaintext_zero));

    ge25519_p3 plaintext_one;
    ge25519_one(&plaintext_one);

    // Set the subset of idxs to ones
    for (size_t i = 0; i < n_idxs; ++i) {
        ge25519_p3 *ptr;
        err = ptr_at_idx_chkd(&ptr, new_array, idxs[i]);
        if (err) {
            ft_array_free(new_array);
            return err;
        }
        memcpy(ptr, &plaintext_one, sizeof(plaintext_one));
    }
    *array = new_array;
    return FT_ERR_NO_ERROR;
}


ft_error ft_array_from_subset(
    ft_ge25519_array *array,
    const size_t *idxs, size_t n_idxs,
    const_ft_ge25519_array src)
{
    ft_error err = FT_ERR_NO_ERROR;
    ft_ge25519_array new_array;

    err = ft_allocate_array(&new_array, n_idxs);
    if (err) return err;
    for (size_t i = 0; i < n_idxs; ++i) {
        const ge25519_p3 *src_ptr;
        err = cptr_at_idx_chkd(&src_ptr, src, idxs[i]);
        if (err) {
            ft_array_free(new_array);
            return err;
        }
        memcpy(ptr_at_idx(new_array, i), src_ptr, sizeof(*src_ptr));
    }
    *array = new_array;
    return err;
}


ft_error ft_array_to_bytes(
    uint8_t *mem, size_t n_bytes,
    const_ft_ge25519_array array)
{
    size_t max_elts = n_bytes / FT_FOLDED_POINT_BYTES;
    if (max_elts > array->n_elts)
        return FT_ERR_INSUFFICIENT_SPACE;

    for (size_t i = 0; i < array->n_elts; ++i) {
        const ge25519_p3 *src = cptr_at_idx(array, i);
        uint8_t *dest = mem + i * FT_FOLDED_POINT_BYTES;
        ge25519_p3_tobytes(dest, src);
    }

    return FT_ERR_NO_ERROR;
}


ft_error ft_array_from_bytes(
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
    for (size_t i = 0; i < n_elts; ++i) {
        ge25519_p3 *dest = ptr_at_idx(new_array, i);
        const uint8_t *src = mem + i * FT_FOLDED_POINT_BYTES;
        int r = ge25519_frombytes(dest, src);
        if (r) {
            ft_array_free(new_array);
            return FT_ERR_BAD_CIPHERTEXT;
        }
    }
    *array = new_array;

    return FT_ERR_NO_ERROR;
}


void ft_scale_all(ft_ge25519_array array, const uint8_t *scalar)
{
    for (size_t i = 0; i < array->n_elts; ++i) {
        ge25519_p3 *pt = ptr_at_idx(array, i);
        ge25519_scalarmult(pt, scalar, pt);
    }
}


ft_error ft_mul_all(ft_ge25519_array array, const uint8_t *scalars, size_t n_scalars)
{
    if (n_scalars != array->n_elts)
        return FT_ERR_SIZE_MISMATCH;

    for (size_t i = 0; i < array->n_elts; ++i) {
        ge25519_p3 *pt = ptr_at_idx(array, i);
        const uint8_t *scalar = scalars + i * FT_ED25519_SCALARBYTES;
        ge25519_scalarmult(pt, scalar, pt);
    }
    return FT_ERR_NO_ERROR;
}


ft_error ft_add_all(
    ft_ge25519_array res,
    const_ft_ge25519_array summand)
{
    if (res->n_elts != summand->n_elts)
        return FT_ERR_SIZE_MISMATCH;

    for (size_t i = 0; i < res->n_elts; ++i) {
        ge25519_addto(ptr_at_idx(res, i), cptr_at_idx(summand, i));
    }

    return FT_ERR_NO_ERROR;
}


ft_error ft_add_subset(
    ft_ge25519_array res,
    const size_t *idxs, size_t n_idxs,
    const_ft_ge25519_array summand)
{
    ft_error err = FT_ERR_NO_ERROR;

    if (summand->n_elts != n_idxs)
        return FT_ERR_SIZE_MISMATCH;

    for (size_t i = 0; i < n_idxs; ++i) {
        ge25519_p3 *res_ptr;
        err = ptr_at_idx_chkd(&res_ptr, res, idxs[i]);
        if (err) break;
        ge25519_addto(res_ptr, cptr_at_idx(summand, i));
    }

    return err;
}


ft_error ft_sub_all(
    ft_ge25519_array res,
    const_ft_ge25519_array summand)
{
    if (res->n_elts != summand->n_elts)
        return FT_ERR_SIZE_MISMATCH;

    for (size_t i = 0; i < res->n_elts; ++i) {
        ge25519_p3 *ptr = ptr_at_idx(res, i);
        ge25519_subfrom(ptr, cptr_at_idx(summand, i));
    }

    return FT_ERR_NO_ERROR;
}


ft_error ft_sub_subset(
    ft_ge25519_array res,
    const size_t *idxs, size_t n_idxs,
    const_ft_ge25519_array summand)
{
    ft_error err = FT_ERR_NO_ERROR;

    if (summand->n_elts != n_idxs)
        return FT_ERR_SIZE_MISMATCH;

    for (size_t i = 0; i < n_idxs; ++i) {
        ge25519_p3 *ptr;
        err = ptr_at_idx_chkd(&ptr, res, idxs[i]);
        if (err) break;
        ge25519_subfrom(ptr, cptr_at_idx(summand, i));
    }

    return err;
}

ft_error ft_sparse_matmul(
    ft_ge25519_array *array,
    const size_t **idx_rows, size_t *idx_row_lens, size_t n_rows,
    const_ft_ge25519_array vec)
{
    ft_ge25519_array res;
    ft_error err = FT_ERR_NO_ERROR;

    err = ft_allocate_array(&res, n_rows);
    if (err) return err;

    for (size_t i = 0; i < n_rows; ++i) {
        const size_t *idxs = idx_rows[i];
        size_t n_cols = idx_row_lens[i];
        ge25519_p3 *sum = ptr_at_idx(res, i);

        if (n_cols == 0) {
            ge25519_zero(sum);
            continue;
        }

        const ge25519_p3 *elt;
        err = cptr_at_idx_chkd(&elt, vec, idxs[0]);
        if (err) goto cleanup;
        memcpy(sum, elt, sizeof(*elt));

        for (size_t j = 1; j < n_cols; ++j) {
            err = cptr_at_idx_chkd(&elt, vec, idxs[j]);
            if (err) goto cleanup;
            ge25519_addto(sum, elt);
        }
    }

cleanup:
    if (err)
        ft_array_free(res);
    else
        *array = res;
    return err;
}
