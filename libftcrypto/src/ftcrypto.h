/* =====================================
 *
 * Copyright (c) 2023, AUSTRAC Australian Government
 * All rights reserved.
 *
 * Licensed under BSD 3 clause license
 *
 */

#ifndef FT_GE25519_ARRAY
#define FT_GE25519_ARRAY

#include <stdint.h>
#include <stdlib.h>

/**
 * Number of bytes required to hold an Ed25519int.
 */
#define FT_ED25519_SCALARBYTES 32

/**
 * Number of bytes required to hold an Ed25519 point in folded
 * representation.
 */
#define FT_FOLDED_POINT_BYTES 32

/**
 * Number of bytes required to hold an Ed25519 point in uncompressed
 * representation (happens to be 160).
 */
#define FT_EXTENDED_POINT_BYTES (sizeof(ge25519_p3))

/**
 * Number of bytes required to hold an Ed25519 point in affine
 * representation (happens to be 64, 32 for each of the x and y
 * coordinates).
 */
#define FT_AFFINE_POINT_BYTES (2 * FT_ED25519_SCALARBYTES)

/**
 * Opaque pointer to an array of Ed25519 points (referred to as
 * ge25519's ("ge" = "group element")).
 */
struct ft_ge25519_array_s;
typedef struct ft_ge25519_array_s *ft_ge25519_array;
typedef struct ft_ge25519_array_s const *const_ft_ge25519_array;

/**
 * Error codes.
 */
typedef int ft_error;
#define FT_ERR_NO_ERROR 0
#define FT_ERR_OUT_OF_MEMORY 1
#define FT_ERR_SIZE_MISMATCH 2
#define FT_ERR_BAD_SIZE 3
#define FT_ERR_OUT_OF_BOUNDS 4
#define FT_ERR_INVALID_POINT 5
#define FT_ERR_SODIUM_INIT 6
#define FT_ERR_INSUFFICIENT_SPACE 7
#define FT_ERR_BAD_CIPHERTEXT 8
#define FT_ERR_INTERNAL_LIBRARY_ERROR 9
#define FT_ERR_INSUFFICIENT_DEVICE_OR_DRIVER 10
#define FT_ERR_MAX_ERR 10

const char *ft_error_str(ft_error err);

/**
 * Initialise the ft_crypto library. Must be called before any other
 * functions are called.
 */
ft_error ft_crypto_init();

/**
 * Return the available and total memory for the current GPU device.
 */
ft_error ft_crypto_device_memory(size_t *free_mem, size_t *total_mem);

/**
 * Create a new array containing n_elts copies of scalar (which should
 * be FT_ED25519_SCALARBYTES bytes value in little-endian order less
 * than the group order, or NULL to indicate that the array should be
 * initialised to zeros).
 *
 * array must eventually be deallocated with ft_array_free().
 */
ft_error ft_array_init_scalar(
    ft_ge25519_array *array, size_t n_elts,
    const uint8_t *scalar);

/**
 * Create a new array containing n_elts copies of src[idx].
 *
 * array must eventually be deallocated with ft_array_free().
 */
ft_error ft_array_init_point(
    ft_ge25519_array *array, size_t n_elts,
    const_ft_ge25519_array src, size_t idx);

/**
 * Create a new array containing the n_idxs elements src[idx[i]] for
 * i = 0,...,n_idxs-1.
 *
 * array must eventually be deallocated with ft_array_free().
 */
ft_error ft_array_get_subset(
    ft_ge25519_array *array,
    const size_t *idxs, size_t n_idxs,
    const_ft_ge25519_array src);

/**
 * Create a new array containing the ceiling((stop - start)/step)
 * elements src[start + i * step] for i = 0, ..., ceiling((stop - start)/step)-1.
 *
 * array must eventually be deallocated with ft_array_free().
 */
ft_error ft_array_get_range(
    ft_ge25519_array *array,
    size_t start, size_t stop, size_t step,
    const_ft_ge25519_array src);

/**
 * Assign the ith element of src to the idx[i]th element of array.
 */
ft_error ft_array_assign_to_subset(
    ft_ge25519_array array,
    const size_t *idxs, size_t n_idxs,
    const_ft_ge25519_array src);

/**
 * Assign zero to the idx[i]th element of array.
 */
ft_error ft_array_assign_zero_to_subset(
    ft_ge25519_array array,
    const size_t *idxs, size_t n_idxs);

/**
 * Assign the ith element of src to the start + i*step element of array.
 */
ft_error ft_array_assign_to_range(
    ft_ge25519_array array,
    size_t start, size_t stop, size_t step,
    const_ft_ge25519_array src);

/**
 * Delete element idx[i] from array for i = 0, ..., n_idxs-1.
 */
ft_error ft_array_delete_subset(
    ft_ge25519_array array,
    const size_t *idxs, size_t n_idxs);

/**
 * Delete element start + i*step from array for i = 0, ..., n_idxs-1.
 */
ft_error ft_array_delete_range(
    ft_ge25519_array array,
    size_t start, size_t stop, size_t step);

/**
 * Compress and serialise array into bytes starting at mem. n_bytes
 * must be less than or equal to the number of bytes available at
 * mem. The number of bytes written will be ft_array_length(array) *
 * FT_FOLDED_POINT_BYTES.
 *
 * The points are serialised into folded representation which is
 * unique and can be used for equality testing.
 */
ft_error ft_array_to_bytes_folded(
    uint8_t *mem, size_t n_bytes,
    const_ft_ge25519_array array);

/**
 * Serialise array into affine bytes starting at mem. n_bytes
 * must be less than or equal to the number of bytes available at
 * mem. The number of bytes written will be ft_array_length(array) *
 * FT_AFFINE_POINT_BYTES.
 *
 * The points are serialised into affine representation which is
 * unique and can be used for equality testing.
 */
ft_error ft_array_to_bytes_affine(
    uint8_t *mem, size_t n_bytes,
    const_ft_ge25519_array array);

/**
 * Serialise array into bytes starting at mem. n_bytes must be less
 * than or equal to the number of bytes available at mem. The number
 * of bytes written will be ft_array_length(array) * FT_EXTENDED_POINT_BYTES.
 */
ft_error ft_array_to_bytes(
    uint8_t *mem, size_t n_bytes,
    const_ft_ge25519_array array);

/**
 * Deserialise from compressed bytes starting at mem into new array
 * 'array'.  n_bytes must be exactly the number of bytes that have
 * been serialised. The total number of ge25519s read will be n_bytes
 * / FT_FOLDED_POINT_BYTES.
 *
 * The points in mem must be serialised as folded representations by a
 * function such as ft_array_to_bytes_folded() above. This function
 * implicitly verifies that the given points are indeed in the correct
 * subgroup on the curve so no additional check is necessary. If the
 * conversion failed for any reason, then the value of *res will be
 * the index of the last failed conversion; if there were no errors
 * then *res = -1.
 *
 * array must eventually be deallocated with ft_array_free().
 */
ft_error ft_array_from_bytes_folded(
    int64_t *res,
    ft_ge25519_array *array,
    const uint8_t *mem, size_t n_bytes);

/**
 * Deserialise from affine bytes starting at mem into new array
 * 'array'.  n_bytes must be exactly the number of bytes that have
 * been serialised. The total number of ge25519s read will be
 * n_bytes/FT_AFFINE_POINT_BYTES.
 *
 * The points in mem must be serialised as folded representations by
 * the function ft_array_to_bytes_affine() above. THIS FUNCTION DOES
 * NOT VERIFY THAT THE GIVEN POINTS ARE VALID; if they come from
 * uncertain provenance, you must check the result of this function
 * with ft_array_validate().
 *
 * array must eventually be deallocated with ft_array_free().
 */
ft_error ft_array_from_bytes_affine(
    ft_ge25519_array *array,
    const uint8_t *mem, size_t n_bytes);

/**
 * Deserialise from bytes starting at mem into new array 'array'.
 * n_bytes must be exactly the number of bytes that have been
 * serialised. The total number of ge25519s read will be
 * n_bytes/FT_EXTENDED_POINT_BYTES.
 *
 * The points in mem must be serialised as extended representations
 * ONLY by the function ft_array_to_bytes() above. THIS FUNCTION DOES
 * NOT VERIFY THAT THE GIVEN POINTs ARE VALID; if they come from
 * uncertain provenance, you must check the result of this function
 * with ft_array_validate().
 *
 * array must eventually be deallocated with ft_array_free().
 */
ft_error ft_array_from_bytes(
    ft_ge25519_array *array,
    const uint8_t *mem, size_t n_bytes);

/**
 * Checks that each point of array is a valid point in the correct
 * subgroup of the curve. This should be called after
 * ft_array_from_bytes() or ft_array_from_bytes_affine() if a
 * potential attacker may have had access to the bytes before
 * ingestion.
 *
 * The return value is the largest index holding an invalid point, or
 * -1 if there were no invalid points.
 */
ft_error ft_array_validate(
    int64_t *res,
    const_ft_ge25519_array array);

/**
 * Given an array scalars of n_scalars non-negative integers in
 * little-endian order less than p=2^255-19 (i.e. each of size
 * FT_ED25519_SCALARBYTES = 32 bytes), return an array whose ith
 * element is scalars[i]*G where G is the conventional fixed generator
 * of the curve.
 *
 * The total size of scalars must be at least
 * n_scalars * FT_ED25519_SCALARBYTES bytes.
 */
ft_error ft_array_from_scalars(
    ft_ge25519_array *array,
    const uint8_t *scalars, size_t n_scalars);

/**
 * As for `ft_array_from_scalars` but the scalars are just `uint64_t`s.
 */
ft_error ft_array_from_small_scalars(
    ft_ge25519_array *array,
    const uint64_t *small_scalars, size_t n_scalars);

/**
 * Free the memory associated with array.
 */
void ft_array_free(ft_ge25519_array array);

/**
 * Return the number of elements in array.
 */
size_t ft_array_get_length(const_ft_ge25519_array array);

/**
 * Set the number of elements in array.
 *
 * Let len = ft_array_get_length(array) be the current length of
 * array. If newlen < len, then the last len - newlen elements of
 * array are removed; otherwise newlen - len copies of zero are added
 * to the end.
 *
 * This function will allocate a new array (and do the requisite
 * copying) if it is not possible to expand into array's reserved
 * space, or if newlen is less than half of the original array length.
 */
ft_error ft_array_set_length(ft_ge25519_array array, size_t newlen);

/**
 * Set iftrue[i] = (cond[i] != 0) ? iftrue[i] : iffalse[i]
 *
 * Obviously the lengths of the arrays must all be the same.
 */
ft_error ft_array_mux(
    ft_ge25519_array iftrue,
    const int64_t *cond,
    const_ft_ge25519_array iffalse);

/**
 * Set res[i] = 1 if array1[i] is in array2, and 0 otherwise.
 *
 * res must have the same length as array1. Performance is best if the
 * smaller of the two arrays is in the position of array1.
 */
ft_error ft_array_contains(
    int64_t *res,
    const_ft_ge25519_array array1,
    const_ft_ge25519_array array2);

/**
 * Set res[i] = res[i] + summand[i] for all i. The arrays must have
 * the same length.
 */
ft_error ft_add(
    ft_ge25519_array res,
    const_ft_ge25519_array summand);

/**
 * Set res[idxs[i]] = res[idxs[i]] + summand[i] for
 * i=0,...,n_idxs. Number of unique values in idxs must equal the
 * lenght of summand. Obviously the largest of the idxs[i] must be
 * less than the length of res.
 */
ft_error ft_reduce_isum(
    ft_ge25519_array res,
    const size_t *idxs, size_t n_idxs,
    const_ft_ge25519_array summand);

/**
 * Set res[i] = res[i] - summand[i] for all i. The arrays must have
 * the same length.
 */
ft_error ft_sub(
    ft_ge25519_array res,
    const_ft_ge25519_array summand);

/**
 * Set array[i] = scalar * array[i]. scalar must point to
 * FT_ED25519_SCALARBYTES bytes in little-endian order.
 *
 * NB: Must have scalar[FT_ED25519_SCALARBYTES - 1] <= 127, which is
 * always the case if scalar < 2^255-19.
 */
void ft_scale(
    ft_ge25519_array array,
    const uint8_t *scalar);

/**
 * Set array[i] = -array[i] for all i. 
 */
void ft_neg(ft_ge25519_array array);

/**
 * Set array[i] = scalar_i * array[i], where scalar_i is the 32 bytes
 * starting at i*FT_ED25519_SCALARBYTES in little-endian
 * order. scalars must point to n_scalars*FT_ED25519_SCALARBYTES bytes
 * total, and n_scalars must equal ft_array_length(array).
 *
 * NB: Must have scalar[i * FT_ED25519_SCALARBYTES - 1] <= 127 for all i,
 * which is always the case if scalar_i < 2^255-19.
 */
ft_error ft_mul(
    ft_ge25519_array array,
    const uint8_t *scalars, size_t n_scalars);

/**
 * Set res[i] = 1 if array[i] != 0, and 0 otherwise. res must point to
 * memory with at least ft_array_length(array)*sizeof(int64_t) bytes space.
 */
void ft_index(
    int64_t *res,
    const_ft_ge25519_array array);

/**
 * Set res[i] = 1 if array1[i] == array2[i], and 0 otherwise.
 *
 * array1 and array2 must have the same length. res must point to
 * memory with at least ft_array_length(array1)*sizeof(int64_t) bytes space.
 */
ft_error ft_equal(
    int64_t *res,
    const_ft_ge25519_array array1,
    const_ft_ge25519_array array2);

/**
 * Set res[i] = 1 if array1[i] != array2[i], and 0 otherwise.
 *
 * array1 and array2 must have the same length. res must point to
 * memory with at least ft_array_length(array1)*sizeof(int64_t) bytes space.
 */
ft_error ft_not_equal(
    int64_t *res,
    const_ft_ge25519_array array1,
    const_ft_ge25519_array array2);

/**
 * Replace array with its exclusive scan.
 *
 * Specifically, set array[j] <- \sum_{i=0}^{j-1} array[i] (where the
 * empty sum is interpreted as zero).
 */
ft_error ft_array_prescan(ft_ge25519_array array);

/**
 * Replace array with its inclusive scan.
 *
 * Specifically, set array[j] <- \sum_{i=0}^j array[i].
 */
ft_error ft_array_scan(ft_ge25519_array array);

/**
 * Replace array with the single element array containing the sum of
 * the elements of array.
 */
ft_error ft_array_reduce(ft_ge25519_array array);

#endif
