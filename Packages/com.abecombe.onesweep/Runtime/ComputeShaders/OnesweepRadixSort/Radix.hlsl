#ifndef CS_ONE_SWEEP_RADIX_SORT_RADIX_HLSL
#define CS_ONE_SWEEP_RADIX_SORT_RADIX_HLSL

//#pragma multi_compile KEY_TYPE_UINT KEY_TYPE_INT KEY_TYPE_FLOAT
//#pragma multi_compile SORTING_ORDER_ASCENDING SORTING_ORDER_DESCENDING

#if !defined(KEY_TYPE_UINT) && !defined(KEY_TYPE_INT) && !defined(KEY_TYPE_FLOAT)
#define KEY_TYPE_UINT
#endif
#if !defined(SORTING_ORDER_ASCENDING) && !defined(SORTING_ORDER_DESCENDING)
#define SORTING_ORDER_ASCENDING
#endif

#define RADIX_BITS (8u)
#define RADIX_BASE (1u << RADIX_BITS) // 1 << 8 = 256
#define RADIX_BASE_MASK (RADIX_BASE - 1u) // 256 - 1 = 255
#define RADIX_STEP_COUNT (4u)
#define CURRENT_RADIX_STEP(current_pass_radix_shift) (current_pass_radix_shift >> 3u) // 0, 1, 2, 3

// input key struct
#if defined(KEY_TYPE_UINT)
#define DATA_TYPE uint
#elif defined(KEY_TYPE_INT)
#define DATA_TYPE int
#elif defined(KEY_TYPE_FLOAT)
#define DATA_TYPE float
#endif

// Radix Tricks
// http://stereopsis.com/radix.html
inline uint int_to_uint_for_sorting(in int i)
{
    return asuint(i ^ 0x80000000);
}
inline int inv_int_to_uint_for_sorting(in uint u)
{
    return asint(u ^ 0x80000000);
}
inline uint float_to_uint_for_sorting(in float f)
{
    uint mask = -(int)(asuint(f) >> 31) | 0x80000000;
    return asuint(f) ^ mask;
}
inline float inv_float_to_uint_for_sorting(in uint u)
{
    uint mask = (u >> 31) - 1 | 0x80000000;
    return asfloat(u ^ mask);
}
inline uint from_original_key_to_sorting_key(in DATA_TYPE original_key)
{
#if defined(KEY_TYPE_UINT)
    uint sorting_key = original_key;
#elif defined(KEY_TYPE_INT)
    uint sorting_key = int_to_uint_for_sorting(original_key);
#elif defined(KEY_TYPE_FLOAT)
    uint sorting_key = float_to_uint_for_sorting(original_key);
#endif
#if defined(SORTING_ORDER_DESCENDING)
    sorting_key = ~sorting_key;
#endif
    return sorting_key;
}
inline DATA_TYPE from_sorting_key_to_original_key(in uint sorting_key)
{
#if defined(SORTING_ORDER_DESCENDING)
    sorting_key = ~sorting_key;
#endif
#if defined(KEY_TYPE_UINT)
    return sorting_key;
#elif defined(KEY_TYPE_INT)
    return inv_int_to_uint_for_sorting(sorting_key);
#elif defined(KEY_TYPE_FLOAT)
    return inv_float_to_uint_for_sorting(sorting_key);
#endif
}

inline uint get_radix_digit(in uint sorting_key, in uint radix_shift)
{
    return (sorting_key >> radix_shift) & RADIX_BASE_MASK;
}

#endif /* CS_ONE_SWEEP_RADIX_SORT_RADIX_HLSL */