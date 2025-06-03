#ifndef CS_TRADITIONAL_RADIX_SORT_UTILS_HLSL
#define CS_TRADITIONAL_RADIX_SORT_UTILS_HLSL

uint scanned_bucket_count_buffer_group_count;

inline uint get_bucket_count_buffer_address(in uint bucket_id, in uint group_id, in uint radix_base)
{
    return group_id * radix_base + bucket_id;
}

inline uint get_scanned_bucket_count_buffer_address(in uint bucket_id, in uint group_id)
{
    return bucket_id * scanned_bucket_count_buffer_group_count + group_id;
}

#endif /* CS_TRADITIONAL_RADIX_SORT_UTILS_HLSL */