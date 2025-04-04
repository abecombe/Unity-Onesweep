#ifndef CS_ONE_SWEEP_RADIX_SORT_DISPATCH_UTILS_HLSL
#define CS_ONE_SWEEP_RADIX_SORT_DISPATCH_UTILS_HLSL

//#pragma multi_compile USE_DIRECT_DISPATCH USE_INDIRECT_DISPATCH

#if !defined(USE_DIRECT_DISPATCH) && !defined(USE_INDIRECT_DISPATCH)
#define USE_DIRECT_DISPATCH
#endif

#if defined(USE_DIRECT_DISPATCH)
uint sort_count;
uint group_count;
#elif defined(USE_INDIRECT_DISPATCH)
ByteAddressBuffer sort_count_group_count_buffer;
inline void get_sort_count_group_count(out uint sort_count, out uint group_count)
{
    const uint2 sort_count_group_count = sort_count_group_count_buffer.Load2(0);
    sort_count = sort_count_group_count.x;
    group_count = sort_count_group_count.y;
}
inline uint get_sort_count()
{
    return sort_count_group_count_buffer.Load(0);
}
inline uint get_group_count()
{
    return sort_count_group_count_buffer.Load(4);
}
#endif

#endif /* CS_ONE_SWEEP_RADIX_SORT_DISPATCH_UTILS_HLSL */