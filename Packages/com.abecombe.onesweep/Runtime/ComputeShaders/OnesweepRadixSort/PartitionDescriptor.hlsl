#ifndef CS_ONE_SWEEP_RADIX_SORT_PARTITION_DESCRIPTOR_HLSL
#define CS_ONE_SWEEP_RADIX_SORT_PARTITION_DESCRIPTOR_HLSL

#include "../OnesweepCommon/PartitionDescriptor.hlsl"
#include "../RadixCommon/Radix.hlsl"

inline uint get_partition_descriptor_buffer_address(in uint bucket_id, in uint group_id, in uint group_size, in uint radix_step)
{
    return RADIX_BASE * (group_size * radix_step + group_id) + bucket_id;
}

#endif /* CS_ONE_SWEEP_RADIX_SORT_PARTITION_DESCRIPTOR_HLSL */