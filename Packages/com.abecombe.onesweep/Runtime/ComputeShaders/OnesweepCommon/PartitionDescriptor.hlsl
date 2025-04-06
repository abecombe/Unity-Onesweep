#ifndef CS_ONE_SWEEP_COMMON_PARTITION_DESCRIPTOR_HLSL
#define CS_ONE_SWEEP_COMMON_PARTITION_DESCRIPTOR_HLSL

#define FLAG uint
#define PARTITION_DESCRIPTOR uint
static const FLAG FLAG_INVALID   = 0u;
static const FLAG FLAG_AGGREGATE = 1u;
static const FLAG FLAG_PREFIX    = 2u;
static const uint partition_descriptor_flag_mask = 0x00000003u;
static const uint partition_descriptor_value_shift = 2u;

inline PARTITION_DESCRIPTOR create_partition_descriptor(in uint value, in FLAG flag)
{
    return (value << partition_descriptor_value_shift) | flag;
}
inline FLAG get_partition_descriptor_flag(in PARTITION_DESCRIPTOR partition_descriptor)
{
    return partition_descriptor & partition_descriptor_flag_mask;
}
inline uint get_partition_descriptor_value(in PARTITION_DESCRIPTOR partition_descriptor)
{
    return partition_descriptor >> partition_descriptor_value_shift;
}

#endif /* CS_ONE_SWEEP_COMMON_PARTITION_DESCRIPTOR_HLSL */