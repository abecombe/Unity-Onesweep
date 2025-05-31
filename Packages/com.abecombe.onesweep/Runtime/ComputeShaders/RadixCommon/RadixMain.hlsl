// This file includes code adapted from GPUSorting by Thomas Smith
// https://github.com/b0nes164/GPUSorting
// Licensed under the MIT License
// Modified by abecombe, 2025

#ifndef CS_RADIX_COMMON_RADIX_MAIN_HLSL
#define CS_RADIX_COMMON_RADIX_MAIN_HLSL

//#pragma use_dxc
//#pragma require wavebasic
//#pragma require waveballot
//#pragma multi_compile WAVE_SIZE_32 WAVE_SIZE_64
//#pragma multi_compile KEY_TYPE_UINT KEY_TYPE_INT KEY_TYPE_FLOAT
//#pragma multi_compile SORTING_ORDER_ASCENDING SORTING_ORDER_DESCENDING
//#pragma multi_compile KEY_ONLY KEY_PAYLOAD

#include "../Common/Constants.hlsl"
#include "../Common/Wave.hlsl"
#include "Radix.hlsl"

#if !defined(KEY_ONLY) && !defined(KEY_PAYLOAD)
#define KEY_ONLY
#endif

#define THREADS_PER_GROUP (RADIX_BASE)

#define ITEMS_PER_THREAD (15u) // ITEMS_PER_THREAD * THREADS_PER_GROUP + RADIX_BASE = SHARED_MEMORY_MAX_SIZE
#define ITEMS_PER_GROUP (THREADS_PER_GROUP * ITEMS_PER_THREAD)
#define ITEMS_PER_WAVE (WAVE_SIZE * ITEMS_PER_THREAD)

StructuredBuffer<DATA_TYPE> key_in_buffer;
RWStructuredBuffer<DATA_TYPE> key_out_buffer;
#if defined(KEY_PAYLOAD)
StructuredBuffer<uint> payload_in_buffer;
RWStructuredBuffer<uint> payload_out_buffer;
#endif

uint current_pass_radix_shift;

// Allocating too much shared memory per thread group may limit the number of concurrent thread groups.
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#feature-availability
#define SHARED_MEMORY_MAX_SIZE (4096u) // 16KB
groupshared uint group_shared[SHARED_MEMORY_MAX_SIZE];

struct ItemsArray
{
    uint data[ITEMS_PER_THREAD];
};
struct ItemsArray16bit
{
    min16uint data[ITEMS_PER_THREAD];
};

/**
 * \brief Computes the start index into the key and payload buffers for the current thread.
 */
inline uint GetThreadKeyPayloadStartIndex(in uint group_thread_id, in uint group_id)
{
    return LANE_INDEX + WAVE_INDEX(group_thread_id) * ITEMS_PER_WAVE + group_id * ITEMS_PER_GROUP;
}

/**
 * \brief Loads keys into local storage for the current thread.
 */
inline ItemsArray LoadKeys(in uint group_thread_id, in uint group_id, in uint sort_count, in uint group_count)
{
    ItemsArray keys;
    if (group_id < group_count - 1u)
    {
        [unroll(ITEMS_PER_THREAD)]
        for (uint i = 0u, key_index = GetThreadKeyPayloadStartIndex(group_thread_id, group_id); i < ITEMS_PER_THREAD; i++, key_index += WAVE_SIZE)
        {
            keys.data[i] = from_original_key_to_sorting_key(key_in_buffer[key_index]);
        }
    }
    else
    {
        [unroll(ITEMS_PER_THREAD)]
        for (uint i = 0u, key_index = GetThreadKeyPayloadStartIndex(group_thread_id, group_id); i < ITEMS_PER_THREAD; i++, key_index += WAVE_SIZE)
        {
            if (key_index < sort_count)
                keys.data[i] = from_original_key_to_sorting_key(key_in_buffer[key_index]);
            else
                keys.data[i] = ALL_BITS_SET;
        }
    }
    return keys;
}

/**
 * \brief Computes a bit mask indicating which lanes within a wave belong to the same bucket.
 * \return A bit mask where set bits represent lanes that are in the same bucket.
 *
 * \note Adapted from "GPU Multisplit": Algorithm 3 - Warp-level local offset computation.
 *       https://madalgo.au.dk/fileadmin/madalgo/OA_PDF_s/C417.pdf
 */
inline WAVE_MASK_TYPE ComputeSameBucketLaneBitMaskInWave(in uint radix_digit)
{
    WAVE_MASK_TYPE same_bucket_lane_bit_mask_in_wave = ALL_BITS_SET;
    [unroll(RADIX_BITS)]
    for (uint i = 0u; i < RADIX_BITS; i++)
    {
        const bool bit_flag = (radix_digit >> i) & 1u;
        const WAVE_MASK_TYPE bit_mask_in_wave = WAVE_ACTIVE_BALLOT(bit_flag);
        same_bucket_lane_bit_mask_in_wave &= (bit_flag ? NO_BITS_SET : ALL_BITS_SET) ^ bit_mask_in_wave;
    }
    return same_bucket_lane_bit_mask_in_wave;
}

/**
 * \brief Computes the number of bits set before the current lane (prefix count),
 *        and the total number of bits set in the wave.
 */
inline void ComputePrefixTotalBitCountInWave(in WAVE_MASK_TYPE bit_mask, out uint prefix_bit_count, out uint total_bit_count)
{
#if defined(WAVE_SIZE_32)
    const uint lane_mask = (1u << LANE_INDEX) - 1u; // e.g., 00000000, 00000001, 00000011, 00000111, ... (grows with each lane index)
    // The following code may cause undefined behavior if LANE_INDEX is 0:
    // const uint lane_mask = 0xffffffffu >> (32u - LANE_INDEX);
    prefix_bit_count = countbits(bit_mask & lane_mask);
    total_bit_count = countbits(bit_mask);
#elif defined(WAVE_SIZE_64)
    uint2 lane_mask = NO_BITS_SET;
    if (LANE_INDEX < 32u)
    {
        lane_mask.x = (1u << LANE_INDEX) - 1u;
    }
    else
    {
        lane_mask.x = ALL_BITS_SET;
        lane_mask.y = (1u << (LANE_INDEX - 32u)) - 1u;
    }
    uint2 count_temp = countbits(bit_mask & lane_mask);
    prefix_bit_count = count_temp.x + count_temp.y;
    count_temp = countbits(bit_mask);
    total_bit_count = count_temp.x + count_temp.y;
#endif
}

/**
 * \brief Computes wave-level local offsets.
 *
 * \note Writes to group_shared memory:
 *       [bucket_id (0–255) + WAVE_INDEX (0–7 or 0–3) * RADIX_BASE] stores the total count of each bucket ID within a wave.
 */
inline ItemsArray16bit ComputeWaveLevelLocalOffsets(in uint group_thread_id, in ItemsArray keys)
{
    // Initialize group_shared memory
    [unroll(RADIX_BASE / WAVE_SIZE)]
    for (uint i = LANE_INDEX + WAVE_INDEX(group_thread_id) * RADIX_BASE; i < (WAVE_INDEX(group_thread_id) + 1u) * RADIX_BASE; i += WAVE_SIZE)
    {
        group_shared[i] = 0u;
    }

    ItemsArray16bit offsets;
    [unroll(ITEMS_PER_THREAD)]
    for (uint i = 0u; i < ITEMS_PER_THREAD; i++)
    {
        const uint radix_digit = get_radix_digit(keys.data[i], current_pass_radix_shift);

        const WAVE_MASK_TYPE same_bucket_lane_bit_mask_in_wave = ComputeSameBucketLaneBitMaskInWave(radix_digit);

        uint same_bucket_lane_prefix_count;
        uint same_bucket_lane_total_count;
        ComputePrefixTotalBitCountInWave(same_bucket_lane_bit_mask_in_wave, same_bucket_lane_prefix_count, same_bucket_lane_total_count);

        const uint shared_memory_address = radix_digit + WAVE_INDEX(group_thread_id) * RADIX_BASE;
        const uint previous_loop_same_bucket_lane_total_count = group_shared[shared_memory_address];
        offsets.data[i] = previous_loop_same_bucket_lane_total_count + same_bucket_lane_prefix_count;
        if (same_bucket_lane_prefix_count == 0)
            group_shared[shared_memory_address] += same_bucket_lane_total_count;
    }
    return offsets;
}

/**
 * \brief Computes the total count of each bucket ID in the thread group and performs an exclusive prefix scan.
 *
 * \note Writes to group_shared memory:
 *       [bucket_id (0–255) + WAVE_INDEX (1–7 or 1–3) * RADIX_BASE (256)] stores the exclusive prefix sum of each bucket ID across waves.
 */
inline uint ExclusiveScanBucketCountsInGroup(in uint bucket_id)
{
    uint bucket_total_count_in_group = group_shared[bucket_id];
    [unroll(WAVE_COUNT_IN_GROUP(THREADS_PER_GROUP) - 1u)]
    for (uint i = bucket_id + RADIX_BASE; i < WAVE_COUNT_IN_GROUP(THREADS_PER_GROUP) * RADIX_BASE; i += RADIX_BASE)
    {
        bucket_total_count_in_group += group_shared[i];
        group_shared[i] = bucket_total_count_in_group - group_shared[i];
    }
    return bucket_total_count_in_group;
}

/**
 * \brief Computes the exclusive prefix sum of bucket counts across the group.
 *
 * \note Writes to group_shared memory:
 *       [bucket_id (0-255)] stores the exclusive prefix sum for each bucket.
 */
inline void ScanBucketTotalCountExclusiveToSharedMemory(in uint group_thread_id, in uint bucket_total_count_in_group) // group_thread_id = bucket_id
{
    bucket_total_count_in_group += WavePrefixSum(bucket_total_count_in_group); // inclusive scan
    // ((LANE_INDEX + 1u) & WAVE_SIZE_MASK) + (group_thread_id & ~WAVE_SIZE_MASK) means
    // 1, 2, .. , 31, 0, 33, 34, .. , 63, 32, 65, 66, .. , 95, 64, ...
    group_shared[((LANE_INDEX + 1u) & WAVE_SIZE_MASK) + (group_thread_id & ~WAVE_SIZE_MASK)] = bucket_total_count_in_group;

    GroupMemoryBarrierWithGroupSync();

    if (group_thread_id < WAVE_COUNT_IN_GROUP(THREADS_PER_GROUP))
        group_shared[group_thread_id * WAVE_SIZE] = WavePrefixSum(group_shared[group_thread_id * WAVE_SIZE]);

    GroupMemoryBarrierWithGroupSync();

    if (LANE_INDEX != 0u && group_thread_id >= WAVE_SIZE)
        group_shared[group_thread_id] += group_shared[WAVE_INDEX(group_thread_id) * WAVE_SIZE];
}

/**
 * \brief Combines per-wave local offsets to produce final group-level local offsets.
 */
inline void UpdateLocalOffsetsFromWaveToGroupLevel(in uint group_thread_id, in ItemsArray keys, inout ItemsArray16bit offsets)
{
    if (group_thread_id >= WAVE_SIZE) // WAVE_INDEX(group_thread_id) >= 1
    {
        const uint wave_offset = WAVE_INDEX(group_thread_id) * RADIX_BASE;
        [unroll(ITEMS_PER_THREAD)]
        for (uint i = 0; i < ITEMS_PER_THREAD; i++)
        {
            const uint bucket_id = get_radix_digit(keys.data[i], current_pass_radix_shift);
            offsets.data[i] += group_shared[bucket_id + wave_offset] + group_shared[bucket_id];
        }
    }
    else
    {
        [unroll(ITEMS_PER_THREAD)]
        for (uint i = 0; i < ITEMS_PER_THREAD; i++)
            offsets.data[i] += group_shared[get_radix_digit(keys.data[i], current_pass_radix_shift)];
    }
}

/**
 * \brief Writes sorted keys to shared memory using their computed new IDs.
 *
 * \note Writes to group_shared memory:
 *       [items (0-ITEMS_PER_GROUP - 1) + RADIX_BASE(256)] stores the sorted keys.
 */
inline void SortKeysInGroup(in ItemsArray16bit new_ids, in ItemsArray keys)
{
    [unroll(ITEMS_PER_THREAD)]
    for (uint i = 0; i < ITEMS_PER_THREAD; i++)
    {
        group_shared[new_ids.data[i] + RADIX_BASE] = keys.data[i];
    }
}

#if defined(KEY_ONLY)
/**
 * \brief Writes sorted keys from shared memory to the global buffer.
 */
inline void StoreKeys(in uint group_thread_id, in uint group_id, in uint sort_count, in uint group_count)
{
    if (group_id < group_count - 1u)
    {
        [unroll(ITEMS_PER_THREAD)]
        for (uint i = 0, j = group_thread_id; i < ITEMS_PER_THREAD; i++, j += THREADS_PER_GROUP)
        {
            const uint key = group_shared[j + RADIX_BASE];
            key_out_buffer[group_shared[get_radix_digit(key, current_pass_radix_shift)] + j] = from_sorting_key_to_original_key(key);
        }
    }
    else
    {
        const uint group_sort_count = sort_count - (group_count - 1u) * ITEMS_PER_GROUP;
        [unroll(ITEMS_PER_THREAD)]
        for (uint i = 0, j = group_thread_id; i < ITEMS_PER_THREAD; i++, j += THREADS_PER_GROUP)
        {
            if (j < group_sort_count)
            {
                const uint key = group_shared[j + RADIX_BASE];
                key_out_buffer[group_shared[get_radix_digit(key, current_pass_radix_shift)] + j] = from_sorting_key_to_original_key(key);
            }
        }
    }
}

#elif defined(KEY_PAYLOAD)
/**
 * \brief Writes sorted keys from shared memory to the global buffer and returns their final indices.
 */
inline ItemsArray StoreKeys(in uint group_thread_id, in uint group_id, in uint sort_count, in uint group_count)
{
    ItemsArray out_buffer_indices;
    if (group_id < group_count - 1u)
    {
        [unroll(ITEMS_PER_THREAD)]
        for (uint i = 0, j = group_thread_id; i < ITEMS_PER_THREAD; i++, j += THREADS_PER_GROUP)
        {
            const uint key = group_shared[j + RADIX_BASE];
            out_buffer_indices.data[i] = group_shared[get_radix_digit(key, current_pass_radix_shift)] + j;
            key_out_buffer[out_buffer_indices.data[i]] = from_sorting_key_to_original_key(key);
        }
    }
    else
    {
        const uint group_sort_count = sort_count - (group_count - 1u) * ITEMS_PER_GROUP;
        [unroll(ITEMS_PER_THREAD)]
        for (uint i = 0, j = group_thread_id; i < ITEMS_PER_THREAD; i++, j += THREADS_PER_GROUP)
        {
            if (j < group_sort_count)
            {
                const uint key = group_shared[j + RADIX_BASE];
                out_buffer_indices.data[i] = group_shared[get_radix_digit(key, current_pass_radix_shift)] + j;
                key_out_buffer[out_buffer_indices.data[i]] = from_sorting_key_to_original_key(key);
            }
        }
    }
    return out_buffer_indices;
}

/**
 * \brief Loads original payloads and writes them to shared memory at their new computed positions.
 *
 * \note Writes to group_shared memory:
 *       [items (0-ITEMS_PER_GROUP - 1) + RADIX_BASE(256)] stores the sorted payloads.
 */
inline void WriteSortedPayloadsToSharedMemory(in uint group_thread_id, in uint group_id, in uint sort_count, in uint group_count, in ItemsArray16bit new_ids)
{
    if (group_id < group_count - 1u)
    {
        [unroll(ITEMS_PER_THREAD)]
        for (uint i = 0u, payload_index = GetThreadKeyPayloadStartIndex(group_thread_id, group_id); i < ITEMS_PER_THREAD; i++, payload_index += WAVE_SIZE)
        {
            group_shared[new_ids.data[i] + RADIX_BASE] = payload_in_buffer[payload_index];
        }
    }
    else
    {
        [unroll(ITEMS_PER_THREAD)]
        for (uint i = 0u, payload_index = GetThreadKeyPayloadStartIndex(group_thread_id, group_id); i < ITEMS_PER_THREAD; i++, payload_index += WAVE_SIZE)
        {
            if (payload_index < sort_count)
                group_shared[new_ids.data[i] + RADIX_BASE] = payload_in_buffer[payload_index];
        }
    }
}

/**
 * \brief Writes sorted payloads from shared memory to the global buffer.
 */
inline void StorePayloads(in uint group_thread_id, in uint group_id, in uint sort_count, in uint group_count, in ItemsArray out_buffer_indices)
{
    if (group_id < group_count - 1u)
    {
        [unroll(ITEMS_PER_THREAD)]
        for (uint i = 0, j = group_thread_id; i < ITEMS_PER_THREAD; i++, j += THREADS_PER_GROUP)
        {
            payload_out_buffer[out_buffer_indices.data[i]] = group_shared[j + RADIX_BASE];
        }
    }
    else
    {
        const uint group_sort_count = sort_count - (group_count - 1u) * ITEMS_PER_GROUP;
        [unroll(ITEMS_PER_THREAD)]
        for (uint i = 0, j = group_thread_id; i < ITEMS_PER_THREAD; i++, j += THREADS_PER_GROUP)
        {
            if (j < group_sort_count)
            {
                payload_out_buffer[out_buffer_indices.data[i]] = group_shared[j + RADIX_BASE];
            }
        }
    }
}
#endif

#endif /* CS_RADIX_COMMON_RADIX_MAIN_HLSL */