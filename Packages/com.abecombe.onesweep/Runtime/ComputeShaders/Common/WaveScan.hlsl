#ifndef CS_COMMON_WAVE_SCAN_HLSL
#define CS_COMMON_WAVE_SCAN_HLSL

/**
 * \brief Exclusively scans the Wave8 totals stored at the start of each wave's shared-memory range.
 *
 * \note The first Wave8 processes eight wave totals per iteration. On return,
 *       group_shared[wave_index * 8] contains the sum of all preceding wave totals.
 */
inline void ExclusiveScanWaveTotalsWave8(in uint group_thread_id)
{
    if (group_thread_id >= 8u)
        return;

    uint reduction = 0u;
    [unroll(THREADS_PER_GROUP / 64u)]
    for (uint wave_index = LANE_INDEX; wave_index < THREADS_PER_GROUP / 8u; wave_index += 8u)
    {
        const uint wave_start = wave_index * 8u;
        const uint wave_total = group_shared[wave_start];
        group_shared[wave_start] = reduction + WavePrefixSum(wave_total);
        reduction += WaveActiveSum(wave_total);
    }
}

#endif /* CS_COMMON_WAVE_SCAN_HLSL */