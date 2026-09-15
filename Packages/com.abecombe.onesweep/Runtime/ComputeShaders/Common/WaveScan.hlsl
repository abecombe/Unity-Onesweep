#ifndef CS_COMMON_WAVE_SCAN_HLSL
#define CS_COMMON_WAVE_SCAN_HLSL

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