#ifndef CS_COMMON_WAVE_HLSL
#define CS_COMMON_WAVE_HLSL

/**
 * \brief Provides common macros for GPU wave-level operations.
 */

//#pragma use_dxc
//#pragma require wavebasic
//#pragma require waveballot
#define WAVE_SIZE (WaveGetLaneCount())
#define WAVE_SHIFT ((WAVE_SIZE == 32u) ? 5u : 6u)
#define WAVE_MASK_TYPE uint2
#define LANE_INDEX (WaveGetLaneIndex())
#define WAVE_INDEX(group_thread_index) ((group_thread_index) >> WAVE_SHIFT)
#define WAVE_COUNT_IN_GROUP(threads_per_group) ((threads_per_group) >> WAVE_SHIFT)
#define WAVE_ACTIVE_BALLOT(bool_value) (WaveActiveBallot(bool_value).xy)
#define WAVE_ACTIVE_LANE_MASK (WaveActiveBallot(true).xy)

#define WAVE_SIZE_MASK (WAVE_SIZE - 1u)

#endif /* CS_COMMON_WAVE_HLSL */