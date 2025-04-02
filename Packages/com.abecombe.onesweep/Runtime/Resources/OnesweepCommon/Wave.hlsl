#ifndef CS_ONE_SWEEP_COMMON_WAVE_HLSL
#define CS_ONE_SWEEP_COMMON_WAVE_HLSL

//#pragma use_dxc
//#pragma require wavebasic
//#pragma require waveballot
//#pragma multi_compile WAVE_SIZE_32 WAVE_SIZE_64

#if !defined(WAVE_SIZE_32) && !defined(WAVE_SIZE_64)
#define WAVE_SIZE_32
#endif

#if defined(WAVE_SIZE_32)
#define WAVE_SIZE (32u)
#define WAVE_MASK_TYPE uint
#define LANE_INDEX (WaveGetLaneIndex()) // 0, 1, 2, .. ,31, 0, 1, 2, .. ,31, ...
#define WAVE_INDEX(group_thread_id) (group_thread_id >> 5u) // 0, 0, 0, .. ,0, 1, 1, 1, .. ,1, ...
#define WAVE_COUNT_IN_GROUP(threads_per_group) (threads_per_group >> 5u) // threads_per_group / 32
#define WAVE_ACTIVE_BALLOT(bool_value) (WaveActiveBallot(bool_value).x)

#elif defined(WAVE_SIZE_64)
#define WAVE_SIZE (64u)
#define WAVE_MASK_TYPE uint2
#define LANE_INDEX (WaveGetLaneIndex()) // 0, 1, 2, .. ,63, 0, 1, 2, .. ,63, ...
#define WAVE_INDEX(group_thread_index) (group_thread_index >> 6u) // 0, 0, 0, .. ,0, 1, 1, 1, .. ,1, ...
#define WAVE_COUNT_IN_GROUP(threads_per_group) (threads_per_group >> 6u) // threads_per_group / 64
#define WAVE_ACTIVE_BALLOT(bool_value) (WaveActiveBallot(bool_value).xy)

#endif

#define WAVE_SIZE_MASK (WAVE_SIZE - 1u)

#endif /* CS_ONE_SWEEP_COMMON_WAVE_HLSL */