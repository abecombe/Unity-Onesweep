using System;
using UnityEngine;
using UnityEngine.Rendering;
using Object = UnityEngine.Object;

namespace Onesweep
{
    /// <summary>
    /// Implements the GPU "Onesweep" Radix Sort algorithm.
    /// Based on the paper "Onesweep: A Faster Least Significant Digit Radix Sort for GPUs" (https://arxiv.org/abs/2206.01784).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Onesweep algorithm utilizes complex GPU synchronization techniques aiming for high performance.
    /// While it can offer significant speed advantages, this approach also introduces certain considerations
    /// regarding stability and performance consistency.
    /// </para>
    /// <para>
    /// Users should be aware that sporadic application freezes (hangs) may occur due to potential GPU deadlocks.
    /// These issues can be influenced by various factors, including the specific GPU model, driver version,
    /// operating system, or conflicts with other concurrent GPU tasks. Furthermore, the actual sorting speed
    /// can be inconsistent and may vary significantly depending on the hardware, drivers, and data characteristics.
    /// </para>
    /// <para>
    /// For scenarios requiring greater stability and more predictable performance, using an alternative
    /// like <c>TraditionalSorter</c> is recommended. While <c>TraditionalSorter</c> might be slightly slower
    /// on average, it generally offers more consistent processing performance without the risk of deadlocks
    /// sometimes associated with the Onesweep technique.
    /// </para>
    /// <para>
    /// This implementation includes code adapted from the GPUSorting project by Thomas Smith
    /// (https://github.com/b0nes164/GPUSorting), licensed under the MIT License.
    /// </para>
    /// </remarks>
    public class OnesweepSorter : ISorter
    {
        #region Constants
        private const int MaxDispatchSize = 65535;

        private const int RadixBase = 256;
        private const int RadixStepCount = 4;

        private const int CountKernelItemsPerThread = 64;
        private const int CountKernelThreadsPerGroup = 128;
        private const int CountKernelItemsPerGroup = CountKernelItemsPerThread * CountKernelThreadsPerGroup;
        private const int SortKernelItemsPerThread = 15;
        private const int SortKernelThreadsPerGroup = RadixBase;
        private const int SortKernelItemsPerGroup = SortKernelItemsPerThread * SortKernelThreadsPerGroup;

        private const int PrecomputeKernelDispatchGroupSize = 1;
        private const int InitKernelDispatchGroupSize = 128;
        private static int CountKernelDispatchGroupSize(int sortCount) => (sortCount + CountKernelItemsPerGroup - 1) / CountKernelItemsPerGroup;
        private static int ScanKernelDispatchGroupSize => RadixStepCount;
        private static int SortKernelDispatchGroupSize(int sortCount) => (sortCount + SortKernelItemsPerGroup - 1) / SortKernelItemsPerGroup;
        #endregion

        #region Shader Property IDs
        private static readonly int SortCountID = Shader.PropertyToID("sort_count");
        private static readonly int GroupCountID = Shader.PropertyToID("group_count");
        private static readonly int BucketCountBufferID = Shader.PropertyToID("bucket_count_buffer");
        private static readonly int PartitionIndexBufferID = Shader.PropertyToID("partition_index_buffer");
        private static readonly int PartitionDescriptorBufferID = Shader.PropertyToID("partition_descriptor_buffer");
        private static readonly int KeyInBufferID = Shader.PropertyToID("key_in_buffer");
        private static readonly int KeyOutBufferID = Shader.PropertyToID("key_out_buffer");
        private static readonly int PayloadInBufferID = Shader.PropertyToID("payload_in_buffer");
        private static readonly int PayloadOutBufferID = Shader.PropertyToID("payload_out_buffer");
        private static readonly int CurrentPassRadixShiftID = Shader.PropertyToID("current_pass_radix_shift");
        private static readonly int CountKernelItemsPerGroupID = Shader.PropertyToID("count_kernel_items_per_group");
        private static readonly int SortKernelItemsPerGroupID = Shader.PropertyToID("sort_kernel_items_per_group");
        private static readonly int MaxSortCountID = Shader.PropertyToID("max_sort_count");
        private static readonly int SortCountBufferID = Shader.PropertyToID("sort_count_buffer");
        private static readonly int SortCountBufferOffsetID = Shader.PropertyToID("sort_count_buffer_offset");
        private static readonly int CountKernelDispatchArgsBufferID = Shader.PropertyToID("count_kernel_dispatch_args_buffer");
        private static readonly int SortKernelDispatchArgsBufferID = Shader.PropertyToID("sort_kernel_dispatch_args_buffer");
        private static readonly int SortCountGroupCountBufferID = Shader.PropertyToID("sort_count_group_count_buffer");
        #endregion

        #region Private Fields
        private ComputeShader _precomputeCs;
        private ComputeShader _initCs;
        private ComputeShader _countCs;
        private ComputeShader _scanCs;
        private ComputeShader _sortCs;
        private ComputeShader[] _computeShaders;
        private int _precomputeKernel;
        private int _initKernel;
        private int _countKernel;
        private int _scanKernel;
        private int _sortKernel;
        private int[] _kernels = new int[4];

        // Stores the temporary key data
        // size: input buffer size
        private GraphicsBuffer _tempKeyBuffer;
        // Stores the temporary payload data
        // size: input buffer size
        private GraphicsBuffer _tempPayloadBuffer;

        // Contains the global histogram data (counts for all buckets, across all radix passes).
        // size: RadixBase * RadixStepCount
        private GraphicsBuffer _bucketCountBuffer;

        // Provides atomic counters for assigning unique partition indices to processing groups per radix pass.
        // size: RadixStepCount
        private GraphicsBuffer _partitionIndexBuffer;

        // Stores PARTITION_DESCRIPTORs (bucket aggregates/prefixes and flags) used for the inter-group lookback scan mechanism.
        // size: RadixBase * groupCount * RadixStepCount
        private GraphicsBuffer _partitionDescriptorBuffer;

        // Stores the sort count and group count
        // size: 2
        private GraphicsBuffer _sortCountGroupCountBuffer;
        // StoreS the dispatch args for the count kernel
        // size: 3
        private GraphicsBuffer _countKernelDispatchArgsBuffer;
        // Stores the dispatch args for the sort kernel
        // size: 3
        private GraphicsBuffer _sortKernelDispatchArgsBuffer;
        #endregion

        #region Public Properties
        public bool Inited { get; private set; } = false;

        public SortingAlgorithm SortingAlgorithm => SortingAlgorithm.Onesweep;
        public SortMode SortMode { get; private set; }
        public KeyType KeyType { get; private set; }
        public SortingOrder SortingOrder { get; private set; }
        public DispatchMode DispatchMode { get; private set; }
        public WaveSize WaveSize { get; private set; }
        public int MaxSortCount { get; private set; }

        private static bool _hasDisplayedOnesweepWarning = false;

        /// <summary>
        /// Initializes the sorter with specified configurations.
        /// </summary>
        /// <param name="onesweepComputeConfig">Compute shader configuration asset.</param>
        /// <param name="maxSortCount">Maximum number of elements this sorter instance can handle.</param>
        /// <param name="sortMode">Specifies whether to sort keys only, or keys with an accompanying payload.</param>
        /// <param name="keyType">Data type of the keys to sort (UInt, Int, Float).</param>
        /// <param name="sortingOrder">Order of sorting (ascending/descending).</param>
        /// <param name="dispatchMode">Dispatch mode (Direct, Indirect) for compute shaders. If you pass the sort count using GraphicsBuffer, you should use Indirect.</param>
        /// <param name="waveSize">GPU wave size for shader execution.</param>
        /// <param name="forceClearBuffers">Whether to force clear existing internal buffers upon initialization.</param>
        /// <returns>The sorter instance for chaining or IDisposable usage.</returns>
        public IDisposable Init(OnesweepComputeConfig onesweepComputeConfig, int maxSortCount, SortMode sortMode, KeyType keyType, SortingOrder sortingOrder, DispatchMode dispatchMode, WaveSize waveSize, bool forceClearBuffers = false)
        {
            Inited = false;

            if (!_hasDisplayedOnesweepWarning)
            {
                Debug.LogWarning(
                    "Onesweep Warning:\n" +
                    "Potential for deadlocks cannot be definitively ruled out, and execution speed may be unstable at times. Please use with caution.\n" +
                    "A stable alternative, TraditionalSorter, is also available."
                    );
                _hasDisplayedOnesweepWarning = true;
            }

            if (!SorterCommon.GraphicsDeviceTypeIsDirect3D12())
                throw new InvalidOperationException(
                    $"DirectX 12 is required, but current Graphics API is: {SorterCommon.GetGraphicsDeviceType()}"
                );

            SortMode = sortMode;
            KeyType = keyType;
            SortingOrder = sortingOrder;
            DispatchMode = dispatchMode;

            if (SorterCommon.StoredWaveSize != WaveSize.Unknown && waveSize != WaveSize.Unknown && SorterCommon.StoredWaveSize != waveSize)
                Debug.LogWarning($"This device wave size is {SorterCommon.StoredWaveSize}. Requested {waveSize} is different.");
            if (waveSize != WaveSize.Unknown)
            {
                WaveSize = waveSize;
            }
            else if (SorterCommon.StoredWaveSize != WaveSize.Unknown)
            {
                WaveSize = SorterCommon.StoredWaveSize;
            }
            else
            {
                SorterCommon.GetStoreWaveSize(onesweepComputeConfig, out var waveSizeUInt);
                if (SorterCommon.StoredWaveSize == WaveSize.Unknown)
                    throw new NotSupportedException($"Could not determine a supported wave size (32 or 64) for this device. Detected: {waveSizeUInt}.");
                WaveSize = SorterCommon.StoredWaveSize;
            }

            MaxSortCount = Mathf.Max(maxSortCount, 1);
            if (MaxSortCount > SortKernelItemsPerGroup * MaxDispatchSize)
                throw new ArgumentException($"MaxSortCount ({MaxSortCount}) exceeds the sorter's limit ({SortKernelItemsPerGroup * MaxDispatchSize}).");

            _precomputeCs ??= Object.Instantiate(onesweepComputeConfig.OnesweepPrecomputeCs);
            _initCs ??= Object.Instantiate(onesweepComputeConfig.OnesweepInitCs);
            _countCs ??= Object.Instantiate(onesweepComputeConfig.OnesweepCountCs);
            _scanCs ??= Object.Instantiate(onesweepComputeConfig.OnesweepScanCs);
            _sortCs ??= Object.Instantiate(onesweepComputeConfig.OnesweepSortCs);
            _computeShaders ??= new[] { _initCs, _countCs, _scanCs, _sortCs };

            _precomputeKernel = _precomputeCs.FindKernel("PrecomputeForIndirectDispatch");
            _kernels[0] = _initKernel = _initCs.FindKernel("InitBuffers");
            _kernels[1] = _countKernel = _countCs.FindKernel("CountRadixBuckets");
            _kernels[2] = _scanKernel = _scanCs.FindKernel("ScanRadixBuckets");
            _kernels[3] = _sortKernel = _sortCs.FindKernel("Sort");

            if (forceClearBuffers) ReleaseBuffers();

            if (_tempKeyBuffer is not null && _tempKeyBuffer.count < MaxSortCount)
            {
                _tempKeyBuffer.Release();
                _tempKeyBuffer = null;
            }
            _tempKeyBuffer ??= new GraphicsBuffer(GraphicsBuffer.Target.Structured, MaxSortCount, sizeof(uint));

            if (SortMode == SortMode.KeyPayload)
            {
                if (_tempPayloadBuffer is not null && _tempPayloadBuffer.count < MaxSortCount)
                {
                    _tempPayloadBuffer.Release();
                    _tempPayloadBuffer = null;
                }
                _tempPayloadBuffer ??= new GraphicsBuffer(GraphicsBuffer.Target.Structured, MaxSortCount, sizeof(uint));
            }

            _bucketCountBuffer ??= new GraphicsBuffer(GraphicsBuffer.Target.Structured, RadixBase * RadixStepCount, sizeof(uint));
            _partitionIndexBuffer ??= new GraphicsBuffer(GraphicsBuffer.Target.Structured, RadixStepCount, sizeof(uint));
            int sortKernelMaxDispatchGroupCount = SortKernelDispatchGroupSize(MaxSortCount);
            if (_partitionDescriptorBuffer is not null && _partitionDescriptorBuffer.count < RadixBase * sortKernelMaxDispatchGroupCount * RadixStepCount)
            {
                _partitionDescriptorBuffer.Release();
                _partitionDescriptorBuffer = null;
            }
            _partitionDescriptorBuffer ??= new GraphicsBuffer(GraphicsBuffer.Target.Structured, RadixBase * sortKernelMaxDispatchGroupCount * RadixStepCount, sizeof(uint));

            if (DispatchMode is DispatchMode.Indirect)
            {
                _sortCountGroupCountBuffer ??= new GraphicsBuffer(GraphicsBuffer.Target.Raw, 2, sizeof(uint));
                _countKernelDispatchArgsBuffer ??= new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, 3, sizeof(uint));
                _sortKernelDispatchArgsBuffer ??= new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, 3, sizeof(uint));
            }

            foreach (var cs in _computeShaders)
            {
                SorterCommon.SetShaderKeywords(cs, SortMode, KeyType, SortingOrder, DispatchMode, WaveSize);
            }

            Inited = true;
            return this;
        }

        /// <summary>
        /// Sorts the key buffer. If SortMode is SortMode.KeyPayload, the payloadBuffer is also sorted.
        /// </summary>
        /// <param name="keyBuffer">Buffer containing the keys to sort. Must not be null. Stride must be 4 bytes.</param>
        /// <param name="payloadBuffer">
        /// Buffer containing payloads. Required if SortMode is KeyPayload (stride must be 4 bytes).
        /// Should be null if SortMode is KeyOnly.
        /// </param>
        /// <param name="sortCount">
        /// Number of elements to sort. If -1 (default), sorts all elements in keyBuffer
        /// (respecting MaxSortCount and buffer capacity).
        /// </param>
        public void Sort(GraphicsBuffer keyBuffer, GraphicsBuffer payloadBuffer, int sortCount = -1)
        {
            if (!Inited) throw new InvalidOperationException("The sorter is not initialized.");
            if (DispatchMode == DispatchMode.Indirect) throw new ArgumentException("For Indirect dispatch, use the Sort overload that accepts sortCountBuffer.");
            if (keyBuffer == null) throw new ArgumentNullException(nameof(keyBuffer));

            if (keyBuffer.stride != sizeof(uint)) throw new ArgumentException("The stride of keyBuffer must be 4 bytes.", nameof(keyBuffer));

            if (sortCount < 0) sortCount = keyBuffer.count;
            if (sortCount == 0) return;
            if (sortCount > MaxSortCount) throw new ArgumentOutOfRangeException(nameof(sortCount), $"sortCount ({sortCount}) exceeds MaxSortCount ({MaxSortCount}).");
            if (sortCount > keyBuffer.count) throw new ArgumentOutOfRangeException(nameof(sortCount), "sortCount exceeds keyBuffer's capacity.");

            if (SortMode == SortMode.KeyPayload)
            {
                if (payloadBuffer == null) throw new ArgumentNullException(nameof(payloadBuffer), "PayloadBuffer cannot be null when SortMode is KeyPayload.");
                if (payloadBuffer.stride != sizeof(uint)) throw new ArgumentException("The stride of payloadBuffer must be 4 bytes for KeyPayload sort.", nameof(payloadBuffer));
                if (sortCount > 0 && sortCount > payloadBuffer.count) throw new ArgumentOutOfRangeException(nameof(sortCount), "sortCount exceeds payloadBuffer's capacity.");
            }

            foreach (var cs in _computeShaders)
            {
                cs.SetInt(SortCountID, sortCount);
                int groupCount = SortKernelDispatchGroupSize(sortCount);
                cs.SetInt(GroupCountID, groupCount);
            }

            _initCs.SetBuffer(_initKernel, BucketCountBufferID, _bucketCountBuffer);
            _initCs.SetBuffer(_initKernel, PartitionIndexBufferID, _partitionIndexBuffer);
            _initCs.SetBuffer(_initKernel, PartitionDescriptorBufferID, _partitionDescriptorBuffer);
            _initCs.Dispatch(_initKernel, InitKernelDispatchGroupSize, 1, 1);

            _countCs.SetBuffer(_countKernel, BucketCountBufferID, _bucketCountBuffer);
            _countCs.SetBuffer(_countKernel, KeyInBufferID, keyBuffer);
            _countCs.Dispatch(_countKernel, CountKernelDispatchGroupSize(sortCount), 1, 1);

            _scanCs.SetBuffer(_scanKernel, BucketCountBufferID, _bucketCountBuffer);
            _scanCs.SetBuffer(_scanKernel, PartitionDescriptorBufferID, _partitionDescriptorBuffer);
            _scanCs.Dispatch(_scanKernel, ScanKernelDispatchGroupSize, 1, 1);

            _sortCs.SetBuffer(_sortKernel, PartitionIndexBufferID, _partitionIndexBuffer);
            _sortCs.SetBuffer(_sortKernel, PartitionDescriptorBufferID, _partitionDescriptorBuffer);

            bool usePayload = SortMode == SortMode.KeyPayload;

            for (int i = 0; i < RadixStepCount; i++)
            {
                var currentKeyIn = i % 2 == 0 ? keyBuffer : _tempKeyBuffer;
                var currentKeyOut = i % 2 == 0 ? _tempKeyBuffer : keyBuffer;

                _sortCs.SetInt(CurrentPassRadixShiftID, i << 3);
                _sortCs.SetBuffer(_sortKernel, KeyInBufferID, currentKeyIn);
                _sortCs.SetBuffer(_sortKernel, KeyOutBufferID, currentKeyOut);
                if (usePayload)
                {
                    var currentPayloadIn = i % 2 == 0 ? payloadBuffer : _tempPayloadBuffer;
                    var currentPayloadOut = i % 2 == 0 ? _tempPayloadBuffer : payloadBuffer;
                    _sortCs.SetBuffer(_sortKernel, PayloadInBufferID, currentPayloadIn);
                    _sortCs.SetBuffer(_sortKernel, PayloadOutBufferID, currentPayloadOut);
                }
                _sortCs.Dispatch(_sortKernel, SortKernelDispatchGroupSize(sortCount), 1, 1);
            }
        }

        /// <summary>
        /// Sorts the key buffer. If SortMode is SortMode.KeyPayload, the payloadBuffer is also sorted.
        /// </summary>
        /// <param name="cmd">Command buffer to record dispatches. Must not be null.</param>
        /// <param name="keyBuffer">Buffer containing the keys to sort. Must not be null. Stride must be 4 bytes.</param>
        /// <param name="payloadBuffer">
        /// Buffer containing payloads. Required if SortMode is KeyPayload (stride must be 4 bytes).
        /// Should be null if SortMode is KeyOnly.
        /// </param>
        /// <param name="sortCount">
        /// Number of elements to sort. If -1 (default), sorts all elements in keyBuffer
        /// (respecting MaxSortCount and buffer capacity).
        /// </param>
        public void Sort(CommandBuffer cmd, GraphicsBuffer keyBuffer, GraphicsBuffer payloadBuffer, int sortCount = -1)
        {
            if (!Inited) throw new InvalidOperationException("The sorter is not initialized.");
            if (DispatchMode == DispatchMode.Indirect) throw new ArgumentException("For Indirect dispatch, use the Sort overload that accepts sortCountBuffer.");
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (keyBuffer == null) throw new ArgumentNullException(nameof(keyBuffer));

            if (keyBuffer.stride != sizeof(uint)) throw new ArgumentException("The stride of keyBuffer must be 4 bytes.", nameof(keyBuffer));

            if (sortCount < 0) sortCount = keyBuffer.count;
            if (sortCount == 0) return;
            if (sortCount > MaxSortCount) throw new ArgumentOutOfRangeException(nameof(sortCount), $"sortCount ({sortCount}) exceeds MaxSortCount ({MaxSortCount}).");
            if (sortCount > keyBuffer.count) throw new ArgumentOutOfRangeException(nameof(sortCount), "sortCount exceeds keyBuffer's capacity.");

            if (SortMode == SortMode.KeyPayload)
            {
                if (payloadBuffer == null) throw new ArgumentNullException(nameof(payloadBuffer), "PayloadBuffer cannot be null when SortMode is KeyPayload.");
                if (payloadBuffer.stride != sizeof(uint)) throw new ArgumentException("The stride of payloadBuffer must be 4 bytes for KeyPayload sort.", nameof(payloadBuffer));
                if (sortCount > 0 && sortCount > payloadBuffer.count) throw new ArgumentOutOfRangeException(nameof(sortCount), "sortCount exceeds payloadBuffer's capacity.");
            }

            foreach (var cs in _computeShaders)
            {
                cmd.SetComputeIntParam(cs, SortCountID, sortCount);
                int groupCount = SortKernelDispatchGroupSize(sortCount);
                cmd.SetComputeIntParam(cs, GroupCountID, groupCount);
            }

            cmd.SetComputeBufferParam(_initCs, _initKernel, BucketCountBufferID, _bucketCountBuffer);
            cmd.SetComputeBufferParam(_initCs, _initKernel, PartitionIndexBufferID, _partitionIndexBuffer);
            cmd.SetComputeBufferParam(_initCs, _initKernel, PartitionDescriptorBufferID, _partitionDescriptorBuffer);
            cmd.DispatchCompute(_initCs, _initKernel, InitKernelDispatchGroupSize, 1, 1);

            cmd.SetComputeBufferParam(_countCs, _countKernel, BucketCountBufferID, _bucketCountBuffer);
            cmd.SetComputeBufferParam(_countCs, _countKernel, KeyInBufferID, keyBuffer);
            cmd.DispatchCompute(_countCs, _countKernel, CountKernelDispatchGroupSize(sortCount), 1, 1);

            cmd.SetComputeBufferParam(_scanCs, _scanKernel, BucketCountBufferID, _bucketCountBuffer);
            cmd.SetComputeBufferParam(_scanCs, _scanKernel, PartitionDescriptorBufferID, _partitionDescriptorBuffer);
            cmd.DispatchCompute(_scanCs, _scanKernel, ScanKernelDispatchGroupSize, 1, 1);

            cmd.SetComputeBufferParam(_sortCs, _sortKernel, PartitionIndexBufferID, _partitionIndexBuffer);
            cmd.SetComputeBufferParam(_sortCs, _sortKernel, PartitionDescriptorBufferID, _partitionDescriptorBuffer);

            bool usePayload = SortMode == SortMode.KeyPayload;

            for (int i = 0; i < RadixStepCount; i++)
            {
                var currentKeyIn = i % 2 == 0 ? keyBuffer : _tempKeyBuffer;
                var currentKeyOut = i % 2 == 0 ? _tempKeyBuffer : keyBuffer;

                cmd.SetComputeIntParam(_sortCs, CurrentPassRadixShiftID, i << 3);
                cmd.SetComputeBufferParam(_sortCs, _sortKernel, KeyInBufferID, currentKeyIn);
                cmd.SetComputeBufferParam(_sortCs, _sortKernel, KeyOutBufferID, currentKeyOut);
                if (usePayload)
                {
                    var currentPayloadIn = i % 2 == 0 ? payloadBuffer : _tempPayloadBuffer;
                    var currentPayloadOut = i % 2 == 0 ? _tempPayloadBuffer : payloadBuffer;
                    cmd.SetComputeBufferParam(_sortCs, _sortKernel, PayloadInBufferID, currentPayloadIn);
                    cmd.SetComputeBufferParam(_sortCs, _sortKernel, PayloadOutBufferID, currentPayloadOut);
                }
                cmd.DispatchCompute(_sortCs, _sortKernel, SortKernelDispatchGroupSize(sortCount), 1, 1);
            }
        }

        /// <summary>
        /// Sorts the key buffer using a sort count from a GraphicsBuffer (for indirect dispatch).
        /// If SortMode is SortMode.KeyPayload, the payloadBuffer is also sorted.
        /// </summary>
        /// <param name="keyBuffer">Buffer containing keys. Must not be null. Stride must be 4 bytes.</param>
        /// <param name="payloadBuffer">
        /// Buffer containing payloads. Required if SortMode is KeyPayload (stride must be 4 bytes).
        /// Should be null if SortMode is KeyOnly.
        /// </param>
        /// <param name="sortCountBuffer">Buffer containing the sort count (uint). Must not be null. Stride must be 4 bytes.</param>
        /// <param name="sortCountBufferOffset">Element offset in sortCountBuffer for the count value.</param>
        public void Sort(GraphicsBuffer keyBuffer, GraphicsBuffer payloadBuffer, GraphicsBuffer sortCountBuffer, uint sortCountBufferOffset)
        {
            if (!Inited) throw new InvalidOperationException("The sorter is not initialized.");
            if (DispatchMode == DispatchMode.Direct) throw new ArgumentException("For Direct dispatch, use the Sort overload that accepts sortCount directly.");
            if (keyBuffer == null) throw new ArgumentNullException(nameof(keyBuffer));
            if (sortCountBuffer == null) throw new ArgumentNullException(nameof(sortCountBuffer));

            if (SortMode == SortMode.KeyPayload)
            {
                if (payloadBuffer == null) throw new ArgumentNullException(nameof(payloadBuffer), "PayloadBuffer cannot be null when SortMode is KeyPayload.");
                if (payloadBuffer.stride != sizeof(uint)) throw new ArgumentException("The stride of payloadBuffer must be 4 bytes for KeyPayload sort.", nameof(payloadBuffer));
            }

            if (keyBuffer.stride != sizeof(uint)) throw new ArgumentException("The stride of keyBuffer must be 4 bytes.", nameof(keyBuffer));
            if (sortCountBuffer.stride != sizeof(uint)) throw new ArgumentException("The stride of sortCountBuffer must be 4 bytes (for uint count).", nameof(sortCountBuffer));

            int effectiveMaxSortCount = SortMode == SortMode.KeyPayload
                ? Mathf.Min(MaxSortCount, Mathf.Min(keyBuffer.count, payloadBuffer.count))
                : Mathf.Min(MaxSortCount, keyBuffer.count);

            _precomputeCs.SetInt(CountKernelItemsPerGroupID, CountKernelItemsPerGroup);
            _precomputeCs.SetInt(SortKernelItemsPerGroupID, SortKernelItemsPerGroup);
            _precomputeCs.SetInt(MaxSortCountID, effectiveMaxSortCount);
            _precomputeCs.SetBuffer(_precomputeKernel, SortCountBufferID, sortCountBuffer);
            _precomputeCs.SetInt(SortCountBufferOffsetID, (int)sortCountBufferOffset);
            _precomputeCs.SetBuffer(_precomputeKernel, CountKernelDispatchArgsBufferID, _countKernelDispatchArgsBuffer);
            _precomputeCs.SetBuffer(_precomputeKernel, SortKernelDispatchArgsBufferID, _sortKernelDispatchArgsBuffer);
            _precomputeCs.SetBuffer(_precomputeKernel, SortCountGroupCountBufferID, _sortCountGroupCountBuffer);
            _precomputeCs.Dispatch(_precomputeKernel, PrecomputeKernelDispatchGroupSize, 1, 1);

            for (int i = 0; i < _computeShaders.Length; i++)
            {
                var cs = _computeShaders[i];
                var kernel = _kernels[i];
                cs.SetBuffer(kernel, SortCountGroupCountBufferID, _sortCountGroupCountBuffer);
            }

            _initCs.SetBuffer(_initKernel, BucketCountBufferID, _bucketCountBuffer);
            _initCs.SetBuffer(_initKernel, PartitionIndexBufferID, _partitionIndexBuffer);
            _initCs.SetBuffer(_initKernel, PartitionDescriptorBufferID, _partitionDescriptorBuffer);
            _initCs.Dispatch(_initKernel, InitKernelDispatchGroupSize, 1, 1);

            _countCs.SetBuffer(_countKernel, BucketCountBufferID, _bucketCountBuffer);
            _countCs.SetBuffer(_countKernel, KeyInBufferID, keyBuffer);
            _countCs.DispatchIndirect(_countKernel, _countKernelDispatchArgsBuffer);

            _scanCs.SetBuffer(_scanKernel, BucketCountBufferID, _bucketCountBuffer);
            _scanCs.SetBuffer(_scanKernel, PartitionDescriptorBufferID, _partitionDescriptorBuffer);
            _scanCs.Dispatch(_scanKernel, ScanKernelDispatchGroupSize, 1, 1);

            _sortCs.SetBuffer(_sortKernel, PartitionIndexBufferID, _partitionIndexBuffer);
            _sortCs.SetBuffer(_sortKernel, PartitionDescriptorBufferID, _partitionDescriptorBuffer);

            bool usePayload = SortMode == SortMode.KeyPayload;

            for (int i = 0; i < RadixStepCount; i++)
            {
                var currentKeyIn = i % 2 == 0 ? keyBuffer : _tempKeyBuffer;
                var currentKeyOut = i % 2 == 0 ? _tempKeyBuffer : keyBuffer;

                _sortCs.SetInt(CurrentPassRadixShiftID, i << 3);
                _sortCs.SetBuffer(_sortKernel, KeyInBufferID, currentKeyIn);
                _sortCs.SetBuffer(_sortKernel, KeyOutBufferID, currentKeyOut);
                if (usePayload)
                {
                    var currentPayloadIn = i % 2 == 0 ? payloadBuffer : _tempPayloadBuffer;
                    var currentPayloadOut = i % 2 == 0 ? _tempPayloadBuffer : payloadBuffer;
                    _sortCs.SetBuffer(_sortKernel, PayloadInBufferID, currentPayloadIn);
                    _sortCs.SetBuffer(_sortKernel, PayloadOutBufferID, currentPayloadOut);
                }
                _sortCs.DispatchIndirect(_sortKernel, _sortKernelDispatchArgsBuffer);
            }
        }

        /// <summary>
        /// Sorts the key buffer using a CommandBuffer and a sort count from a GraphicsBuffer (for indirect dispatch).
        /// If SortMode is SortMode.KeyPayload, the payloadBuffer is also sorted.
        /// </summary>
        /// <param name="cmd">Command buffer to record dispatches. Must not be null.</param>
        /// <param name="keyBuffer">Buffer containing keys. Must not be null. Stride must be 4 bytes.</param>
        /// <param name="payloadBuffer">
        /// Buffer containing payloads. Required if SortMode is KeyPayload (stride must be 4 bytes).
        /// Should be null if SortMode is KeyOnly.
        /// </param>
        /// <param name="sortCountBuffer">Buffer containing the sort count (uint). Must not be null. Stride must be 4 bytes.</param>
        /// <param name="sortCountBufferOffset">Element offset in sortCountBuffer for the count value.</param>
        public void Sort(CommandBuffer cmd, GraphicsBuffer keyBuffer, GraphicsBuffer payloadBuffer, GraphicsBuffer sortCountBuffer, uint sortCountBufferOffset)
        {
            if (!Inited) throw new InvalidOperationException("The sorter is not initialized.");
            if (DispatchMode == DispatchMode.Direct) throw new ArgumentException("For Direct dispatch, use the Sort overload that accepts sortCount directly.");
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (keyBuffer == null) throw new ArgumentNullException(nameof(keyBuffer));
            if (sortCountBuffer == null) throw new ArgumentNullException(nameof(sortCountBuffer));

            if (SortMode == SortMode.KeyPayload)
            {
                if (payloadBuffer == null) throw new ArgumentNullException(nameof(payloadBuffer), "PayloadBuffer cannot be null when SortMode is KeyPayload.");
                if (payloadBuffer.stride != sizeof(uint)) throw new ArgumentException("The stride of payloadBuffer must be 4 bytes for KeyPayload sort.", nameof(payloadBuffer));
            }

            if (keyBuffer.stride != sizeof(uint)) throw new ArgumentException("The stride of keyBuffer must be 4 bytes.", nameof(keyBuffer));
            if (sortCountBuffer.stride != sizeof(uint)) throw new ArgumentException("The stride of sortCountBuffer must be 4 bytes (for uint count).", nameof(sortCountBuffer));

            int effectiveMaxSortCount = SortMode == SortMode.KeyPayload
                ? Mathf.Min(MaxSortCount, Mathf.Min(keyBuffer.count, payloadBuffer.count))
                : Mathf.Min(MaxSortCount, keyBuffer.count);

            cmd.SetComputeIntParam(_precomputeCs, CountKernelItemsPerGroupID, CountKernelItemsPerGroup);
            cmd.SetComputeIntParam(_precomputeCs, SortKernelItemsPerGroupID, SortKernelItemsPerGroup);
            cmd.SetComputeIntParam(_precomputeCs, MaxSortCountID, effectiveMaxSortCount);
            cmd.SetComputeBufferParam(_precomputeCs, _precomputeKernel, SortCountBufferID, sortCountBuffer);
            cmd.SetComputeIntParam(_precomputeCs, SortCountBufferOffsetID, (int)sortCountBufferOffset);
            cmd.SetComputeBufferParam(_precomputeCs, _precomputeKernel, CountKernelDispatchArgsBufferID, _countKernelDispatchArgsBuffer);
            cmd.SetComputeBufferParam(_precomputeCs, _precomputeKernel, SortKernelDispatchArgsBufferID, _sortKernelDispatchArgsBuffer);
            cmd.SetComputeBufferParam(_precomputeCs, _precomputeKernel, SortCountGroupCountBufferID, _sortCountGroupCountBuffer);
            cmd.DispatchCompute(_precomputeCs, _precomputeKernel, PrecomputeKernelDispatchGroupSize, 1, 1);

            for (int i = 0; i < _computeShaders.Length; i++)
            {
                var cs = _computeShaders[i];
                var kernel = _kernels[i];
                cmd.SetComputeBufferParam(cs, kernel, SortCountGroupCountBufferID, _sortCountGroupCountBuffer);
            }

            cmd.SetComputeBufferParam(_initCs, _initKernel, BucketCountBufferID, _bucketCountBuffer);
            cmd.SetComputeBufferParam(_initCs, _initKernel, PartitionIndexBufferID, _partitionIndexBuffer);
            cmd.SetComputeBufferParam(_initCs, _initKernel, PartitionDescriptorBufferID, _partitionDescriptorBuffer);
            cmd.DispatchCompute(_initCs, _initKernel, InitKernelDispatchGroupSize, 1, 1);

            cmd.SetComputeBufferParam(_countCs, _countKernel, BucketCountBufferID, _bucketCountBuffer);
            cmd.SetComputeBufferParam(_countCs, _countKernel, KeyInBufferID, keyBuffer);
            cmd.DispatchCompute(_countCs, _countKernel, _countKernelDispatchArgsBuffer, 0);

            cmd.SetComputeBufferParam(_scanCs, _scanKernel, BucketCountBufferID, _bucketCountBuffer);
            cmd.SetComputeBufferParam(_scanCs, _scanKernel, PartitionDescriptorBufferID, _partitionDescriptorBuffer);
            cmd.DispatchCompute(_scanCs, _scanKernel, ScanKernelDispatchGroupSize, 1, 1);

            cmd.SetComputeBufferParam(_sortCs, _sortKernel, PartitionIndexBufferID, _partitionIndexBuffer);
            cmd.SetComputeBufferParam(_sortCs, _sortKernel, PartitionDescriptorBufferID, _partitionDescriptorBuffer);

            bool usePayload = SortMode == SortMode.KeyPayload;

            for (int i = 0; i < RadixStepCount; i++)
            {
                var currentKeyIn = i % 2 == 0 ? keyBuffer : _tempKeyBuffer;
                var currentKeyOut = i % 2 == 0 ? _tempKeyBuffer : keyBuffer;

                cmd.SetComputeIntParam(_sortCs, CurrentPassRadixShiftID, i << 3);
                cmd.SetComputeBufferParam(_sortCs, _sortKernel, KeyInBufferID, currentKeyIn);
                cmd.SetComputeBufferParam(_sortCs, _sortKernel, KeyOutBufferID, currentKeyOut);
                if (usePayload)
                {
                    var currentPayloadIn = i % 2 == 0 ? payloadBuffer : _tempPayloadBuffer;
                    var currentPayloadOut = i % 2 == 0 ? _tempPayloadBuffer : payloadBuffer;
                    cmd.SetComputeBufferParam(_sortCs, _sortKernel, PayloadInBufferID, currentPayloadIn);
                    cmd.SetComputeBufferParam(_sortCs, _sortKernel, PayloadOutBufferID, currentPayloadOut);
                }
                cmd.DispatchCompute(_sortCs, _sortKernel, _sortKernelDispatchArgsBuffer, 0);
            }
        }

        /// <summary>
        /// Releases all resources used by the sorter instance.
        /// </summary>
        public void Dispose()
        {
            ReleaseBuffers();
            Inited = false;
        }
        #endregion

        #region Private Methods
        /// <summary>
        /// Releases all allocated GPU buffers.
        /// </summary>
        private void ReleaseBuffers()
        {
            _tempKeyBuffer?.Release(); _tempKeyBuffer = null;
            _tempPayloadBuffer?.Release(); _tempPayloadBuffer = null;
            _bucketCountBuffer?.Release(); _bucketCountBuffer = null;
            _partitionIndexBuffer?.Release(); _partitionIndexBuffer = null;
            _partitionDescriptorBuffer?.Release(); _partitionDescriptorBuffer = null;
            _sortCountGroupCountBuffer?.Release(); _sortCountGroupCountBuffer = null;
            _countKernelDispatchArgsBuffer?.Release(); _countKernelDispatchArgsBuffer = null;
            _sortKernelDispatchArgsBuffer?.Release(); _sortKernelDispatchArgsBuffer = null;
        }
        #endregion
    }
}