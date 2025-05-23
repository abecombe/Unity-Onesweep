using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace Onesweep
{
    /// /// <summary>
    /// GPU One Sweep Radix Sort
    /// Implementation of Paper "Onesweep: A Faster Least Significant Digit Radix Sort for GPUs"
    /// https://arxiv.org/abs/2206.01784
    ///
    /// This implementation includes code adapted from the GPUSorting project by Thomas Smith
    /// https://github.com/b0nes164/GPUSorting
    /// Licensed under the MIT License
    /// </summary>
    public class OnesweepSorter : ISorter
    {
        #region Constants
        private const int MaxDispatchSize = 65535;

        private const int RadixBase = 256;
        private const int RadixStepCount = 4;

        private const int BuildKernelItemsPerThread = 64;
        private const int BuildKernelThreadsPerGroup = 128;
        private const int SortKernelItemsPerThread = 15;
        private const int SortKernelThreadsPerGroup = RadixBase;

        private const int PrecomputeKernelDispatchGroupSize = 1;
        private const int InitKernelDispatchGroupSize = 128;
        private static int BuildKernelDispatchGroupSize(int sortCount) =>
            (sortCount + BuildKernelItemsPerThread * BuildKernelThreadsPerGroup - 1) / (BuildKernelItemsPerThread * BuildKernelThreadsPerGroup);
        private static int ScanKernelDispatchGroupSize => RadixStepCount;
        private static int SortKernelDispatchGroupSize(int sortCount) =>
            (sortCount + SortKernelItemsPerThread * SortKernelThreadsPerGroup - 1) / (SortKernelItemsPerThread * SortKernelThreadsPerGroup);
        #endregion

        #region Shader Property IDs
        private static readonly int SortCountID = Shader.PropertyToID("sort_count");
        private static readonly int GroupID = Shader.PropertyToID("group_count");
        private static readonly int BucketCountBufferID = Shader.PropertyToID("bucket_count_buffer");
        private static readonly int PartitionIndexBufferID = Shader.PropertyToID("partition_index_buffer");
        private static readonly int PartitionDescriptorBufferID = Shader.PropertyToID("partition_descriptor_buffer");
        private static readonly int KeyInBufferID = Shader.PropertyToID("key_in_buffer");
        private static readonly int CurrentPassRadixShiftID = Shader.PropertyToID("current_pass_radix_shift");
        private static readonly int KeyOutBufferID = Shader.PropertyToID("key_out_buffer");
        private static readonly int IndexInBufferID = Shader.PropertyToID("index_in_buffer");
        private static readonly int IndexOutBufferID = Shader.PropertyToID("index_out_buffer");
        private static readonly int BuildKernelItemsPerGroupID = Shader.PropertyToID("build_kernel_items_per_group");
        private static readonly int SortKernelItemsPerGroupID = Shader.PropertyToID("sort_kernel_items_per_group");
        private static readonly int MaxSortCountID = Shader.PropertyToID("max_sort_count");
        private static readonly int SortCountBufferID = Shader.PropertyToID("sort_count_buffer");
        private static readonly int SortCountBufferOffsetID = Shader.PropertyToID("sort_count_buffer_offset");
        private static readonly int BuildKernelDispatchArgsBufferID = Shader.PropertyToID("build_kernel_dispatch_args_buffer");
        private static readonly int SortKernelDispatchArgsBufferID = Shader.PropertyToID("sort_kernel_dispatch_args_buffer");
        private static readonly int SortCountGroupCountBufferID = Shader.PropertyToID("sort_count_group_count_buffer");
        #endregion

        #region Private Fields
        private ComputeShader _precomputeCs;
        private ComputeShader _initCs;
        private ComputeShader _buildCs;
        private ComputeShader _scanCs;
        private ComputeShader _sortCs;
        private ComputeShader[] _computeShaders;
        private int _precomputeKernel;
        private int _initKernel;
        private int _buildKernel;
        private int _scanKernel;
        private int _sortKernel;
        private int[] _kernels;

        // buffer to store the temporary key data
        // size: input buffer size
        private GraphicsBuffer _tempKeyBuffer;
        // buffer to store the temporary index data
        // size: input buffer size
        private GraphicsBuffer _tempIndexBuffer;
        // buffer to store the total count of each radix bucket
        // size: RadixBase * RadixStepCount
        private GraphicsBuffer _bucketCountBuffer;
        // buffer to use for retrieving the partition index of each thread group
        // size: RadixStepCount
        private GraphicsBuffer _partitionIndexBuffer;
        // buffer to use for lookback scan
        // size: RadixBase * groupCount * RadixStepCount
        private GraphicsBuffer _partitionDescriptorBuffer;
        // buffer to store the sort count and group count
        // size: 2
        private GraphicsBuffer _sortCountGroupCountBuffer;
        // buffer to store the dispatch args for the build kernel
        // size: 3
        private GraphicsBuffer _buildKernelDispatchArgsBuffer;
        // buffer to store the dispatch args for the sort kernel
        // size: 3
        private GraphicsBuffer _sortKernelDispatchArgsBuffer;
        #endregion

        #region Public Properties
        public bool Inited { get; private set; } = false;

        public KeyType KeyType { get; private set; }
        public SortingOrder SortingOrder { get; private set; }
        public DispatchMode DispatchMode { get; private set; }
        public WaveSize WaveSize { get; private set; }
        public int MaxSortCount { get; private set; }

        private static bool _hasDisplayedOnesweepWarning = false;

        /// <summary>
        /// Initializes Sorter.
        /// </summary>
        /// <param name="onesweepComputeConfig">
        /// Compute shader configuration for Onesweep.
        /// </param>
        /// <param name="maxSortCount">
        /// The expected maximum number of elements to be sorted.
        /// This defines the upper limit of sortable elements for this instance.
        /// Attempting to sort more than this number will result in an error.
        /// </param>
        /// <param name="keyType">Sorting key type (UInt, Int, Float).</param>
        /// <param name="sortingOrder">Sorting order (Ascending, Descending).</param>
        /// <param name="dispatchMode">
        /// Dispatch type (Direct, Indirect). If you pass the sort count using GraphicsBuffer, you should use Indirect.
        /// </param>
        /// <param name="waveSize">
        /// Wave size (32, 64). If unknown, it will be determined by the compute shader.
        /// </param>
        /// <param name="forceClearBuffers">If true, forces buffer clearing.</param>
        /// <returns>RadixSort instance for IDisposable use.</returns>
        public IDisposable Init(OnesweepComputeConfig onesweepComputeConfig, int maxSortCount, KeyType keyType, SortingOrder sortingOrder, DispatchMode dispatchMode, WaveSize waveSize, bool forceClearBuffers = false)
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

            if (SorterCommon.GraphicsDeviceTypeIsDirect3D12())
                throw new InvalidOperationException(
                    $"DirectX 12 is required, but current Graphics API is: {SystemInfo.graphicsDeviceType}"
                );

            KeyType = keyType;
            SortingOrder = sortingOrder;
            DispatchMode = dispatchMode;

            if (SorterCommon.StoredWaveSize != WaveSize.Unknown && waveSize != WaveSize.Unknown && SorterCommon.StoredWaveSize != waveSize)
                throw new ArgumentException($"This device wave size is {SorterCommon.StoredWaveSize}.");
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
                    throw new ArgumentException($"This device wave size is {waveSizeUInt}. Wave size must be 32 or 64.");
                WaveSize = SorterCommon.StoredWaveSize;
            }

            MaxSortCount = Mathf.Max(maxSortCount, 1);
            if (MaxSortCount > SortKernelThreadsPerGroup * SortKernelItemsPerThread * MaxDispatchSize)
                throw new ArgumentException($"Buffer size must be less than or equal to {SortKernelThreadsPerGroup * SortKernelItemsPerThread * MaxDispatchSize}.");

            _precomputeCs = onesweepComputeConfig.PrecomputeCs;
            _initCs = onesweepComputeConfig.InitCs;
            _buildCs = onesweepComputeConfig.BuildCs;
            _scanCs = onesweepComputeConfig.ScanCs;
            _sortCs = onesweepComputeConfig.SortCs;
            _computeShaders = new[] { _initCs, _buildCs, _scanCs, _sortCs };

            _precomputeKernel = _precomputeCs.FindKernel("PrecomputeForIndirectDispatch");
            _initKernel = _initCs.FindKernel("InitBuffers");
            _buildKernel = _buildCs.FindKernel("BuildRadixBucketGlobalHistogram");
            _scanKernel = _scanCs.FindKernel("ScanRadixBucketGlobalHistogram");
            _sortKernel = _sortCs.FindKernel("SortOnesweep");
            _kernels = new[] { _initKernel, _buildKernel, _scanKernel, _sortKernel };

            if (forceClearBuffers) ReleaseBuffers();

            if (_tempKeyBuffer is not null && _tempKeyBuffer.count < MaxSortCount)
            {
                _tempKeyBuffer.Release();
                _tempKeyBuffer = null;
            }
            _tempKeyBuffer ??= new GraphicsBuffer(GraphicsBuffer.Target.Structured, MaxSortCount, sizeof(uint));
            if (_tempIndexBuffer is not null && _tempIndexBuffer.count < MaxSortCount)
            {
                _tempIndexBuffer.Release();
                _tempIndexBuffer = null;
            }
            _tempIndexBuffer ??= new GraphicsBuffer(GraphicsBuffer.Target.Structured, MaxSortCount, sizeof(uint));
            _bucketCountBuffer ??= new GraphicsBuffer(GraphicsBuffer.Target.Structured, RadixBase * RadixStepCount, sizeof(uint));
            _partitionIndexBuffer ??= new GraphicsBuffer(GraphicsBuffer.Target.Structured, RadixStepCount, sizeof(uint));
            int sortKernelMaxDispatchGroupCount = SortKernelDispatchGroupSize(MaxSortCount);
            if (_partitionDescriptorBuffer is not null && _partitionDescriptorBuffer.count < RadixBase * sortKernelMaxDispatchGroupCount * RadixStepCount)
            {
                _partitionDescriptorBuffer.Release();
                _partitionDescriptorBuffer = null;
            }
            _partitionDescriptorBuffer ??=
                new GraphicsBuffer(GraphicsBuffer.Target.Structured, RadixBase * sortKernelMaxDispatchGroupCount * RadixStepCount, sizeof(uint));
            if (DispatchMode is DispatchMode.Indirect)
            {
                _sortCountGroupCountBuffer ??= new GraphicsBuffer(GraphicsBuffer.Target.Raw, 2, sizeof(uint));
                _buildKernelDispatchArgsBuffer ??= new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, 3, sizeof(uint));
                _sortKernelDispatchArgsBuffer ??= new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, 3, sizeof(uint));
            }

            foreach (var cs in _computeShaders)
            {
                SorterCommon.SetShaderKeywords(cs, keyType, sortingOrder, dispatchMode, WaveSize);
            }

            Inited = true;

            return this;
        }

        /// <summary>
        /// Sorts the key and index buffers using the specified sort count.
        /// </summary>
        /// <param name="keyBuffer">
        /// The key buffer to be sorted.
        /// The buffer must have a stride of 4 bytes.
        /// Each element must match the KeyType specified during Init().
        /// </param>
        /// <param name="indexBuffer">
        /// The index buffer to be sorted alongside the key buffer.
        /// The buffer must have a stride of 4 bytes.
        /// </param>
        /// <param name="sortCount">
        /// Number of elements to sort (starting from the beginning of the buffer).
        /// If -1, the full buffer length is used.
        /// </param>
        public void Sort(GraphicsBuffer keyBuffer, GraphicsBuffer indexBuffer, int sortCount = -1)
        {
            if (!Inited)
                throw new ArgumentException("RadixSort is not initialized.");
            if (DispatchMode == DispatchMode.Indirect)
                throw new ArgumentException("The sort count must be passed via a GraphicsBuffer when using Indirect dispatch mode.");

            if (sortCount < 0) sortCount = keyBuffer.count;
            if (sortCount == 0) return;

            if (keyBuffer.stride != sizeof(uint) || indexBuffer.stride != sizeof(uint))
                throw new ArgumentException("The stride of keyBuffer and indexBuffer must be 4 bytes.");
            if (sortCount > MaxSortCount)
                throw new ArgumentException("The sort count must be less than or equal to the buffer size specified in Init.");
            if (sortCount > keyBuffer.count || sortCount > indexBuffer.count)
                throw new ArgumentException("The sort count must be less than or equal to the buffer size of keyBuffer and indexBuffer.");

            foreach (var cs in _computeShaders)
            {
                cs.SetInt(SortCountID, sortCount);
                int groupCount = SortKernelDispatchGroupSize(sortCount);
                cs.SetInt(GroupID, groupCount);
            }

            // init buffers
            _initCs.SetBuffer(_initKernel, BucketCountBufferID, _bucketCountBuffer);
            _initCs.SetBuffer(_initKernel, PartitionIndexBufferID, _partitionIndexBuffer);
            _initCs.SetBuffer(_initKernel, PartitionDescriptorBufferID, _partitionDescriptorBuffer);
            _initCs.Dispatch(_initKernel, InitKernelDispatchGroupSize, 1, 1);

            // build radix bucket global histogram
            _buildCs.SetBuffer(_buildKernel, BucketCountBufferID, _bucketCountBuffer);
            _buildCs.SetBuffer(_buildKernel, KeyInBufferID, keyBuffer);
            _buildCs.Dispatch(_buildKernel, BuildKernelDispatchGroupSize(sortCount), 1, 1);

            // scan radix bucket global histogram
            _scanCs.SetBuffer(_scanKernel, BucketCountBufferID, _bucketCountBuffer);
            _scanCs.SetBuffer(_scanKernel, PartitionDescriptorBufferID, _partitionDescriptorBuffer);
            _scanCs.Dispatch(_scanKernel, ScanKernelDispatchGroupSize, 1, 1);

            // sort onesweep
            _sortCs.SetBuffer(_sortKernel, PartitionIndexBufferID, _partitionIndexBuffer);
            _sortCs.SetBuffer(_sortKernel, PartitionDescriptorBufferID, _partitionDescriptorBuffer);
            for (int i = 0; i < RadixStepCount; i++)
            {
                _sortCs.SetInt(CurrentPassRadixShiftID, i << 3);
                _sortCs.SetBuffer(_sortKernel, KeyInBufferID, i % 2 == 0 ? keyBuffer : _tempKeyBuffer);
                _sortCs.SetBuffer(_sortKernel, KeyOutBufferID, i % 2 == 0 ? _tempKeyBuffer : keyBuffer);
                _sortCs.SetBuffer(_sortKernel, IndexInBufferID, i % 2 == 0 ? indexBuffer : _tempIndexBuffer);
                _sortCs.SetBuffer(_sortKernel, IndexOutBufferID, i % 2 == 0 ? _tempIndexBuffer : indexBuffer);
                _sortCs.Dispatch(_sortKernel, SortKernelDispatchGroupSize(sortCount), 1, 1);
            }
        }

        /// <summary>
        /// Sorts the key and index buffers using the specified sort count,
        /// and dispatches GPU compute work using the provided CommandBuffer.
        /// </summary>
        /// <param name="cmd">The command buffer to record compute dispatches into.</param>
        /// <param name="keyBuffer">
        /// The key buffer to be sorted.
        /// The buffer must have a stride of 4 bytes.
        /// Each element must match the KeyType specified during Init().
        /// </param>
        /// <param name="indexBuffer">
        /// The index buffer to be sorted alongside the key buffer.
        /// The buffer must have a stride of 4 bytes.
        /// </param>
        /// <param name="sortCount">
        /// Number of elements to sort (starting from the beginning of the buffer).
        /// If -1, the full buffer length is used.
        /// </param>
        public void Sort(CommandBuffer cmd, GraphicsBuffer keyBuffer, GraphicsBuffer indexBuffer, int sortCount = -1)
        {
            if (!Inited)
                throw new ArgumentException("RadixSort is not initialized.");
            if (DispatchMode == DispatchMode.Indirect)
                throw new ArgumentException("The sort count must be passed via a GraphicsBuffer when using Indirect dispatch mode.");

            if (sortCount < 0) sortCount = keyBuffer.count;
            if (sortCount == 0) return;

            if (keyBuffer.stride != sizeof(uint) || indexBuffer.stride != sizeof(uint))
                throw new ArgumentException("The stride of keyBuffer and indexBuffer must be 4 bytes.");
            if (sortCount > MaxSortCount)
                throw new ArgumentException("The sort count must be less than or equal to the buffer size specified in Init.");
            if (sortCount > keyBuffer.count || sortCount > indexBuffer.count)
                throw new ArgumentException("The sort count must be less than or equal to the buffer size of keyBuffer and indexBuffer.");

            foreach (var cs in _computeShaders)
            {
                cmd.SetComputeIntParam(cs, SortCountID, sortCount);
                int groupCount = SortKernelDispatchGroupSize(sortCount);
                cmd.SetComputeIntParam(cs, GroupID, groupCount);
            }

            // init buffers
            cmd.SetComputeBufferParam(_initCs, _initKernel, BucketCountBufferID, _bucketCountBuffer);
            cmd.SetComputeBufferParam(_initCs, _initKernel, PartitionIndexBufferID, _partitionIndexBuffer);
            cmd.SetComputeBufferParam(_initCs, _initKernel, PartitionDescriptorBufferID, _partitionDescriptorBuffer);
            cmd.DispatchCompute(_initCs, _initKernel, InitKernelDispatchGroupSize, 1, 1);

            // build radix bucket global histogram
            cmd.SetComputeBufferParam(_buildCs, _buildKernel, BucketCountBufferID, _bucketCountBuffer);
            cmd.SetComputeBufferParam(_buildCs, _buildKernel, KeyInBufferID, keyBuffer);
            cmd.DispatchCompute(_buildCs, _buildKernel, BuildKernelDispatchGroupSize(sortCount), 1, 1);

            // scan radix bucket global histogram
            cmd.SetComputeBufferParam(_scanCs, _scanKernel, BucketCountBufferID, _bucketCountBuffer);
            cmd.SetComputeBufferParam(_scanCs, _scanKernel, PartitionDescriptorBufferID, _partitionDescriptorBuffer);
            cmd.DispatchCompute(_scanCs, _scanKernel, ScanKernelDispatchGroupSize, 1, 1);

            // sort onesweep
            cmd.SetComputeBufferParam(_sortCs, _sortKernel, PartitionIndexBufferID, _partitionIndexBuffer);
            cmd.SetComputeBufferParam(_sortCs, _sortKernel, PartitionDescriptorBufferID, _partitionDescriptorBuffer);
            for (int i = 0; i < RadixStepCount; i++)
            {
                cmd.SetComputeIntParam(_sortCs, CurrentPassRadixShiftID, i << 3);
                cmd.SetComputeBufferParam(_sortCs, _sortKernel, KeyInBufferID, i % 2 == 0 ? keyBuffer : _tempKeyBuffer);
                cmd.SetComputeBufferParam(_sortCs, _sortKernel, KeyOutBufferID, i % 2 == 0 ? _tempKeyBuffer : keyBuffer);
                cmd.SetComputeBufferParam(_sortCs, _sortKernel, IndexInBufferID, i % 2 == 0 ? indexBuffer : _tempIndexBuffer);
                cmd.SetComputeBufferParam(_sortCs, _sortKernel, IndexOutBufferID, i % 2 == 0 ? _tempIndexBuffer : indexBuffer);
                cmd.DispatchCompute(_sortCs, _sortKernel, SortKernelDispatchGroupSize(sortCount), 1, 1);
            }
        }

        /// <summary>
        /// Sorts the key and index buffers using the sort count provided via a GraphicsBuffer.
        /// </summary>
        /// <param name="keyBuffer">
        /// The key buffer to be sorted.
        /// The buffer must have a stride of 4 bytes.
        /// Each element must match the KeyType specified during Init().
        /// </param>
        /// <param name="indexBuffer">
        /// The index buffer to be sorted alongside the key buffer.
        /// The buffer must have a stride of 4 bytes.
        /// </param>
        /// <param name="sortCountBuffer">
        /// The buffer that contains the sort count.
        /// The buffer must have a stride of 4 bytes (i.e., each element must be a uint).
        /// </param>
        /// <param name="sortCountBufferOffset">
        /// Offset within the sort count buffer, in units of 4 bytes (i.e., uint offset).
        /// </param>
        public void Sort(GraphicsBuffer keyBuffer, GraphicsBuffer indexBuffer, GraphicsBuffer sortCountBuffer, int sortCountBufferOffset)
        {
            if (!Inited)
                throw new ArgumentException("RadixSort is not initialized.");
            if (DispatchMode == DispatchMode.Direct)
                throw new ArgumentException("The sort count must be passed as an argument when using Direct dispatch mode.");

            if (keyBuffer.stride != sizeof(uint) || indexBuffer.stride != sizeof(uint))
                throw new ArgumentException("The stride of keyBuffer and indexBuffer must be 4 bytes.");

            if (sortCountBuffer.stride != sizeof(uint))
                throw new ArgumentException("The stride of sortCountBuffer must be 4 bytes.");

            // precompute for indirect dispatch
            _precomputeCs.SetInt(BuildKernelItemsPerGroupID, BuildKernelItemsPerThread * BuildKernelThreadsPerGroup);
            _precomputeCs.SetInt(SortKernelItemsPerGroupID, SortKernelItemsPerThread * SortKernelThreadsPerGroup);
            _precomputeCs.SetInt(MaxSortCountID, Mathf.Min(MaxSortCount, Mathf.Min(keyBuffer.count, indexBuffer.count)));
            _precomputeCs.SetBuffer(_precomputeKernel, SortCountBufferID, sortCountBuffer);
            _precomputeCs.SetInt(SortCountBufferOffsetID, sortCountBufferOffset);
            _precomputeCs.SetBuffer(_precomputeKernel, BuildKernelDispatchArgsBufferID, _buildKernelDispatchArgsBuffer);
            _precomputeCs.SetBuffer(_precomputeKernel, SortKernelDispatchArgsBufferID, _sortKernelDispatchArgsBuffer);
            _precomputeCs.SetBuffer(_precomputeKernel, SortCountGroupCountBufferID, _sortCountGroupCountBuffer);
            _precomputeCs.Dispatch(_precomputeKernel, PrecomputeKernelDispatchGroupSize, 1, 1);

            for (int i = 0; i < _computeShaders.Length; i++)
            {
                var cs = _computeShaders[i];
                var kernel = _kernels[i];
                cs.SetBuffer(kernel, SortCountGroupCountBufferID, _sortCountGroupCountBuffer);
            }

            // init buffers
            _initCs.SetBuffer(_initKernel, BucketCountBufferID, _bucketCountBuffer);
            _initCs.SetBuffer(_initKernel, PartitionIndexBufferID, _partitionIndexBuffer);
            _initCs.SetBuffer(_initKernel, PartitionDescriptorBufferID, _partitionDescriptorBuffer);
            _initCs.Dispatch(_initKernel, InitKernelDispatchGroupSize, 1, 1);

            // build radix bucket global histogram
            _buildCs.SetBuffer(_buildKernel, BucketCountBufferID, _bucketCountBuffer);
            _buildCs.SetBuffer(_buildKernel, KeyInBufferID, keyBuffer);
            _buildCs.DispatchIndirect(_buildKernel, _buildKernelDispatchArgsBuffer);

            // scan radix bucket global histogram
            _scanCs.SetBuffer(_scanKernel, BucketCountBufferID, _bucketCountBuffer);
            _scanCs.SetBuffer(_scanKernel, PartitionDescriptorBufferID, _partitionDescriptorBuffer);
            _scanCs.Dispatch(_scanKernel, ScanKernelDispatchGroupSize, 1, 1);

            // sort onesweep
            _sortCs.SetBuffer(_sortKernel, PartitionIndexBufferID, _partitionIndexBuffer);
            _sortCs.SetBuffer(_sortKernel, PartitionDescriptorBufferID, _partitionDescriptorBuffer);
            for (int i = 0; i < RadixStepCount; i++)
            {
                _sortCs.SetInt(CurrentPassRadixShiftID, i << 3);
                _sortCs.SetBuffer(_sortKernel, KeyInBufferID, i % 2 == 0 ? keyBuffer : _tempKeyBuffer);
                _sortCs.SetBuffer(_sortKernel, KeyOutBufferID, i % 2 == 0 ? _tempKeyBuffer : keyBuffer);
                _sortCs.SetBuffer(_sortKernel, IndexInBufferID, i % 2 == 0 ? indexBuffer : _tempIndexBuffer);
                _sortCs.SetBuffer(_sortKernel, IndexOutBufferID, i % 2 == 0 ? _tempIndexBuffer : indexBuffer);
                _sortCs.DispatchIndirect(_sortKernel, _sortKernelDispatchArgsBuffer);
            }
        }

        /// <summary>
        /// Sorts the key and index buffers using the sort count provided via a GraphicsBuffer,
        /// and dispatches GPU compute workloads using the provided CommandBuffer.
        /// </summary>
        /// <param name="cmd">The command buffer to record compute dispatches into.</param>
        /// <param name="keyBuffer">
        /// The key buffer to be sorted.
        /// The buffer must have a stride of 4 bytes.
        /// Each element must match the KeyType specified during Init().
        /// </param>
        /// <param name="indexBuffer">
        /// The index buffer to be sorted alongside the key buffer.
        /// The buffer must have a stride of 4 bytes.
        /// </param>
        /// <param name="sortCountBuffer">
        /// The buffer that contains the sort count.
        /// The buffer must have a stride of 4 bytes (i.e., each element must be a uint).
        /// </param>
        /// <param name="sortCountBufferOffset">
        /// Offset within the sort count buffer, in units of 4 bytes (i.e., uint offset).
        /// </param>
        public void Sort(CommandBuffer cmd, GraphicsBuffer keyBuffer, GraphicsBuffer indexBuffer, GraphicsBuffer sortCountBuffer, int sortCountBufferOffset)
        {
            if (!Inited)
                throw new ArgumentException("RadixSort is not initialized.");
            if (DispatchMode == DispatchMode.Direct)
                throw new ArgumentException("The sort count must be passed as an argument when using Direct dispatch mode.");

            if (keyBuffer.stride != sizeof(uint) || indexBuffer.stride != sizeof(uint))
                throw new ArgumentException("The stride of keyBuffer and indexBuffer must be 4 bytes.");

            if (sortCountBuffer.stride != sizeof(uint))
                throw new ArgumentException("The stride of sortCountBuffer must be 4 bytes.");

            // precompute for indirect dispatch
            cmd.SetComputeIntParam(_precomputeCs, BuildKernelItemsPerGroupID, BuildKernelItemsPerThread * BuildKernelThreadsPerGroup);
            cmd.SetComputeIntParam(_precomputeCs, SortKernelItemsPerGroupID, SortKernelItemsPerThread * SortKernelThreadsPerGroup);
            cmd.SetComputeIntParam(_precomputeCs, MaxSortCountID, Mathf.Min(MaxSortCount, Mathf.Min(keyBuffer.count, indexBuffer.count)));
            cmd.SetComputeBufferParam(_precomputeCs, _precomputeKernel, SortCountBufferID, sortCountBuffer);
            cmd.SetComputeIntParam(_precomputeCs, SortCountBufferOffsetID, sortCountBufferOffset);
            cmd.SetComputeBufferParam(_precomputeCs, _precomputeKernel, BuildKernelDispatchArgsBufferID, _buildKernelDispatchArgsBuffer);
            cmd.SetComputeBufferParam(_precomputeCs, _precomputeKernel, SortKernelDispatchArgsBufferID, _sortKernelDispatchArgsBuffer);
            cmd.SetComputeBufferParam(_precomputeCs, _precomputeKernel, SortCountGroupCountBufferID, _sortCountGroupCountBuffer);
            cmd.DispatchCompute(_precomputeCs, _precomputeKernel, PrecomputeKernelDispatchGroupSize, 1, 1);

            for (int i = 0; i < _computeShaders.Length; i++)
            {
                var cs = _computeShaders[i];
                var kernel = _kernels[i];
                cmd.SetComputeBufferParam(cs, kernel, SortCountGroupCountBufferID, _sortCountGroupCountBuffer);
            }

            // init buffers
            cmd.SetComputeBufferParam(_initCs, _initKernel, BucketCountBufferID, _bucketCountBuffer);
            cmd.SetComputeBufferParam(_initCs, _initKernel, PartitionIndexBufferID, _partitionIndexBuffer);
            cmd.SetComputeBufferParam(_initCs, _initKernel, PartitionDescriptorBufferID, _partitionDescriptorBuffer);
            cmd.DispatchCompute(_initCs, _initKernel, InitKernelDispatchGroupSize, 1, 1);

            // build radix bucket global histogram
            cmd.SetComputeBufferParam(_buildCs, _buildKernel, BucketCountBufferID, _bucketCountBuffer);
            cmd.SetComputeBufferParam(_buildCs, _buildKernel, KeyInBufferID, keyBuffer);
            cmd.DispatchCompute(_buildCs, _buildKernel, _buildKernelDispatchArgsBuffer, 0);

            // scan radix bucket global histogram
            cmd.SetComputeBufferParam(_scanCs, _scanKernel, BucketCountBufferID, _bucketCountBuffer);
            cmd.SetComputeBufferParam(_scanCs, _scanKernel, PartitionDescriptorBufferID, _partitionDescriptorBuffer);
            cmd.DispatchCompute(_scanCs, _scanKernel, ScanKernelDispatchGroupSize, 1, 1);

            // sort onesweep
            cmd.SetComputeBufferParam(_sortCs, _sortKernel, PartitionIndexBufferID, _partitionIndexBuffer);
            cmd.SetComputeBufferParam(_sortCs, _sortKernel, PartitionDescriptorBufferID, _partitionDescriptorBuffer);
            for (int i = 0; i < RadixStepCount; i++)
            {
                cmd.SetComputeIntParam(_sortCs, CurrentPassRadixShiftID, i << 3);
                cmd.SetComputeBufferParam(_sortCs, _sortKernel, KeyInBufferID, i % 2 == 0 ? keyBuffer : _tempKeyBuffer);
                cmd.SetComputeBufferParam(_sortCs, _sortKernel, KeyOutBufferID, i % 2 == 0 ? _tempKeyBuffer : keyBuffer);
                cmd.SetComputeBufferParam(_sortCs, _sortKernel, IndexInBufferID, i % 2 == 0 ? indexBuffer : _tempIndexBuffer);
                cmd.SetComputeBufferParam(_sortCs, _sortKernel, IndexOutBufferID, i % 2 == 0 ? _tempIndexBuffer : indexBuffer);
                cmd.DispatchCompute(_sortCs, _sortKernel, _sortKernelDispatchArgsBuffer, 0);
            }
        }

        /// <summary>
        /// Releases all resources used by the RadixSort instance.
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
            if (_tempKeyBuffer is not null) { _tempKeyBuffer.Release(); _tempKeyBuffer = null; }
            if (_tempIndexBuffer is not null) { _tempIndexBuffer.Release(); _tempIndexBuffer = null; }
            if (_bucketCountBuffer is not null) { _bucketCountBuffer.Release(); _bucketCountBuffer = null; }
            if (_partitionIndexBuffer is not null) { _partitionIndexBuffer.Release(); _partitionIndexBuffer = null; }
            if (_partitionDescriptorBuffer is not null) { _partitionDescriptorBuffer.Release(); _partitionDescriptorBuffer = null; }
            if (_sortCountGroupCountBuffer is not null) { _sortCountGroupCountBuffer.Release(); _sortCountGroupCountBuffer = null; }
            if (_buildKernelDispatchArgsBuffer is not null) { _buildKernelDispatchArgsBuffer.Release(); _buildKernelDispatchArgsBuffer = null; }
            if (_sortKernelDispatchArgsBuffer is not null) { _sortKernelDispatchArgsBuffer.Release(); _sortKernelDispatchArgsBuffer = null; }
        }
        #endregion
    }
}