using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace Onesweep
{
    /// <summary>
    /// Interface for GPU radix sorting implementations.
    /// </summary>
    public interface ISorter : IDisposable
    {
        /// <summary>
        /// Gets a value indicating whether the sorter has been initialized.
        /// </summary>
        bool Inited { get; }

        /// <summary>
        /// Gets the sorting algorithm used by this sorter instance.
        /// </summary>
        SortingAlgorithm SortingAlgorithm { get; }

        /// <summary>
        /// Gets the configured sorting mode (key-only or key-payload).
        /// </summary>
        SortMode SortMode { get; }

        /// <summary>
        /// Gets the key type used for sorting.
        /// </summary>
        KeyType KeyType { get; }

        /// <summary>
        /// Gets the sorting order (ascending or descending).
        /// </summary>
        SortingOrder SortingOrder { get; }

        /// <summary>
        /// Gets the dispatch mode (direct or indirect).
        /// </summary>
        DispatchMode DispatchMode { get; }

        /// <summary>
        /// Gets the GPU wave size used for sorting.
        /// </summary>
        WaveSize WaveSize { get; }

        /// <summary>
        /// Gets the maximum number of elements this sorter instance can handle.
        /// </summary>
        int MaxSortCount { get; }

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
        IDisposable Init(OnesweepComputeConfig onesweepComputeConfig, int maxSortCount, SortMode sortMode, KeyType keyType, SortingOrder sortingOrder, DispatchMode dispatchMode, WaveSize waveSize, bool forceClearBuffers = false);

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
        void Sort(GraphicsBuffer keyBuffer, GraphicsBuffer payloadBuffer, int sortCount = -1);

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
        void Sort(CommandBuffer cmd, GraphicsBuffer keyBuffer, GraphicsBuffer payloadBuffer, int sortCount = -1);

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
        void Sort(GraphicsBuffer keyBuffer, GraphicsBuffer payloadBuffer, GraphicsBuffer sortCountBuffer, uint sortCountBufferOffset);

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
        void Sort(CommandBuffer cmd, GraphicsBuffer keyBuffer, GraphicsBuffer payloadBuffer, GraphicsBuffer sortCountBuffer, uint sortCountBufferOffset);
    }
}