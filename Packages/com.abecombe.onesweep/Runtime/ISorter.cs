using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace Onesweep
{
    /// <summary>
    /// Interface for GPU sorting implementations.
    /// </summary>
    public interface ISorter : IDisposable
    {
        /// <summary>
        /// Gets a value indicating whether the sorter has been initialized.
        /// </summary>
        bool Inited { get; }

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
        /// <param name="onesweepComputeConfig">Compute shader configuration.</param>
        /// <param name="maxSortCount">Maximum number of elements to sort.</param>
        /// <param name="keyType">Data type of the keys to sort.</param>
        /// <param name="sortingOrder">Order of sorting (ascending/descending).</param>
        /// <param name="dispatchMode">Dispatch mode for compute shaders.</param>
        /// <param name="waveSize">GPU wave size.</param>
        /// <param name="forceClearBuffers">Whether to force clear existing buffers.</param>
        /// <returns>An IDisposable instance, typically the sorter itself.</returns>
        IDisposable Init(OnesweepComputeConfig onesweepComputeConfig, int maxSortCount, KeyType keyType, SortingOrder sortingOrder, DispatchMode dispatchMode, WaveSize waveSize, bool forceClearBuffers = false);

        /// <summary>
        /// Sorts the key and index buffers.
        /// </summary>
        /// <param name="keyBuffer">Buffer containing the keys to sort.</param>
        /// <param name="indexBuffer">Buffer containing the indices to sort alongside keys.</param>
        /// <param name="sortCount">Number of elements to sort. If -1, sorts the entire buffer.</param>
        void Sort(GraphicsBuffer keyBuffer, GraphicsBuffer indexBuffer, int sortCount = -1);

        /// <summary>
        /// Sorts the key and index buffers using a CommandBuffer.
        /// </summary>
        /// <param name="cmd">Command buffer to record dispatches.</param>
        /// <param name="keyBuffer">Buffer containing the keys to sort.</param>
        /// <param name="indexBuffer">Buffer containing the indices to sort alongside keys.</param>
        /// <param name="sortCount">Number of elements to sort. If -1, sorts the entire buffer.</param>
        void Sort(CommandBuffer cmd, GraphicsBuffer keyBuffer, GraphicsBuffer indexBuffer, int sortCount = -1);

        /// <summary>
        /// Sorts the key and index buffers using a sort count from a GraphicsBuffer (for indirect dispatch).
        /// </summary>
        /// <param name="keyBuffer">Buffer containing the keys to sort.</param>
        /// <param name="indexBuffer">Buffer containing the indices to sort alongside keys.</param>
        /// <param name="sortCountBuffer">Buffer containing the number of elements to sort.</param>
        /// <param name="sortCountBufferOffset">Offset in sortCountBuffer where the count is located (in uints).</param>
        void Sort(GraphicsBuffer keyBuffer, GraphicsBuffer indexBuffer, GraphicsBuffer sortCountBuffer, int sortCountBufferOffset);

        /// <summary>
        /// Sorts the key and index buffers using a CommandBuffer and a sort count from a GraphicsBuffer (for indirect dispatch).
        /// </summary>
        /// <param name="cmd">Command buffer to record dispatches.</param>
        /// <param name="keyBuffer">Buffer containing the keys to sort.</param>
        /// <param name="indexBuffer">Buffer containing the indices to sort alongside keys.</param>
        /// <param name="sortCountBuffer">Buffer containing the number of elements to sort.</param>
        /// <param name="sortCountBufferOffset">Offset in sortCountBuffer where the count is located (in uints).</param>
        void Sort(CommandBuffer cmd, GraphicsBuffer keyBuffer, GraphicsBuffer indexBuffer, GraphicsBuffer sortCountBuffer, int sortCountBufferOffset);
    }
}