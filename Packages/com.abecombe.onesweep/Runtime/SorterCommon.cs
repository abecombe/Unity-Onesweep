using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace Onesweep
{
    /// <summary>
    /// Specifies the underlying GPU radix sort algorithm implementation.
    /// </summary>
    public enum SortingAlgorithm
    {
        /// <summary>
        /// The "Onesweep" algorithm, designed for high performance with potentially fewer passes.
        /// May have stability or performance variability on some hardware/drivers.
        /// </summary>
        Onesweep = 0,
        /// <summary>
        /// A conventional multi-pass radix sort algorithm, generally offering greater stability
        /// and more predictable performance.
        /// </summary>
        Traditional
    }

    /// <summary>
    /// Specifies the sorting mode, determining whether payloads are sorted alongside keys.
    /// </summary>
    public enum SortMode
    {
        /// <summary>
        /// Sorts only the key buffer. The payload buffer is ignored and can be null.
        /// </summary>
        KeyOnly = 0,
        /// <summary>
        /// Sorts the key buffer and an accompanying payload buffer.
        /// The payload buffer must be provided.
        /// </summary>
        KeyPayload
    }

    /// <summary>
    /// Defines the data type of the keys to be sorted.
    /// </summary>
    public enum KeyType
    {
        /// <summary>
        /// Keys are unsigned 32-bit integers.
        /// </summary>
        UInt = 0,
        /// <summary>
        /// Keys are signed 32-bit integers.
        /// </summary>
        Int,
        /// <summary>
        /// Keys are 32-bit single-precision floating-point numbers.
        /// </summary>
        Float
    }

    /// <summary>
    /// Specifies the desired order for the sort operation.
    /// </summary>
    public enum SortingOrder
    {
        /// <summary>
        /// Sorts keys from the smallest to the largest value.
        /// </summary>
        Ascending = 0,
        /// <summary>
        /// Sorts keys from the largest to the smallest value.
        /// </summary>
        Descending
    }

    /// <summary>
    /// Specifies how the sort compute shaders are dispatched and how the sort count is provided.
    /// </summary>
    /// <remarks>
    /// Direct dispatch mode: the sort count is passed as an argument.
    /// Indirect dispatch mode: the sort count is passed via a GraphicsBuffer.
    /// </remarks>
    public enum DispatchMode
    {
        /// <summary>
        /// Uses direct dispatch; sort count is provided directly by the CPU.
        /// </summary>
        Direct = 0,
        /// <summary>
        /// Uses indirect dispatch; sort count is read from a GraphicsBuffer on the GPU.
        /// </summary>
        Indirect
    }

    internal static class SorterCommon
    {
        /// <summary>
        /// Sets the shader keywords for the compute shader based on the provided parameters.
        /// </summary>
        public static void SetShaderKeywords(ComputeShader cs, SortMode sortMode, KeyType keyType, SortingOrder sortingOrder, DispatchMode dispatchMode)
        {
            switch (sortMode)
            {
                case SortMode.KeyOnly:
                    cs.EnableKeyword("KEY_ONLY");
                    cs.DisableKeyword("KEY_PAYLOAD");
                    break;
                case SortMode.KeyPayload:
                    cs.DisableKeyword("KEY_ONLY");
                    cs.EnableKeyword("KEY_PAYLOAD");
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(sortMode), sortMode, null);
            }
            switch (keyType)
            {
                case KeyType.UInt:
                    cs.EnableKeyword("KEY_TYPE_UINT");
                    cs.DisableKeyword("KEY_TYPE_INT");
                    cs.DisableKeyword("KEY_TYPE_FLOAT");
                    break;
                case KeyType.Int:
                    cs.DisableKeyword("KEY_TYPE_UINT");
                    cs.EnableKeyword("KEY_TYPE_INT");
                    cs.DisableKeyword("KEY_TYPE_FLOAT");
                    break;
                case KeyType.Float:
                    cs.DisableKeyword("KEY_TYPE_UINT");
                    cs.DisableKeyword("KEY_TYPE_INT");
                    cs.EnableKeyword("KEY_TYPE_FLOAT");
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(keyType), keyType, null);
            }
            switch (sortingOrder)
            {
                case SortingOrder.Ascending:
                    cs.EnableKeyword("SORTING_ORDER_ASCENDING");
                    cs.DisableKeyword("SORTING_ORDER_DESCENDING");
                    break;
                case SortingOrder.Descending:
                    cs.DisableKeyword("SORTING_ORDER_ASCENDING");
                    cs.EnableKeyword("SORTING_ORDER_DESCENDING");
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(sortingOrder), sortingOrder, null);
            }
            switch (dispatchMode)
            {
                case DispatchMode.Direct:
                    cs.EnableKeyword("USE_DIRECT_DISPATCH");
                    cs.DisableKeyword("USE_INDIRECT_DISPATCH");
                    break;
                case DispatchMode.Indirect:
                    cs.DisableKeyword("USE_DIRECT_DISPATCH");
                    cs.EnableKeyword("USE_INDIRECT_DISPATCH");
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(dispatchMode), dispatchMode, null);
            }
        }

        /// <summary>
        /// Checks if the current graphics device type is Direct3D12.
        /// </summary>
        public static bool GraphicsDeviceTypeIsDirect3D12()
        {
            return SystemInfo.graphicsDeviceType == GraphicsDeviceType.Direct3D12;
        }

        /// <summary>
        /// Gets the current graphics device type.
        /// </summary>
        public static GraphicsDeviceType GetGraphicsDeviceType()
        {
            return SystemInfo.graphicsDeviceType;
        }
    }
}