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
        KeyOnly,
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

    /// <summary>
    /// Specifies the GPU wave size (number of threads in a hardware execution unit) to be used or targeted.
    /// </summary>
    public enum WaveSize
    {
        /// <summary>
        /// Target or configure for a wave size of 32 lanes. Typically used by NVIDIA GPUs.
        /// </summary>
        WaveSize32 = 32,
        /// <summary>
        /// Target or configure for a wave size of 64 lanes. Typically used by AMD GPUs.
        /// </summary>
        WaveSize64 = 64,
        /// <summary>
        /// Attempt to auto-detect the optimal.
        /// </summary>
        Unknown = 0
    }

    internal static class SorterCommon
    {
        private static readonly int WaveSizeBufferID = Shader.PropertyToID("wave_size_buffer");

        public static WaveSize StoredWaveSize { get; private set; } = WaveSize.Unknown; // for storing the wave size of the device

        /// <summary>
        /// Gets and Stores the wave size from the compute shader.
        /// </summary>
        /// <param name="onesweepComputeConfig">
        /// Compute shader configuration for Onesweep.
        /// </param>
        /// <param name="waveSize">Outputs the detected wave size.</param>
        /// <returns>
        /// Returns the wave size (32 or 64). If the size is something else, returns WaveSize.Unknown.
        /// </returns>
        public static WaveSize GetStoreWaveSize(OnesweepComputeConfig onesweepComputeConfig, out uint waveSize)
        {
            var waveSizeCs = onesweepComputeConfig.WaveSizeCs;
            var waveSizeKernel = waveSizeCs.FindKernel("GetWaveSize");
            var waveSizeBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1, sizeof(uint));
            waveSizeCs.SetBuffer(waveSizeKernel, WaveSizeBufferID, waveSizeBuffer);
            waveSizeCs.Dispatch(waveSizeKernel, 1, 1, 1);
            uint[] waveSizeData = new uint[1];
            waveSizeBuffer.GetData(waveSizeData);
            waveSizeBuffer.Release();
            waveSize = waveSizeData[0];

            StoredWaveSize = waveSizeData[0] switch
            {
                32 => WaveSize.WaveSize32,
                64 => WaveSize.WaveSize64,
                _ => WaveSize.Unknown
            };

            return StoredWaveSize;
        }

        /// <summary>
        /// Sets the shader keywords for the compute shader based on the provided parameters.
        /// </summary>
        public static void SetShaderKeywords(ComputeShader cs, SortMode sortMode, KeyType keyType, SortingOrder sortingOrder, DispatchMode dispatchMode, WaveSize waveSize)
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
            switch (waveSize)
            {
                case WaveSize.WaveSize32:
                    cs.EnableKeyword("WAVE_SIZE_32");
                    cs.DisableKeyword("WAVE_SIZE_64");
                    break;
                case WaveSize.WaveSize64:
                    cs.DisableKeyword("WAVE_SIZE_32");
                    cs.EnableKeyword("WAVE_SIZE_64");
                    break;
                case WaveSize.Unknown:
                default:
                    throw new ArgumentOutOfRangeException(nameof(waveSize), waveSize, null);
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