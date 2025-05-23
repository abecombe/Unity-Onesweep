using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace Onesweep
{
    public enum KeyType
    {
        UInt = 0,
        Int,
        Float
    }
    public enum SortingOrder
    {
        Ascending = 0,
        Descending
    }
    /// <remarks>
    /// Direct dispatch mode: the sort count is passed as an argument
    /// Indirect dispatch mode: the sort count is passed via a GraphicsBuffer
    /// </remarks>
    public enum DispatchMode
    {
        Direct = 0,
        Indirect
    }
    public enum WaveSize
    {
        WaveSize32 = 32,
        WaveSize64 = 64,
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
        public static void SetShaderKeywords(ComputeShader cs, KeyType keyType, SortingOrder sortingOrder, DispatchMode dispatchMode, WaveSize waveSize)
        {
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
    }
}