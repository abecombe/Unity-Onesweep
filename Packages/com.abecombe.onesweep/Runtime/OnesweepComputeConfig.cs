using UnityEngine;

namespace Onesweep
{
    /// <summary>
    /// Configuration for Onesweep compute shaders.
    /// </summary>
    /// <remarks>
    /// This class holds references to the compute shaders used in the Onesweep Package.
    /// </remarks>
    [CreateAssetMenu(fileName = "OnesweepComputeConfig", menuName = "Onesweep/ComputeConfig")]
    public class OnesweepComputeConfig : ScriptableObject
    {
        [Header("Onesweep RadixSort Compute Shaders")]
        public ComputeShader WaveSizeCs;
        public ComputeShader PrecomputeCs;
        public ComputeShader InitCs;
        public ComputeShader BuildCs;
        public ComputeShader ScanCs;
        public ComputeShader SortCs;

        [Header("Traditional Radix Sort Compute Shaders")]
        public ComputeShader TraditionalSortCs;
    }
}