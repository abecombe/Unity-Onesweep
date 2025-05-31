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
        [Header("Common Compute Shaders")]
        public ComputeShader WaveSizeCs;

        [Header("Onesweep Sorter Compute Shaders")]
        public ComputeShader OnesweepPrecomputeCs;
        public ComputeShader OnesweepInitCs;
        public ComputeShader OnesweepCountCs;
        public ComputeShader OnesweepScanCs;
        public ComputeShader OnesweepSortCs;

        [Header("Traditional Sorter Compute Shaders")]
        public ComputeShader TraditionalPrecomputeCs;
        public ComputeShader TraditionalCountCs;
        public ComputeShader TraditionalScanLocalCs;
        public ComputeShader TraditionalScanGlobalCs;
        public ComputeShader TraditionalSortCs;
    }
}