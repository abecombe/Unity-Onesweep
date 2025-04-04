using UnityEngine;

namespace Onesweep
{
    /// <summary>
    /// Configuration for Onesweep compute shaders.
    /// </summary>
    /// <remarks>
    /// This class holds references to the compute shaders used in the Onesweep algorithm.
    /// </remarks>
    [CreateAssetMenu(fileName = "OnesweepComputeConfig", menuName = "Onesweep/ComputeConfig")]
    public class OnesweepComputeConfig : ScriptableObject
    {
        public ComputeShader WaveSizeCs;
        public ComputeShader PrecomputeCs;
        public ComputeShader InitCs;
        public ComputeShader BuildCs;
        public ComputeShader ScanCs;
        public ComputeShader SortCs;
    }
}