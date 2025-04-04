using UnityEngine;

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