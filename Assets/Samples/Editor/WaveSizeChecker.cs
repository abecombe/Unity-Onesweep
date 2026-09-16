using System;
using UnityEditor;
using UnityEngine;

internal static class WaveSizeChecker
{
    private const string ComputeShaderPath = "Assets/Samples/Editor/WaveSizeChecker.compute";
    private static readonly int ResultId = Shader.PropertyToID("result");

    [MenuItem("Tools/Onesweep/Check Wave Size")]
    private static void CheckWaveSize()
    {
        if (!SystemInfo.supportsComputeShaders)
        {
            Debug.LogError("Wave size check failed: Compute shaders are not supported.");
            return;
        }

        if (SystemInfo.graphicsDeviceType != UnityEngine.Rendering.GraphicsDeviceType.Direct3D12)
        {
            Debug.LogError($"Wave size check failed: DirectX 12 is required, but current Graphics API is {SystemInfo.graphicsDeviceType}.");
            return;
        }

        var computeShader = AssetDatabase.LoadAssetAtPath<ComputeShader>(ComputeShaderPath);
        if (computeShader == null)
        {
            Debug.LogError($"Wave size check failed: {ComputeShaderPath} was not found.");
            return;
        }

        try
        {
            var kernel = computeShader.FindKernel("CheckWaveSize");
            using var resultBuffer = new ComputeBuffer(1, sizeof(uint));
            computeShader.SetBuffer(kernel, ResultId, resultBuffer);
            computeShader.Dispatch(kernel, 1, 1, 1);

            var result = new uint[1];
            resultBuffer.GetData(result);
            Debug.Log($"Wave Size: {result[0]} (GPU: {SystemInfo.graphicsDeviceName}, API: {SystemInfo.graphicsDeviceType})");
        }
        catch (Exception exception)
        {
            Debug.LogError($"Wave size check failed: {exception.Message}");
        }
    }
}