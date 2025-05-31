using System;
using System.Linq;
using Onesweep;
using UnityEngine;
using UnityEngine.Rendering;

#if UNITY_EDITOR
using UnityEditor;
#endif

using Random = UnityEngine.Random;

using type = System.UInt32;

public class SortSample : MonoBehaviour
{
    [SerializeField, ReadOnlyOnPlay] private SortingAlgorithm _sortingAlgorithm = SortingAlgorithm.Onesweep;
    [SerializeField, ReadOnlyOnPlay] private DispatchMode _dispatchMode = DispatchMode.Direct;
    [SerializeField, ReadOnlyOnPlay] private bool _useCommandBuffer = false;
    [SerializeField, ReadOnlyOnPlay] private int _numData = 100;
    [SerializeField, ReadOnlyOnPlay] private int _randomValueMax = 100000;
    [SerializeField, ReadOnlyOnPlay] private int _randomSeed = 0;
    [SerializeField, ReadOnlyOnPlay] private OnesweepComputeConfig _config;

    private ISorter _sorter;

    private GraphicsBuffer _keyBuffer;
    private GraphicsBuffer _payloadBuffer;
    private GraphicsBuffer _keyTempBuffer;
    private GraphicsBuffer _payloadTempBuffer;
    private GraphicsBuffer _sortCountBuffer;
    private CommandBuffer _commandBuffer;

    private ComputeShader _copyCs;
    private int _copyKernel;
    private const int CopyKernelDispatchGroupSize = 128;

    private static readonly int KeyBufferID = Shader.PropertyToID("key_buffer");
    private static readonly int PayloadBufferID = Shader.PropertyToID("payload_buffer");
    private static readonly int KeyTempBufferID = Shader.PropertyToID("key_temp_buffer");
    private static readonly int PayloadTempBufferID = Shader.PropertyToID("payload_temp_buffer");
    private static readonly int NumElementsID = Shader.PropertyToID("num_elements");

    private type[] _keyArray;
    private uint[] _payloadArray;
    private KeyPayloadCombine[] _combinedDataArray;

    private struct KeyPayloadCombine : IEquatable<KeyPayloadCombine>
    {
        public type Key;
        public uint Payload;

        public KeyPayloadCombine(type key, uint payload)
        {
            this.Key = key;
            this.Payload = payload;
        }

        public bool Equals(KeyPayloadCombine other)
        {
            return Key == other.Key && Payload == other.Payload;
        }
    }

    private void Start()
    {
        _sorter = _sortingAlgorithm switch
        {
            SortingAlgorithm.Onesweep => new OnesweepSorter(),
            SortingAlgorithm.Traditional => new TraditionalSorter(),
            _ => throw new ArgumentOutOfRangeException(nameof(_sortingAlgorithm), _sortingAlgorithm, null)
        };

        _sorter.Init(_config, _numData, SortMode.KeyPayload, KeyType.UInt, SortingOrder.Ascending, _dispatchMode, WaveSize.Unknown);

        _keyBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _numData, sizeof(type));
        _payloadBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _numData, sizeof(uint));
        _keyTempBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _numData, sizeof(type));
        _payloadTempBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _numData, sizeof(uint));
        _sortCountBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 2, sizeof(uint));
        _sortCountBuffer.SetData(new[] { 0u, (uint)_numData });
        _commandBuffer = new CommandBuffer { name = "SortSampleCommandBuffer" };

        _keyArray = new type[_numData];
        _payloadArray = new uint[_numData];
        _combinedDataArray = new KeyPayloadCombine[_numData];

        Random.InitState(_randomSeed);
        for (uint i = 0; i < _numData; i++)
        {
            type value = (type)Random.Range(0, _randomValueMax + 1);
            _keyArray[i] = value;
            _payloadArray[i] = i;
            _combinedDataArray[i] = new KeyPayloadCombine(value, i);
        }
        _keyTempBuffer.SetData(_keyArray);
        _payloadTempBuffer.SetData(_payloadArray);

        _copyCs = Resources.Load<ComputeShader>("Copy");
        _copyKernel = _copyCs.FindKernel("CopySortBuffer");
    }

    private void Update()
    {
        CopySort();
    }

    private void OnDestroy()
    {
        _keyBuffer?.Release();
        _payloadBuffer?.Release();
        _keyTempBuffer?.Release();
        _payloadTempBuffer?.Release();
        _sortCountBuffer?.Release();
        _commandBuffer?.Dispose();
        _sorter?.Dispose();
    }

    private void CopySort()
    {
        if (_useCommandBuffer)
        {
            _commandBuffer.Clear();

            _commandBuffer.SetComputeIntParam(_copyCs, NumElementsID, _numData);
            _commandBuffer.SetComputeBufferParam(_copyCs, _copyKernel, KeyBufferID, _keyBuffer);
            _commandBuffer.SetComputeBufferParam(_copyCs, _copyKernel, PayloadBufferID, _payloadBuffer);
            _commandBuffer.SetComputeBufferParam(_copyCs, _copyKernel, KeyTempBufferID, _keyTempBuffer);
            _commandBuffer.SetComputeBufferParam(_copyCs, _copyKernel, PayloadTempBufferID, _payloadTempBuffer);
            _commandBuffer.DispatchCompute(_copyCs, _copyKernel, CopyKernelDispatchGroupSize, 1, 1);

            switch (_dispatchMode)
            {
                case DispatchMode.Direct:
                    _sorter.Sort(_commandBuffer, _keyBuffer, _payloadBuffer, _numData);
                    break;
                case DispatchMode.Indirect:
                    _sorter.Sort(_commandBuffer, _keyBuffer, _payloadBuffer, _sortCountBuffer, 1);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(_dispatchMode), _dispatchMode, null);
            }

            Graphics.ExecuteCommandBuffer(_commandBuffer);
        }
        else
        {
            _copyCs.SetInt(NumElementsID, _numData);
            _copyCs.SetBuffer(_copyKernel, KeyBufferID, _keyBuffer);
            _copyCs.SetBuffer(_copyKernel, PayloadBufferID, _payloadBuffer);
            _copyCs.SetBuffer(_copyKernel, KeyTempBufferID, _keyTempBuffer);
            _copyCs.SetBuffer(_copyKernel, PayloadTempBufferID, _payloadTempBuffer);
            _copyCs.Dispatch(_copyKernel, CopyKernelDispatchGroupSize, 1, 1);

            switch (_dispatchMode)
            {
                case DispatchMode.Direct:
                    _sorter.Sort(_keyBuffer, _payloadBuffer, _numData);
                    break;
                case DispatchMode.Indirect:
                    _sorter.Sort(_keyBuffer, _payloadBuffer, _sortCountBuffer, 1);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(_dispatchMode), _dispatchMode, null);
            }
        }
    }

    public void CheckSuccess()
    {
        Start();

        CopySort();

        type[] keyArray = new type[_numData];
        _keyBuffer.GetData(keyArray);
        uint[] payloadArray = new uint[_numData];
        _payloadBuffer.GetData(payloadArray);
        KeyPayloadCombine[] combinedDataArray = keyArray.Select((key, i) => new KeyPayloadCombine(key, payloadArray[i])).ToArray();

        _combinedDataArray = _combinedDataArray.OrderBy(data => data.Key).ToArray();
        _keyArray = _keyArray.OrderBy(key => key).ToArray();

        if (_combinedDataArray.SequenceEqual(combinedDataArray))
        {
            Debug.Log("Sorting Success");
        }
        else
        {
            Debug.LogError("Sorting Failure");
        }

        OnDestroy();
    }
}

#if UNITY_EDITOR
[CustomEditor(typeof(SortSample))]
public class SortSampleEditor : Editor
{
    public override void OnInspectorGUI()
    {
        base.OnInspectorGUI();
        GUILayout.Space(5f);
        if (GUILayout.Button("Check Success"))
        {
            var sortSample = target as SortSample;
            sortSample.CheckSuccess();
        }
    }
}
#endif