using System;
using System.Linq;
using Onesweep;
using RosettaUI;
using UnityEngine;
using UnityEngine.Rendering;

#if UNITY_EDITOR
using UnityEditor;
#endif

using Random = UnityEngine.Random;

using type = System.UInt32;

public class SortSample : MonoBehaviour
{
    [SerializeField] private int _numData = 100;
    [SerializeField] private int _randomValueMax = 100000;
    [SerializeField] private int _randomSeed = 0;
    [SerializeField] private SortingAlgorithm _sortingAlgorithm = SortingAlgorithm.Onesweep;
    [SerializeField] private SortMode _sortMode = SortMode.KeyPayload;
    [SerializeField] private DispatchMode _dispatchMode = DispatchMode.Direct;
    [SerializeField] private bool _useCommandBuffer = false;
    [SerializeField] private OnesweepComputeConfig _config;
    [SerializeField] private bool _dispatchOnlyCopyKernel = false;

    private SortingAlgorithm _currentSortingAlgorithm;
    private SortMode _currentSortMode;
    private DispatchMode _currentDispatchMode;
    private int _currentNumData;
    private string RunningKernels => _dispatchOnlyCopyKernel ? "Copy" : "Copy & Sort";

    private ISorter _sorter;
    private bool _successfllyInitialized = false;

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

    private RosettaUIRoot _rosettaUIRoot;

    private void Awake()
    {
        Init();
    }

    private void Start()
    {
        _rosettaUIRoot = FindFirstObjectByType<RosettaUIRoot>();
        _rosettaUIRoot.Build(CreateElement());
    }

    private void Update()
    {
        if (!_successfllyInitialized) return;

        if (_useCommandBuffer) _commandBuffer.Clear();
        DispatchCopyKernel();
        if (!_dispatchOnlyCopyKernel) DispatchSortKernel();
        if (_useCommandBuffer) Graphics.ExecuteCommandBuffer(_commandBuffer);
    }

    private void OnDestroy()
    {
        ReleaseObjects();
    }

    private void ReleaseObjects()
    {
        _keyBuffer?.Release(); _keyArray = null;
        _payloadBuffer?.Release(); _payloadArray = null;
        _keyTempBuffer?.Release(); _keyArray = null;
        _payloadTempBuffer?.Release(); _payloadArray = null;
        _sortCountBuffer?.Release(); _sortCountBuffer = null;
        _commandBuffer?.Dispose(); _commandBuffer = null;
        _sorter?.Dispose(); _sorter = null;
    }

    public void Init()
    {
        _successfllyInitialized = false;

        _currentSortingAlgorithm = _sortingAlgorithm;
        _currentSortMode = _sortMode;
        _currentDispatchMode = _dispatchMode;
        _currentNumData = _numData;

        ReleaseObjects();

        _sorter = _currentSortingAlgorithm switch
        {
            SortingAlgorithm.Onesweep => new OnesweepSorter(),
            SortingAlgorithm.Traditional => new TraditionalSorter(),
            _ => throw new ArgumentOutOfRangeException(nameof(_currentSortingAlgorithm), _currentSortingAlgorithm, null)
        };

        try
        {
            _sorter.Init(_config, _currentNumData, _currentSortMode, KeyType.UInt, SortingOrder.Ascending, _currentDispatchMode, WaveSize.Unknown);
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to initialize sorter: {e.Message}");
            return;
        }

        _keyBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _currentNumData, sizeof(type));
        _payloadBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _currentNumData, sizeof(uint));
        _keyTempBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _currentNumData, sizeof(type));
        _payloadTempBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _currentNumData, sizeof(uint));
        _sortCountBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 2, sizeof(uint));
        _sortCountBuffer.SetData(new[] { 0u, (uint)_currentNumData });
        _commandBuffer = new CommandBuffer { name = "SortSampleCommandBuffer" };

        _keyArray = new type[_currentNumData];
        _payloadArray = new uint[_currentNumData];
        _combinedDataArray = new KeyPayloadCombine[_currentNumData];

        Random.InitState(_randomSeed);
        for (uint i = 0; i < _currentNumData; i++)
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

        _successfllyInitialized = true;
    }

    private void DispatchCopyKernel()
    {
        if (_useCommandBuffer)
        {
            _commandBuffer.SetComputeIntParam(_copyCs, NumElementsID, _currentNumData);
            _commandBuffer.SetComputeBufferParam(_copyCs, _copyKernel, KeyBufferID, _keyBuffer);
            _commandBuffer.SetComputeBufferParam(_copyCs, _copyKernel, PayloadBufferID, _payloadBuffer);
            _commandBuffer.SetComputeBufferParam(_copyCs, _copyKernel, KeyTempBufferID, _keyTempBuffer);
            _commandBuffer.SetComputeBufferParam(_copyCs, _copyKernel, PayloadTempBufferID, _payloadTempBuffer);
            _commandBuffer.DispatchCompute(_copyCs, _copyKernel, CopyKernelDispatchGroupSize, 1, 1);
        }
        else
        {
            _copyCs.SetInt(NumElementsID, _currentNumData);
            _copyCs.SetBuffer(_copyKernel, KeyBufferID, _keyBuffer);
            _copyCs.SetBuffer(_copyKernel, PayloadBufferID, _payloadBuffer);
            _copyCs.SetBuffer(_copyKernel, KeyTempBufferID, _keyTempBuffer);
            _copyCs.SetBuffer(_copyKernel, PayloadTempBufferID, _payloadTempBuffer);
            _copyCs.Dispatch(_copyKernel, CopyKernelDispatchGroupSize, 1, 1);
        }
    }

    private void DispatchSortKernel()
    {
        if (_useCommandBuffer)
        {
            switch (_currentDispatchMode)
            {
                case DispatchMode.Direct:
                    _sorter.Sort(_commandBuffer, _keyBuffer, _currentSortMode == SortMode.KeyPayload ? _payloadBuffer : null, _currentNumData);
                    break;
                case DispatchMode.Indirect:
                    _sorter.Sort(_commandBuffer, _keyBuffer, _currentSortMode == SortMode.KeyPayload ? _payloadBuffer : null, _sortCountBuffer, 1);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(_currentDispatchMode), _currentDispatchMode, null);
            }
        }
        else
        {
            switch (_currentDispatchMode)
            {
                case DispatchMode.Direct:
                    _sorter.Sort(_keyBuffer, _currentSortMode == SortMode.KeyPayload ? _payloadBuffer : null, _currentNumData);
                    break;
                case DispatchMode.Indirect:
                    _sorter.Sort(_keyBuffer, _currentSortMode == SortMode.KeyPayload ? _payloadBuffer : null, _sortCountBuffer, 1u);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(_currentDispatchMode), _currentDispatchMode, null);
            }
        }
    }

    public void CheckSuccess()
    {
        Init();

        if (!_successfllyInitialized)
        {
            Debug.LogError("Sorter initialization failed. Cannot check success.");
            return;
        }

        if (_useCommandBuffer) _commandBuffer.Clear();
        DispatchCopyKernel();
        DispatchSortKernel();
        if (_useCommandBuffer) Graphics.ExecuteCommandBuffer(_commandBuffer);

        type[] keyArray = new type[_currentNumData];
        _keyBuffer.GetData(keyArray);

        switch (_currentSortMode)
        {
            case SortMode.KeyOnly:
                _keyArray = _keyArray.OrderBy(key => key).ToArray();

                if (_keyArray.SequenceEqual(keyArray))
                    Debug.Log("Sorting Success");
                else
                    Debug.LogError("Sorting Failure");
                break;
            case SortMode.KeyPayload:
                uint[] payloadArray = new uint[_currentNumData];
                _payloadBuffer.GetData(payloadArray);

                KeyPayloadCombine[] combinedDataArray = keyArray.Select((key, i) => new KeyPayloadCombine(key, payloadArray[i])).ToArray();
                _combinedDataArray = _combinedDataArray.OrderBy(data => data.Key).ToArray();

                if (_combinedDataArray.SequenceEqual(combinedDataArray))
                    Debug.Log("Sorting Success");
                else
                    Debug.LogError("Sorting Failure");
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(_currentSortMode), _currentSortMode, null);
        }

        ReleaseObjects();
    }

    private Element CreateElement()
    {
        return UI.Window(
            UI.Label("<b>Sort Sample</b>").SetWidth(200f),
            UI.Box().SetHeight(5f).SetBackgroundColor(Color.gray),
            UI.Field("Data Count", () => _numData),
            UI.FieldReadOnly("Current Data Count", () => _currentNumData),
            UI.Space().SetHeight(10f),
            UI.Field("Max Random Value", () => _randomValueMax),
            UI.Field("Random Seed", () => _randomSeed),
            UI.Space().SetHeight(10f),
            UI.Field("Sorting Algorithm", () => _sortingAlgorithm),
            UI.FieldReadOnly("Current Sorting Algorithm", () => _currentSortingAlgorithm),
            UI.Space().SetHeight(10f),
            UI.Field("Sort Mode", () => _sortMode),
            UI.FieldReadOnly("Current Sort Mode", () => _currentSortMode),
            UI.Space().SetHeight(10f),
            UI.Field("Dispatch Mode", () => _dispatchMode),
            UI.FieldReadOnly("Current Dispatch Mode", () => _currentDispatchMode),
            UI.Space().SetHeight(10f),
            UI.Field("Use Command Buffer", () => _useCommandBuffer),
            UI.Space().SetHeight(10f),
            UI.Field("Dispatch Only Copy Kernel", () => _dispatchOnlyCopyKernel),
            UI.FieldReadOnly("Running Kernels", () => RunningKernels),
            UI.Space().SetHeight(10f),
            UI.Button("Reinit", Init)
        ).SetClosable(false);
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

        if (Application.isPlaying)
        {
            if (GUILayout.Button("Reinit"))
            {
                var sortSample = target as SortSample;
                sortSample.Init();
            }
        }

        if (!Application.isPlaying)
        {
            if (GUILayout.Button("Check Success"))
            {
                var sortSample = target as SortSample;
                sortSample.CheckSuccess();
            }
        }
    }
}
#endif