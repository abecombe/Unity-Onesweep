using System;
using System.Linq;
using Onesweep;
using UnityEngine;

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
    [SerializeField] private OnesweepComputeConfig _config;

    private Sorter _sorter = new();

    private GraphicsBuffer _keyBuffer;
    private GraphicsBuffer _indexBuffer;
    private GraphicsBuffer _keyTempBuffer;
    private GraphicsBuffer _indexTempBuffer;

    private ComputeShader _copyCs;
    private int _copyKernel;

    private KeyIndexCombine[] _combinedDataArray;

    private struct KeyIndexCombine : IEquatable<KeyIndexCombine>
    {
        public type Key;
        public uint Index;

        public KeyIndexCombine(type key, uint index)
        {
            this.Key = key;
            this.Index = index;
        }

        public bool Equals(KeyIndexCombine other)
        {
            return Key == other.Key && Index == other.Index;
        }
    }

    private void Start()
    {
        _sorter.Init(_config, _numData, KeyType.UInt, SortingOrder.Ascending, DispatchMode.Direct, WaveSize.Unknown);

        _keyBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _numData, sizeof(type));
        _indexBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _numData, sizeof(uint));
        _keyTempBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _numData, sizeof(type));
        _indexTempBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _numData, sizeof(uint));

        type[] keyArr = new type[_numData];
        uint[] indexArr = new uint[_numData];
        _combinedDataArray = new KeyIndexCombine[_numData];

        Random.InitState(_randomSeed);
        for (uint i = 0; i < _numData; i++)
        {
            type value = (uint)Random.Range(0, _randomValueMax + 1);
            keyArr[i] = value;
            indexArr[i] = i;
            _combinedDataArray[i] = new KeyIndexCombine(value, i);
        }
        _keyTempBuffer.SetData(keyArr);
        _indexTempBuffer.SetData(indexArr);

        _copyCs = Resources.Load<ComputeShader>("Copy");
        _copyKernel = _copyCs.FindKernel("CopySortBuffer");

        _copyCs.SetBuffer(_copyKernel, "key_buffer", _keyBuffer);
        _copyCs.SetBuffer(_copyKernel, "index_buffer", _indexBuffer);
        _copyCs.SetBuffer(_copyKernel, "key_temp_buffer", _keyTempBuffer);
        _copyCs.SetBuffer(_copyKernel, "index_temp_buffer", _indexTempBuffer);
        _copyCs.SetInt("num_elements", _numData);
    }

    private void Update()
    {
        _copyCs.Dispatch(_copyKernel, 128, 1, 1);

        _sorter.Sort(_keyBuffer, _indexBuffer, _numData);
    }

    private void OnDestroy()
    {
        _keyBuffer?.Release();
        _indexBuffer?.Release();
        _keyTempBuffer?.Release();
        _indexTempBuffer?.Release();
        _sorter.Dispose();
    }

    public void CheckSuccess()
    {
        Start();

        Update();

        type[] keyArr = new type[_numData];
        _keyBuffer.GetData(keyArr);
        uint[] indexArr = new uint[_numData];
        _indexBuffer.GetData(indexArr);
        KeyIndexCombine[] combinedDataArray = keyArr.Select((key, i) => new KeyIndexCombine(key, indexArr[i])).ToArray();

        _combinedDataArray = _combinedDataArray.OrderBy(data => data.Key).ToArray();

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