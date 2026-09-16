# GPU Radix Sorter for Unity

DirectX 12 compute-shader implementations of Least Significant Digit (LSD) radix sort for Unity.

- `TraditionalSorter`: A conventional multi-pass implementation and the recommended default.
- `OnesweepSorter`: Based on [Onesweep: A Faster Least Significant Digit Radix Sort for GPUs](https://arxiv.org/abs/2206.01784).

Both sorters include code adapted from the [GPUSorting project by Thomas Smith](https://github.com/b0nes164/GPUSorting).

## Choosing an algorithm

### TraditionalSorter

`TraditionalSorter` does not use inter-group lookback, so it avoids this specific hang risk. Prefer it when reliability and predictable performance matter more than peak speed.

### OnesweepSorter

`OnesweepSorter` targets higher throughput, but its lookback step relies on forward progress between GPU thread groups. On hardware or drivers that do not provide sufficient progress, the loop may spin indefinitely and cause a GPU hang. Performance can also vary by GPU and workload.

## Features

- GPU-accelerated Traditional and Onesweep radix sort implementations.
- `uint`, `int`, and `float` keys.
- Ascending and descending sort orders.
- Key-only and key-payload sorting with `SortMode`.
- Direct and indirect dispatch modes.
- `GraphicsBuffer` input and output.
- `CommandBuffer` support for rendering-pipeline integration.

## Requirements

- Unity 2022.3+
- DirectX 12 as the active graphics API (Windows only)
- Compute Shader support
- A GPU wave size of 8, 16, 32, or 64

Wave8 support is a compatibility path and favors a straightforward implementation over peak performance. Wave16, Wave32, and Wave64 use the regular per-wave path.

## Installation

1. Open the Unity Package Manager.
2. Click the **+** button.
3. Select **Add package from git URL...**.
4. Enter `https://github.com/abecombe/Unity-Onesweep.git?path=Packages/com.abecombe.onesweep`.

## Usage

See `Assets/Samples` for a complete example. The sample project also provides **Tools > Onesweep > Check Wave Size** in the Unity Editor.

> For representative performance measurements, use a standalone build rather than the Unity Editor.

```csharp
using Onesweep; // Namespace for the sorters
using UnityEngine;

public class MySorterBehaviour : MonoBehaviour
{
    ISorter sorter;

    void Start()
    {
        // Choose the sorter implementation:
        sorter = new TraditionalSorter(); // Recommended for stability and predictable performance
        // sorter = new OnesweepSorter(); // Use when peak performance is more important

        // Initialize the sorter, specifying the SortMode
        sorter.Init(
            maxSortCount: 65536,
            sortMode: SortMode.KeyPayload,        // Choose SortMode.KeyOnly or SortMode.KeyPayload
            keyType: KeyType.UInt,                // Choose KeyType.UInt, KeyType.Int, or KeyType.Float
            sortingOrder: SortingOrder.Ascending, // Choose SortingOrder.Ascending or SortingOrder.Descending
            dispatchMode: DispatchMode.Direct     // Choose DispatchMode.Direct or DispatchMode.Indirect
        );
    }

    void Update()
    {
        // For direct dispatch:
        sorter.Sort(keyBuffer, payloadBuffer, sortCount);

        // For indirect dispatch (example):
        // GraphicsBuffer sortCountBuffer = ...; // Contains sortCount at sortCountBufferOffset
        // uint sortCountBufferOffset = 0;
        // sorter.Sort(keyBuffer, payloadBuffer, sortCountBuffer, sortCountBufferOffset);
    }

    void OnDestroy()
    {
        sorter?.Dispose();
    }
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

This implementation includes code adapted from the [GPUSorting project by Thomas Smith](https://github.com/b0nes164/GPUSorting), also licensed under the MIT License.

## References

- [Onesweep (arXiv)](https://arxiv.org/abs/2206.01784)
- [GPU Multisplit](https://madalgo.au.dk/fileadmin/madalgo/OA_PDF_s/C417.pdf)
- [Fast 4-way parallel radix sorting on GPUs](http://www.sci.utah.edu/publications/Ha2009b/Ha_CGF2009.pdf)
- [GPUSorting by Thomas Smith](https://github.com/b0nes164/GPUSorting)