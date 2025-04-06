# Onesweep - GPU Radix Sort for Unity (DX12)

**Onesweep** is a fast GPU-based Least Significant Digit (LSD) Radix Sort implementation for Unity using Compute Shaders and DirectX 12.

This sorting algorithm is based on the paper  
[Onesweep: A Faster Least Significant Digit Radix Sort for GPUs](https://arxiv.org/abs/2206.01784)  
and includes code adapted from the [GPUSorting project by Thomas Smith](https://github.com/b0nes164/GPUSorting).

---

## ⚠️ Important Notice: Potential for Deadlocks ⚠️

**Due to the nature of the Onesweep algorithm and its reliance on complex GPU synchronization, there is a possibility of encountering GPU deadlocks, which can lead to application freezes (hangs) under certain runtime conditions.**

This issue may occur sporadically depending on various factors, including the specific GPU model, driver version, operating system, or conflicts with other running tasks. It is often related to the intricate scheduling and synchronization of GPU tasks.

**If you experience frequent or unexplained application freezes while using Onesweep, please consider using an alternative sorting algorithm to see if the issue persists.**

We apologize for any inconvenience this potential risk may cause and appreciate your understanding.

---

## ✨ Features

- GPU-accelerated LSD radix sort
- Supports `uint`, `int`, and `float` keys
- Ascending and descending order
- Direct and indirect dispatch modes
- Works with Unity's `GraphicsBuffer` for input/output data
- Can be dispatched via `CommandBuffer` for integration into rendering pipelines
- Wave size customization (32 / 64)

## 🚀 Requirements

- Unity 2022.3+
- DirectX 12 as active graphics API (Windows only)
- Compute Shader support
- GPU with a supported wave size: **32** (NVIDIA) or **64** (AMD)

## 📦 Installation

1. Open the Unity Package Manager
2. Click the + button
3. Select "Add package from git URL..."
4. Enter https://github.com/abecombe/Unity-Onesweep.git?path=Packages/com.abecombe.onesweep

## 🛠 Usage

```csharp
using Onesweep;
using UnityEngine;

public class MySorterBehaviour : MonoBehaviour
{
    // Assign this from the Inspector.
    // The asset is included in the package at:
    // "Packages/Onesweep/Runtime/OnesweepComputeConfig.asset"
    [SerializeField] private OnesweepComputeConfig config;

    private RadixSort sorter = new();

    void Start()
    {
        sorter.Init(
            config,
            maxSortCount: 65536,
            keyType: KeyType.UInt,
            sortingOrder: SortingOrder.Ascending,
            dispatchMode: DispatchMode.Direct,
            waveSize: WaveSize.Unknown
        );
    }
    
    void Update()
    {
        // For direct dispatch:
        sorter.Sort(keyBuffer, indexBuffer, sortCount);
        
        // For indirect dispatch:
        sorter.Sort(keyBuffer, indexBuffer, sortCountBuffer, sortCountBufferOffset);
    }

    void OnDestroy()
    {
        sorter?.Dispose();
    }
}
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

This implementation includes code adapted from the [GPUSorting project by Thomas Smith](https://github.com/b0nes164/GPUSorting), also licensed under the MIT License.

## 💬 Acknowledgements

- [Onesweep (arXiv)](https://arxiv.org/abs/2206.01784)
- [GPU Multisplit](https://madalgo.au.dk/fileadmin/madalgo/OA_PDF_s/C417.pdf)
- [GPUSorting by Thomas Smith](https://github.com/b0nes164/GPUSorting)