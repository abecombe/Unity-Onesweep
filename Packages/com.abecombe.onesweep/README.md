# Onesweep & Traditional - GPU Radix Sorter for Unity (DX12)

This package provides two fast GPU-based Least Significant Digit (LSD) Radix Sort implementations for Unity using Compute Shaders and DirectX 12:

1.  **Onesweep**: Based on the paper [Onesweep: A Faster Least Significant Digit Radix Sort for GPUs](https://arxiv.org/abs/2206.01784).
2.  **Traditional**: A more conventional multi-pass GPU radix sort.

Both sorters include code adapted from the [GPUSorting project by Thomas Smith](https://github.com/b0nes164/GPUSorting).

-----

## ⚠️ Important Notice: Algorithm Stability and Performance ⚠️

### Onesweep Sorter

**Potential for Deadlocks:**  
Due to the nature of the Onesweep algorithm and its reliance on complex GPU synchronization (potentially involving device-wide atomics or specific inter-group communication patterns), there is a possibility of encountering GPU deadlocks. This can lead to application freezes (hangs) under certain runtime conditions. This issue may occur sporadically depending on various factors, including the specific GPU model, driver version, operating system, or conflicts with other running tasks. It is often related to the intricate scheduling and synchronization of GPU tasks.

**Performance Instability:**  
The performance of the **Onesweep** sorter can also be unstable. While it aims for high speed by minimizing passes, its actual performance may vary significantly based on the GPU architecture, driver, and the specific dataset being sorted.

**If you experience frequent or unexplained application freezes or highly variable performance while using `OnesweepSorter`, please consider using the `TraditionalSorter` to see if the issue persists or if performance is more consistent.**

We apologize for any inconvenience these potential risks may cause and appreciate your understanding.

### TraditionalSorter

The **`TraditionalSorter`** implements a more conventional multi-pass GPU radix sort. This approach generally offers:

* **Greater Stability**: Deadlocks or hangs related to complex, cutting-edge synchronization techniques do not occur with this sorter.
* **More Predictable Performance**: While potentially not reaching the peak speeds of a perfectly functioning Onesweep on specific hardware, its performance is typically more consistent across different GPUs and scenarios.

It is recommended as a robust alternative if you encounter issues with the Onesweep implementation or require more predictable behavior.

-----

## ✨ Features

* GPU-accelerated LSD radix sort (both Onesweep and Traditional algorithms).
* Supports `uint`, `int`, and `float` keys.
* Ascending and descending sort orders.
* Configurable `SortMode` (`KeyOnly` / `KeyPayload`): Supports optional payloads (must be in a separate 4-byte stride `GraphicsBuffer`).
* Direct and indirect dispatch modes.
* Works with Unity's `GraphicsBuffer` for input/output data.
* Can be dispatched via `CommandBuffer` for integration into rendering pipelines.
* Wave size customization (32 or 64).

## 🚀 Requirements

* Unity 2022.3+
* DirectX 12 as active graphics API (Windows only)
* Compute Shader support
* GPU with a supported wave size: **32** (NVIDIA) or **64** (AMD)

## 📦 Installation

1.  Open the Unity Package Manager
2.  Click the **+** button
3.  Select "**Add package from git URL...**"
4.  Enter `https://github.com/abecombe/Unity-Onesweep.git?path=Packages/com.abecombe.onesweep`

## 🛠 Usage

For more detailed usage examples, please refer to the sample scene included in this package.  
**If you wish to accurately assess performance, it is recommended to create and run a standalone build, as performance figures in the Unity Editor may not be representative of final build performance.**
```csharp
using Onesweep; // Namespace for the sorters
using UnityEngine;

public class MySorterBehaviour : MonoBehaviour
{
    // Assign this from the Inspector.
    // The asset is included in the package at:
    // "Packages/Onesweep/Runtime/OnesweepComputeConfig.asset"
    // This config is used by both sorter types.
    [SerializeField] private OnesweepComputeConfig config;

    ISorter sorter;

    void Start()
    {
        // Choose the sorter implementation:
        sorter = new OnesweepSorter(); // Use OnesweepSorter for high performance, but be aware of potential deadlocks and performance instability
        // or
        sorter = new TraditionalSorter(); // Recommended for stability and predictable performance

        // Initialize the sorter, specifying the SortMode
        sorter.Init(
            config,
            maxSortCount: 65536,
            sortMode: SortMode.KeyPayload,        // Choose SortMode.KeyOnly or SortMode.KeyPayload
            keyType: KeyType.UInt,                // Choose KeyType.UInt, KeyType.Int, or KeyType.Float
            sortingOrder: SortingOrder.Ascending, // Choose SortingOrder.Ascending or SortingOrder.Descending
            dispatchMode: DispatchMode.Direct   , // Choose DispatchMode.Direct or DispatchMode.Indirect
            waveSize: WaveSize.Unknown            // Attempts to auto-detect, or set explicitly e.g., WaveSize.WaveSize32 or WaveSize.WaveSize64
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

## ⚙️ Configuration Notes

**Wave Size Consistency:**
The `WaveSize` (e.g., WaveSize32 or WaveSize64) for sorter operations is established when `sorter.Init()` is called.
* If an explicit `WaveSize` (e.g., `WaveSize.WaveSize32` or `WaveSize.WaveSize64`) is provided to `Init()`, that specific sorter instance will be configured to use the specified size.
* If `WaveSize.Unknown` is specified:
    * Upon the **very first `Init()` call with `WaveSize.Unknown` across all sorter instances** within the application's session, the system attempts to auto-detect a compatible wave size based on the current GPU.
    * This initially auto-detected wave size is then **cached globally**.
    * All **subsequent sorter instances that are also initialized with `WaveSize.Unknown` will reuse this cached, globally determined wave size**.
      Once configured (either explicitly or via the shared auto-detection mechanism), an instance uses its determined wave size for all subsequent operations, typically by selecting appropriate shader variants.

**Due to this caching mechanism for auto-detected wave sizes, the sorting system inherently assumes that the active GPU's wave size capability remains stable throughout the application session after the first auto-detection occurs (when `WaveSize.Unknown` is first used). The system is not designed to handle dynamic changes to the GPU's effective wave size mid-session without a re-evaluation of this cached value (e.g., via an application restart or a specific reset mechanism not currently exposed).**

It is crucial that the GPU consistently supports the wave size established for any sorter instance. Attempting to run a sorter with a wave size configuration that is incompatible with the current runtime GPU environment can lead to undefined behavior, incorrect sorting results, errors, and in the worst case, can cause processing to fail entirely. Such incompatibilities can arise if:
* An explicit, unsupported `WaveSize` is forced (e.g., WaveSize64 on a GPU that only effectively supports WaveSize32 for the deployed shaders).
* The active rendering device changes capabilities during the application's runtime (e.g., **switching from a dedicated GPU (dGPU) to an integrated GPU (iGPU) on a laptop, or vice-versa**) *after* an initial wave size has been auto-detected and cached, leading to a mismatch if new instances using `WaveSize.Unknown` are initialized or existing instances continue to operate based on the cached value.

Always ensure that the `WaveSize` configuration is appropriate for the target hardware and remains compatible with the active rendering device throughout the sorter's usage. If the GPU or its capabilities are expected to change significantly, the application may need to manage the sorter's lifecycle and re-initialization more carefully with explicit `WaveSize` values appropriate for the new environment.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

This implementation includes code adapted from the [GPUSorting project by Thomas Smith](https://github.com/b0nes164/GPUSorting), also licensed under the MIT License.

## 💬 Acknowledgements

* [Onesweep (arXiv)](https://arxiv.org/abs/2206.01784)
* [GPU Multisplit](https://madalgo.au.dk/fileadmin/madalgo/OA_PDF_s/C417.pdf)
* [Fast 4-way parallel radix sorting on GPUs](http://www.sci.utah.edu/publications/Ha2009b/Ha_CGF2009.pdf)
* [GPUSorting by Thomas Smith](https://github.com/b0nes164/GPUSorting)