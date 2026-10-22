# CUDA Course

Repository for CUDA programming coursework, exercises, and reusable examples developed during university studies.

## Current contents

At the moment, the repository contains a ready-to-use CUDA C/C++ template:

- [`template.cu`](./template.cu) — a complete example demonstrating the basic CUDA workflow:
  - CUDA error handling with the `CUDA_CHECK` macro,
  - defining and launching a GPU kernel,
  - calculating a thread's global index using `blockIdx`, `blockDim`, and `threadIdx`,
  - allocating host and device memory,
  - copying data between CPU and GPU,
  - synchronizing the device and checking kernel errors,
  - copying results back to the CPU,
  - releasing allocated resources.

The example initializes an array of integers on the CPU and then doubles every element on the GPU.

## Requirements

- An NVIDIA GPU with CUDA support
- NVIDIA drivers compatible with the installed CUDA Toolkit
- NVIDIA CUDA Toolkit, including `nvcc`
- A supported C/C++ development environment

The project can be built on Linux, Windows, or WSL, provided that the CUDA Toolkit and drivers are configured correctly.

## Building and running the template

From the repository root, compile the CUDA source file with `nvcc`:

```bash
nvcc template.cu -o template
./template
```

On Windows, run the generated executable as appropriate for your environment:

```powershell
nvcc template.cu -o template.exe
.\template.exe
```

The program should print the values after GPU processing:

```text
Wyniki po obliczeniach na GPU:
0 2 4 6 8 10 12 14 16 18
```

## CUDA concepts demonstrated

The template is intentionally focused on the fundamentals needed for first CUDA programs:

- **Host and device code** — `main` runs on the CPU, while `myKernel` runs on the GPU.
- **Kernels** — functions marked with `__global__` are launched using CUDA's `<<<...>>>` syntax.
- **Thread indexing** — each thread computes a unique position in the processed array.
- **Grid and block configuration** — the number of blocks is calculated from the data size and the selected block size.
- **Explicit memory management** — data is allocated with `malloc` on the host and `cudaMalloc` on the device.
- **Memory transfers** — `cudaMemcpy` is used to transfer data in both directions.
- **Error checking and synchronization** — CUDA API calls and kernel execution are checked explicitly.

## Repository development

As the course progresses, additional projects, exercises, and examples may be added. Each larger assignment should include its own build instructions and documentation where necessary.

## Notes

This repository is intended primarily for university coursework, experimentation, and learning. The code may be extended or reorganized as new CUDA topics are introduced.

## License

Unless stated otherwise in a specific directory, the contents of this repository are provided for educational and personal use.
