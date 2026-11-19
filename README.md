# CUDA Course

Repository for CUDA programming coursework, exercises, and reusable examples developed during university studies.

## Projects

### [`template.cu`](./template.cu) — CUDA fundamentals

A complete introductory CUDA C++ example that doubles an integer array on the GPU. It demonstrates:

- CUDA kernels and the `<<<grid, block>>>` launch syntax,
- host and device memory management with `malloc`, `cudaMalloc`, and `cudaFree`,
- CPU/GPU data transfers with `cudaMemcpy`,
- thread indexing with `blockIdx`, `blockDim`, and `threadIdx`,
- CUDA error handling with `CUDA_CHECK`,
- kernel synchronization and result verification.

### [`Lab_2`](./Lab_2) — GPU fractal computation

A CUDA C++ implementation of a fractal renderer that distributes pixel calculations across two-dimensional CUDA thread blocks. The directory also includes [`fractal.c`](./Lab_2/fractal.c), a sequential C reference implementation.

Technologies and concepts:

- CUDA C++ and the CUDA Runtime API,
- GPU kernels and two-dimensional grids/blocks,
- host/device memory allocation and transfers,
- CMake with C++20/CUDA language support,
- comparison of sequential CPU and parallel GPU computation.

### [`Lab_3`](./Lab_3) — NVIDIA Thrust algorithms

Exercises using NVIDIA Thrust to process rainfall time-series data stored in CUDA device vectors. The examples cover data aggregation by day and site and introduce parallel algorithms such as `inner_product`, `sort_by_key`, `reduce_by_key`, `count`, and binary searches.

Technologies and concepts:

- CUDA C++,
- NVIDIA Thrust and `thrust::device_vector`,
- parallel reductions, sorting, counting, and searching,
- array of structures (AoS) and structure of arrays (SoA) data models,
- CMake with C++20/CUDA language support.

### [`k_means_project`](./k_means_project) — Parallel k-means specification

A project specification and implementation plan for a CUDA-based k-means clustering algorithm, including a sequential CPU reference and two GPU variants:

- `cpu` — sequential C/C++ implementation,
- `gpu1` — custom CUDA kernels with shared-memory reductions,
- `gpu2` — NVIDIA Thrust-based parallel algorithms.

The documentation covers large multidimensional datasets, AoS-to-SoA data transformation, coalesced memory access, constant memory for centroids, CUDA error handling, text and binary input formats, and performance measurement. See the [project README](./k_means_project/README.md) and [technical specification](./k_means_project/projekt.pdf) for details.

## Technologies

- **C and C++** — sequential reference programs and host-side application code
- **NVIDIA CUDA** — GPU kernels, device memory, synchronization, and the CUDA Runtime API
- **NVIDIA Thrust** — high-level parallel algorithms and device containers
- **`nvcc`** — CUDA compiler used to build `.cu` sources
- **CMake** — build configuration for the laboratory projects
- **CUDA C++20** — language standard configured by the lab CMake projects

## Requirements

- An NVIDIA GPU with CUDA support
- NVIDIA drivers compatible with the installed CUDA Toolkit
- NVIDIA CUDA Toolkit, including `nvcc`
- CMake 4.3 or newer for `Lab_2` and `Lab_3`
- A supported C/C++ development environment

The projects can be built on Linux, Windows, or WSL when the CUDA Toolkit and drivers are configured correctly.

## Building the laboratory projects

Each laboratory directory contains its own `CMakeLists.txt`. From the selected project directory, configure and build with:

```bash
cmake -S . -B build
cmake --build build
```

The introductory template can be compiled directly from the repository root:

```bash
nvcc template.cu -o template
./template
```

On Windows:

```powershell
nvcc template.cu -o template.exe
.\template.exe
```

The template should print:

```text
Wyniki po obliczeniach na GPU:
0 2 4 6 8 10 12 14 16 18
```

## Notes

This repository is intended primarily for university coursework, experimentation, and learning. Some directories contain exercises or project specifications that are still in development.

## License

Unless stated otherwise in a specific directory, the contents of this repository are provided for educational and personal use.
