# CUDA Course

This repository contains my projects, exercises, and templates for the CUDA course at university.

## Repository Contents

- **Projects** — Coursework and practical CUDA projects.
- **Templates** — Starter templates and reusable boilerplate for assignments.
- **Exercises** — Small examples and experiments for learning CUDA concepts.

## Topics

The repository may include work related to:

- CUDA C/C++ programming
- GPU programming fundamentals
- Kernels and thread hierarchies
- Blocks, grids, and memory management
- Global, shared, constant, and unified memory
- Synchronization and concurrency
- Performance optimization
- Parallel algorithms

## Getting Started

### Prerequisites

- NVIDIA GPU with CUDA support
- NVIDIA CUDA Toolkit
- A C/C++ compiler supported by the installed CUDA Toolkit
- Linux, Windows, or WSL with the required NVIDIA drivers

### Building a CUDA Program

From a project directory containing a `.cu` file, compile it with `nvcc`:

```bash
nvcc main.cu -o main
./main
```

The exact build commands may differ between projects. Refer to each project's documentation and build configuration for details.

## Project Structure

The repository is organized as coursework develops. Individual projects and templates may contain their own source files, build instructions, and notes.

```text
Cuda/
├── projects/     # Course projects
├── templates/    # Starter templates and boilerplate
├── exercises/    # Practice exercises and examples
└── README.md
```

> Directory names may change as the course progresses.

## Notes

This repository is primarily intended for personal university coursework, experimentation, and reference. Code may be updated, refactored, or reorganized throughout the course.

## License

Unless stated otherwise in a specific directory, the contents of this repository are provided for educational and personal use.
