# GPU Partitioner

A Next.js web app for planning GPU VRAM allocation before loading model weights, with an optional live view of the host's NVIDIA GPU memory state.

## What it does

- **Plan mode** — pick a target GPU from a catalog of 40+ NVIDIA, AMD, and Apple Silicon GPUs, then add LLM/diffusion model presets to simulate VRAM usage. The planner stacks allocations visually and reports capacity, utilization, and per-model fit analysis. A 5% overhead reserve is applied automatically.
- **Live mode** — when running in a container with NVIDIA runtime support, the app queries `nvidia-smi` every 5 seconds and displays real memory usage, GPU utilization, and temperature for each device on the host.

## Model catalog

Includes presets for 60+ models across families: Llama, Qwen, Mistral/Mixtral, Phi, Gemma, Code Llama, DeepSeek, Falcon, Stable Diffusion, FLUX, and more. Each preset records parameter count, quantization type, VRAM requirement, context length, and capability tags.

## Getting started

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

## Docker

Build and run without GPU access (plan mode only):

```bash
docker build -t gpu-partitioner .
docker run --rm -p 3000:3000 gpu-partitioner
```

Enable the live view by passing the NVIDIA runtime and all GPUs:

```bash
docker run --rm --gpus all -p 3000:3000 gpu-partitioner
```

If `nvidia-smi` is unavailable the app still works — live mode shows an error message and plan mode is unaffected.

## Stack

- [Next.js 16](https://nextjs.org) (App Router)
- React 19
- Tailwind CSS 4
- TypeScript
