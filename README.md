# VRAM Planner

A Next.js web app for planning GPU VRAM allocation before loading model weights.

## What it does

Pick a target GPU from a catalog of 40+ NVIDIA, AMD, and Apple Silicon GPUs, then add LLM/diffusion model presets to simulate VRAM usage. The planner stacks allocations visually and reports capacity, utilization, and per-model fit analysis. A 5% overhead reserve is applied automatically.

## Model catalog

Includes presets for 60+ models across families: Llama, Qwen, Mistral/Mixtral, Phi, Gemma, Code Llama, DeepSeek, Falcon, Stable Diffusion, FLUX, and more. Each preset records parameter count, quantization type, VRAM requirement, context length, and capability tags.

## Getting started

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

## Static export

The app is configured for static export (`output: "export"` in `next.config.ts`). Run `npm run build` to produce a static site in the `out/` directory that can be served by any static host.

## Stack

- [Next.js 16](https://nextjs.org) (App Router)
- React 19
- Tailwind CSS 4
- TypeScript
