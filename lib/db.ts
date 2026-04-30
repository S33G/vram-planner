export interface GpuSpec {
  id: string
  name: string
  vendor: 'nvidia' | 'amd' | 'apple'
  vram: number
  memoryBandwidth: number
  pcie: string
  releaseYear: number
  tdp: number
  recommended: boolean
}

export interface ModelPreset {
  id: string
  name: string
  family: string
  parameters: number
  quantization: QuantizationType
  vramRequired: number
  contextLength: number
  recommendedGpuVram: number
  minGpuVram: number
  tags: string[]
}

export type QuantizationType = 'fp16' | 'bf16' | 'fp8' | 'int8' | 'int4' | 'nf4' | 'mixed'
export type PerformanceBucket = 'realtime' | 'fast' | 'moderate' | 'slow' | 'infeasible'

export interface Partition {
  id: string
  name: string
  modelId: string
  slot: number
  vramAllocated: number
  performance: PerformanceBucket
}

export interface GpuState {
  gpu: GpuSpec
  partitions: Partition[]
  totalUsedVram: number
  totalFreeVram: number
}

export const GPUs: GpuSpec[] = [
  { id: 'h100-80', name: 'NVIDIA H100 80GB', vendor: 'nvidia', vram: 80, memoryBandwidth: 3350, pcie: '4.0', releaseYear: 2022, tdp: 700, recommended: true },
  { id: 'h100-40', name: 'NVIDIA H100 40GB SXM', vendor: 'nvidia', vram: 40, memoryBandwidth: 3350, pcie: '4.0', releaseYear: 2022, tdp: 350, recommended: true },
  { id: 'a100-80', name: 'NVIDIA A100 80GB', vendor: 'nvidia', vram: 80, memoryBandwidth: 2000, pcie: '4.0', releaseYear: 2020, tdp: 400, recommended: true },
  { id: 'a100-40', name: 'NVIDIA A100 40GB', vendor: 'nvidia', vram: 40, memoryBandwidth: 2000, pcie: '4.0', releaseYear: 2020, tdp: 300, recommended: false },
  { id: 'a6000-48', name: 'NVIDIA RTX Ada 6000 48GB', vendor: 'nvidia', vram: 48, memoryBandwidth: 900, pcie: '4.0', releaseYear: 2022, tdp: 300, recommended: false },
  { id: 'rtx4090-24', name: 'NVIDIA RTX 4090 24GB', vendor: 'nvidia', vram: 24, memoryBandwidth: 1008, pcie: '4.0', releaseYear: 2022, tdp: 450, recommended: true },
  { id: 'rtx4080s-16', name: 'NVIDIA RTX 4080 Super 16GB', vendor: 'nvidia', vram: 16, memoryBandwidth: 770, pcie: '4.0', releaseYear: 2024, tdp: 320, recommended: false },
  { id: 'rtx4080-12', name: 'NVIDIA RTX 4080 12GB', vendor: 'nvidia', vram: 12, memoryBandwidth: 716, pcie: '4.0', releaseYear: 2023, tdp: 320, recommended: false },
  { id: 'rtx4070s-12', name: 'NVIDIA RTX 4070 Super 12GB', vendor: 'nvidia', vram: 12, memoryBandwidth: 504, pcie: '4.0', releaseYear: 2023, tdp: 220, recommended: false },
  { id: 'rtx4060ti-16', name: 'NVIDIA RTX 4060 Ti 16GB', vendor: 'nvidia', vram: 16, memoryBandwidth: 288, pcie: '4.0', releaseYear: 2023, tdp: 160, recommended: false },
  { id: 'rtx4060ti-8', name: 'NVIDIA RTX 4060 Ti 8GB', vendor: 'nvidia', vram: 8, memoryBandwidth: 288, pcie: '4.0', releaseYear: 2023, tdp: 160, recommended: false },
  { id: 'rtx4060-8', name: 'NVIDIA RTX 4060 8GB', vendor: 'nvidia', vram: 8, memoryBandwidth: 275, pcie: '4.0', releaseYear: 2023, tdp: 115, recommended: false },
  { id: 'rtx3090-24', name: 'NVIDIA RTX 3090 24GB', vendor: 'nvidia', vram: 24, memoryBandwidth: 936, pcie: '4.0', releaseYear: 2020, tdp: 350, recommended: false },
  { id: 'rtx3080-10', name: 'NVIDIA RTX 3080 10GB', vendor: 'nvidia', vram: 10, memoryBandwidth: 760, pcie: '4.0', releaseYear: 2020, tdp: 320, recommended: false },
  { id: 'rtx3080-12', name: 'NVIDIA RTX 3080 12GB', vendor: 'nvidia', vram: 12, memoryBandwidth: 722, pcie: '4.0', releaseYear: 2021, tdp: 320, recommended: false },
  { id: 'rtx3070-8', name: 'NVIDIA RTX 3070 8GB', vendor: 'nvidia', vram: 8, memoryBandwidth: 448, pcie: '4.0', releaseYear: 2020, tdp: 220, recommended: false },
  { id: 'rtx3060-12', name: 'NVIDIA RTX 3060 12GB', vendor: 'nvidia', vram: 12, memoryBandwidth: 360, pcie: '4.0', releaseYear: 2021, tdp: 170, recommended: false },
  { id: 'rtx3060-12v2', name: 'NVIDIA RTX 3060 12GB (rev)', vendor: 'nvidia', vram: 12, memoryBandwidth: 360, pcie: '3.0', releaseYear: 2022, tdp: 170, recommended: false },
  { id: 'l40s-48', name: 'NVIDIA L40S 48GB', vendor: 'nvidia', vram: 48, memoryBandwidth: 1812, pcie: '4.0', releaseYear: 2023, tdp: 300, recommended: false },
  { id: 'l4-24', name: 'NVIDIA L4 24GB', vendor: 'nvidia', vram: 24, memoryBandwidth: 864, pcie: '4.0', releaseYear: 2023, tdp: 72, recommended: false },
  { id: 't4-16', name: 'NVIDIA T4 16GB', vendor: 'nvidia', vram: 16, memoryBandwidth: 320, pcie: '3.0', releaseYear: 2018, tdp: 70, recommended: false },
  { id: 'v100-32', name: 'NVIDIA V100 32GB', vendor: 'nvidia', vram: 32, memoryBandwidth: 900, pcie: '3.0', releaseYear: 2017, tdp: 250, recommended: false },
  { id: 'v100-16', name: 'NVIDIA V100 16GB', vendor: 'nvidia', vram: 16, memoryBandwidth: 900, pcie: '3.0', releaseYear: 2019, tdp: 250, recommended: false },
  { id: 'v100-12', name: 'NVIDIA V100 12GB', vendor: 'nvidia', vram: 12, memoryBandwidth: 900, pcie: '3.0', releaseYear: 2017, tdp: 250, recommended: false },
  { id: 'mi300x-192', name: 'AMD MI300X 192GB', vendor: 'amd', vram: 192, memoryBandwidth: 1965, pcie: '5.0', releaseYear: 2024, tdp: 940, recommended: true },
  { id: 'mi250x-128', name: 'AMD MI250X 128GB', vendor: 'amd', vram: 128, memoryBandwidth: 1140, pcie: '4.0', releaseYear: 2022, tdp: 350, recommended: false },
  { id: 'mi250-64', name: 'AMD MI250 64GB', vendor: 'amd', vram: 64, memoryBandwidth: 1140, pcie: '4.0', releaseYear: 2022, tdp: 280, recommended: false },
  { id: 'mi100-32', name: 'AMD MI100 32GB', vendor: 'amd', vram: 32, memoryBandwidth: 1229, pcie: '4.0', releaseYear: 2019, tdp: 250, recommended: false },
  { id: 'rx7900xtx-24', name: 'AMD RX 7900 XTX 24GB', vendor: 'amd', vram: 24, memoryBandwidth: 1228, pcie: '4.0', releaseYear: 2022, tdp: 355, recommended: false },
  { id: 'rx7900xt-20', name: 'AMD RX 7900 XT 20GB', vendor: 'amd', vram: 20, memoryBandwidth: 935, pcie: '4.0', releaseYear: 2022, tdp: 300, recommended: false },
  { id: 'rx7900gre-16', name: 'AMD RX 7900 GRE 16GB', vendor: 'amd', vram: 16, memoryBandwidth: 869, pcie: '4.0', releaseYear: 2023, tdp: 260, recommended: false },
  { id: 'rx7800xt-16', name: 'AMD RX 7800 XT 16GB', vendor: 'amd', vram: 16, memoryBandwidth: 756, pcie: '4.0', releaseYear: 2023, tdp: 263, recommended: false },
  { id: 'rx7700xt-12', name: 'AMD RX 7700 XT 12GB', vendor: 'amd', vram: 12, memoryBandwidth: 575, pcie: '4.0', releaseYear: 2023, tdp: 245, recommended: false },
  { id: 'rx6900xt-16', name: 'AMD RX 6900 XT 16GB', vendor: 'amd', vram: 16, memoryBandwidth: 1024, pcie: '4.0', releaseYear: 2020, tdp: 300, recommended: false },
  { id: 'rx6800xt-16', name: 'AMD RX 6800 XT 16GB', vendor: 'amd', vram: 16, memoryBandwidth: 840, pcie: '4.0', releaseYear: 2020, tdp: 300, recommended: false },
  { id: 'm4-24', name: 'Apple M4 Max 24GB', vendor: 'apple', vram: 24, memoryBandwidth: 410, pcie: 'N/A', releaseYear: 2024, tdp: 45, recommended: false },
  { id: 'm4-32', name: 'Apple M4 Max 32GB', vendor: 'apple', vram: 32, memoryBandwidth: 410, pcie: 'N/A', releaseYear: 2024, tdp: 45, recommended: false },
  { id: 'm3-18', name: 'Apple M3 Max 18GB', vendor: 'apple', vram: 18, memoryBandwidth: 400, pcie: 'N/A', releaseYear: 2023, tdp: 30, recommended: false },
  { id: 'm3-36', name: 'Apple M3 Max 36GB', vendor: 'apple', vram: 36, memoryBandwidth: 400, pcie: 'N/A', releaseYear: 2023, tdp: 30, recommended: false },
  { id: 'm2-16', name: 'Apple M2 Ultra 16GB', vendor: 'apple', vram: 16, memoryBandwidth: 800, pcie: 'N/A', releaseYear: 2023, tdp: 100, recommended: false },
  { id: 'm2-64', name: 'Apple M2 Ultra 64GB', vendor: 'apple', vram: 64, memoryBandwidth: 800, pcie: 'N/A', releaseYear: 2023, tdp: 100, recommended: false },
  { id: 'm2-192', name: 'Apple M2 Ultra 192GB', vendor: 'apple', vram: 192, memoryBandwidth: 800, pcie: 'N/A', releaseYear: 2023, tdp: 100, recommended: false },
]

export const MODELS: ModelPreset[] = [
  // Qwen
  { id: 'qwen2.5-72b', name: 'Qwen 2.5 72B', family: 'Qwen', parameters: 72, quantization: 'bf16', vramRequired: 144, contextLength: 131072, recommendedGpuVram: 160, minGpuVram: 144, tags: ['chat', 'code', 'long-context'] },
  { id: 'qwen2.5-32b', name: 'Qwen 2.5 32B', family: 'Qwen', parameters: 32, quantization: 'bf16', vramRequired: 64, contextLength: 131072, recommendedGpuVram: 64, minGpuVram: 64, tags: ['chat', 'code'] },
  { id: 'qwen2.5-14b', name: 'Qwen 2.5 14B', family: 'Qwen', parameters: 14, quantization: 'bf16', vramRequired: 28, contextLength: 131072, recommendedGpuVram: 32, minGpuVram: 28, tags: ['chat', 'code'] },
  { id: 'qwen2.5-7b', name: 'Qwen 2.5 7B', family: 'Qwen', parameters: 7, quantization: 'bf16', vramRequired: 14, contextLength: 131072, recommendedGpuVram: 16, minGpuVram: 14, tags: ['chat', 'fast'] },
  { id: 'qwen2.5-3b', name: 'Qwen 2.5 3B', family: 'Qwen', parameters: 3, quantization: 'bf16', vramRequired: 6, contextLength: 131072, recommendedGpuVram: 8, minGpuVram: 6, tags: ['chat', 'fast', 'edge'] },
  { id: 'qwen2.5-1.5b', name: 'Qwen 2.5 1.5B', family: 'Qwen', parameters: 1.5, quantization: 'bf16', vramRequired: 3, contextLength: 131072, recommendedGpuVram: 4, minGpuVram: 3, tags: ['chat', 'fast', 'edge'] },
  { id: 'qwen2.5-72b-q4', name: 'Qwen 2.5 72B (Q4_K_M)', family: 'Qwen', parameters: 72, quantization: 'int4', vramRequired: 40, contextLength: 131072, recommendedGpuVram: 48, minGpuVram: 40, tags: ['chat', 'code', 'efficient'] },
  { id: 'qwen2.5-32b-q4', name: 'Qwen 2.5 32B (Q4_K_M)', family: 'Qwen', parameters: 32, quantization: 'int4', vramRequired: 20, contextLength: 131072, recommendedGpuVram: 24, minGpuVram: 20, tags: ['chat', 'code', 'efficient'] },

  // Llama
  { id: 'llama3.3-70b', name: 'Llama 3.3 70B', family: 'Llama', parameters: 70, quantization: 'bf16', vramRequired: 140, contextLength: 131072, recommendedGpuVram: 160, minGpuVram: 140, tags: ['chat', 'reasoning'] },
  { id: 'llama3.1-8b', name: 'Llama 3.1 8B', family: 'Llama', parameters: 8, quantization: 'bf16', vramRequired: 16, contextLength: 131072, recommendedGpuVram: 16, minGpuVram: 16, tags: ['chat', 'fast'] },
  { id: 'llama3.1-70b', name: 'Llama 3.1 70B', family: 'Llama', parameters: 70, quantization: 'bf16', vramRequired: 140, contextLength: 131072, recommendedGpuVram: 160, minGpuVram: 140, tags: ['chat', 'reasoning'] },
  { id: 'llama3.1-70b-q4', name: 'Llama 3.1 70B (Q4_K_M)', family: 'Llama', parameters: 70, quantization: 'int4', vramRequired: 38, contextLength: 131072, recommendedGpuVram: 48, minGpuVram: 38, tags: ['chat', 'efficient'] },
  { id: 'llama3.1-8b-q4', name: 'Llama 3.1 8B (Q4_K_M)', family: 'Llama', parameters: 8, quantization: 'int4', vramRequired: 5, contextLength: 131072, recommendedGpuVram: 8, minGpuVram: 5, tags: ['chat', 'fast', 'edge'] },
  { id: 'llama3-8b', name: 'Llama 3 8B', family: 'Llama', parameters: 8, quantization: 'bf16', vramRequired: 16, contextLength: 8192, recommendedGpuVram: 16, minGpuVram: 16, tags: ['chat', 'fast'] },

  // Mistral
  { id: 'mistral-large', name: 'Mistral Large', family: 'Mistral', parameters: 123, quantization: 'bf16', vramRequired: 246, contextLength: 131072, recommendedGpuVram: 256, minGpuVram: 246, tags: ['chat', 'reasoning'] },
  { id: 'mistral-small', name: 'Mistral Small', family: 'Mistral', parameters: 22, quantization: 'bf16', vramRequired: 44, contextLength: 32768, recommendedGpuVram: 48, minGpuVram: 44, tags: ['chat'] },
  { id: 'mistral-7b', name: 'Mistral 7B', family: 'Mistral', parameters: 7, quantization: 'bf16', vramRequired: 14, contextLength: 32768, recommendedGpuVram: 16, minGpuVram: 14, tags: ['chat', 'fast'] },
  { id: 'mistral-7b-q4', name: 'Mistral 7B (Q4_K_M)', family: 'Mistral', parameters: 7, quantization: 'int4', vramRequired: 4, contextLength: 32768, recommendedGpuVram: 8, minGpuVram: 4, tags: ['chat', 'fast', 'edge'] },
  { id: 'mixtral-8x7b', name: 'Mixtral 8x7B', family: 'Mistral', parameters: 47, quantization: 'bf16', vramRequired: 94, contextLength: 32768, recommendedGpuVram: 96, minGpuVram: 94, tags: ['chat', 'reasoning'] },
  { id: 'mixtral-8x22b', name: 'Mixtral 8x22B', family: 'Mistral', parameters: 141, quantization: 'bf16', vramRequired: 282, contextLength: 65536, recommendedGpuVram: 288, minGpuVram: 282, tags: ['chat', 'reasoning'] },

  // Phi
  { id: 'phi3-mini', name: 'Phi-3 Mini', family: 'Phi', parameters: 3.8, quantization: 'bf16', vramRequired: 7.6, contextLength: 128000, recommendedGpuVram: 8, minGpuVram: 8, tags: ['chat', 'fast'] },
  { id: 'phi3-small', name: 'Phi-3 Small', family: 'Phi', parameters: 7, quantization: 'bf16', vramRequired: 14, contextLength: 128000, recommendedGpuVram: 16, minGpuVram: 14, tags: ['chat'] },
  { id: 'phi3-medium', name: 'Phi-3 Medium', family: 'Phi', parameters: 14, quantization: 'bf16', vramRequired: 28, contextLength: 128000, recommendedGpuVram: 32, minGpuVram: 28, tags: ['chat'] },
  { id: 'phi3.5-mini', name: 'Phi-3.5 Mini', family: 'Phi', parameters: 3.8, quantization: 'bf16', vramRequired: 8, contextLength: 128000, recommendedGpuVram: 8, minGpuVram: 8, tags: ['chat', 'fast', 'multilingual'] },

  // Gemma
  { id: 'gemma2-27b', name: 'Gemma 2 27B', family: 'Gemma', parameters: 27, quantization: 'bf16', vramRequired: 54, contextLength: 8192, recommendedGpuVram: 56, minGpuVram: 54, tags: ['chat', 'reasoning'] },
  { id: 'gemma2-9b', name: 'Gemma 2 9B', family: 'Gemma', parameters: 9, quantization: 'bf16', vramRequired: 18, contextLength: 8192, recommendedGpuVram: 24, minGpuVram: 18, tags: ['chat', 'fast'] },
  { id: 'gemma2-2b', name: 'Gemma 2 2B', family: 'Gemma', parameters: 2.6, quantization: 'bf16', vramRequired: 5.2, contextLength: 4096, recommendedGpuVram: 8, minGpuVram: 5.2, tags: ['chat', 'fast', 'edge'] },

  // Stable
  { id: 'stable-code-3b', name: 'Stable Code 3B', family: 'Stable', parameters: 3, quantization: 'bf16', vramRequired: 6, contextLength: 16384, recommendedGpuVram: 8, minGpuVram: 6, tags: ['code', 'fast'] },
  { id: 'stable-llama-12b', name: 'Stable LM 12B', family: 'Stable', parameters: 12, quantization: 'bf16', vramRequired: 24, contextLength: 4096, recommendedGpuVram: 24, minGpuVram: 24, tags: ['chat', 'code'] },

  // Dolly / RedPajama
  { id: 'dolly-12b', name: 'Dolly 12B', family: 'Dolly', parameters: 12, quantization: 'bf16', vramRequired: 24, contextLength: 2048, recommendedGpuVram: 24, minGpuVram: 24, tags: ['chat'] },
  { id: 'redpajama-12b', name: 'RedPajama 12B', family: 'RedPajama', parameters: 12, quantization: 'bf16', vramRequired: 24, contextLength: 2048, recommendedGpuVram: 24, minGpuVram: 24, tags: ['chat'] },

  // Code-specific
  { id: 'codellama-34b', name: 'Code Llama 34B', family: 'CodeLlama', parameters: 34, quantization: 'bf16', vramRequired: 68, contextLength: 16384, recommendedGpuVram: 80, minGpuVram: 68, tags: ['code'] },
  { id: 'codellama-13b', name: 'Code Llama 13B', family: 'CodeLlama', parameters: 13, quantization: 'bf16', vramRequired: 26, contextLength: 16384, recommendedGpuVram: 32, minGpuVram: 26, tags: ['code'] },
  { id: 'codellama-7b', name: 'Code Llama 7B', family: 'CodeLlama', parameters: 7, quantization: 'bf16', vramRequired: 14, contextLength: 16384, recommendedGpuVram: 16, minGpuVram: 14, tags: ['code', 'fast'] },
  { id: 'deepseek-coder-33b', name: 'DeepSeek Coder 33B', family: 'DeepSeek', parameters: 33, quantization: 'bf16', vramRequired: 66, contextLength: 16384, recommendedGpuVram: 80, minGpuVram: 66, tags: ['code'] },
  { id: 'deepseek-coder-6.7b', name: 'DeepSeek Coder 6.7B', family: 'DeepSeek', parameters: 6.7, quantization: 'bf16', vramRequired: 13.4, contextLength: 16384, recommendedGpuVram: 16, minGpuVram: 14, tags: ['code', 'fast'] },

  // Chat-specific
  { id: 'alpaca-7b', name: 'Alpaca 7B', family: 'Alpaca', parameters: 7, quantization: 'bf16', vramRequired: 14, contextLength: 512, recommendedGpuVram: 16, minGpuVram: 14, tags: ['chat'] },
  { id: 'guanaco-7b', name: 'Guanaco 7B', family: 'Guanaco', parameters: 7, quantization: 'bf16', vramRequired: 14, contextLength: 2048, recommendedGpuVram: 16, minGpuVram: 14, tags: ['chat'] },
  { id: 'falcon-40b', name: 'Falcon 40B', family: 'Falcon', parameters: 40, quantization: 'bf16', vramRequired: 80, contextLength: 2048, recommendedGpuVram: 80, minGpuVram: 80, tags: ['chat'] },
  { id: 'falcon-7b', name: 'Falcon 7B', family: 'Falcon', parameters: 7, quantization: 'bf16', vramRequired: 14, contextLength: 2048, recommendedGpuVram: 16, minGpuVram: 14, tags: ['chat', 'fast'] },

  // Embedding / small models
  { id: 'bert-base', name: 'BERT Base', family: 'BERT', parameters: 0.11, quantization: 'fp16', vramRequired: 0.25, contextLength: 512, recommendedGpuVram: 1, minGpuVram: 0.5, tags: ['embedding', 'nlp'] },
  { id: 'roberta-large', name: 'RoBERTa Large', family: 'RoBERTa', parameters: 0.34, quantization: 'fp16', vramRequired: 0.7, contextLength: 512, recommendedGpuVram: 2, minGpuVram: 1, tags: ['embedding', 'nlp'] },
  { id: 'sentence-transformers', name: 'Sentence Transformers', family: 'SentenceTransformers', parameters: 0.11, quantization: 'fp16', vramRequired: 0.3, contextLength: 128, recommendedGpuVram: 1, minGpuVram: 0.5, tags: ['embedding', 'similarity'] },

  // Diffusion models (for vision)
  { id: 'sd-xl', name: 'Stable Diffusion XL', family: 'StableDiffusion', parameters: 6600, quantization: 'fp16', vramRequired: 8, contextLength: 0, recommendedGpuVram: 12, minGpuVram: 6, tags: ['image-gen', 'diffusion'] },
  { id: 'sd-1.5', name: 'Stable Diffusion 1.5', family: 'StableDiffusion', parameters: 860, quantization: 'fp16', vramRequired: 4, contextLength: 0, recommendedGpuVram: 6, minGpuVram: 4, tags: ['image-gen', 'diffusion'] },
  { id: 'sd3', name: 'Stable Diffusion 3', family: 'StableDiffusion', parameters: 2000, quantization: 'fp16', vramRequired: 12, contextLength: 0, recommendedGpuVram: 16, minGpuVram: 10, tags: ['image-gen', 'diffusion'] },
  { id: 'flux', name: 'FLUX.1', family: 'FLUX', parameters: 8000, quantization: 'fp8', vramRequired: 18, contextLength: 0, recommendedGpuVram: 24, minGpuVram: 16, tags: ['image-gen', 'diffusion'] },
  { id: 'flux-dev', name: 'FLUX.1 Dev', family: 'FLUX', parameters: 8000, quantization: 'bf16', vramRequired: 36, contextLength: 0, recommendedGpuVram: 48, minGpuVram: 32, tags: ['image-gen', 'diffusion'] },
  { id: 'flux-schnell', name: 'FLUX.1 Schnell', family: 'FLUX', parameters: 8000, quantization: 'bf16', vramRequired: 16, contextLength: 0, recommendedGpuVram: 24, minGpuVram: 16, tags: ['image-gen', 'diffusion', 'fast'] },
]

export function calculatePerformance(model: ModelPreset, gpuVram: number, allocatedVram: number): PerformanceBucket {
  if (allocatedVram > gpuVram) {
    return 'infeasible'
  }
  if (allocatedVram <= gpuVram * 0.7) {
    return 'realtime'
  }
  if (allocatedVram <= gpuVram * 0.85) {
    return 'fast'
  }
  if (allocatedVram <= gpuVram * 0.95) {
    return 'moderate'
  }
  if (allocatedVram <= gpuVram) {
    return 'slow'
  }
  return 'infeasible'
}

export function calculatePartitionVram(model: ModelPreset, gpuVram: number): number {
  const overhead = gpuVram * 0.05
  return Math.min(model.vramRequired, gpuVram - overhead)
}

export function getPerformanceColor(bucket: PerformanceBucket): string {
  switch (bucket) {
    case 'realtime': return 'text-green-400'
    case 'fast': return 'text-emerald-400'
    case 'moderate': return 'text-yellow-400'
    case 'slow': return 'text-orange-400'
    case 'infeasible': return 'text-red-400'
  }
}

export function getPartitionBarColor(bucket: PerformanceBucket): string {
  switch (bucket) {
    case 'realtime': return 'bg-green-500'
    case 'fast': return 'bg-emerald-500'
    case 'moderate': return 'bg-yellow-500'
    case 'slow': return 'bg-orange-500'
    case 'infeasible': return 'bg-red-500'
  }
}

export function getRecommendedGpusForModel(model: ModelPreset): GpuSpec[] {
  return GPUs.filter(gpu => gpu.vram >= model.minGpuVram)
    .sort((a, b) => {
      if (a.recommended && !b.recommended) return -1
      if (!a.recommended && b.recommended) return 1
      return a.vram - b.vram
    })
}

export function getGpuForPartition(partition: Partition): GpuSpec {
  return GPUs.find(g => g.id === partition.id) || GPUs[0]
}

export function filterGpusByVram(minVram?: number, maxVram?: number): GpuSpec[] {
  return GPUs.filter(gpu => {
    if (minVram && gpu.vram < minVram) return false
    if (maxVram && gpu.vram > maxVram) return false
    return true
  }).sort((a, b) => b.vram - a.vram)
}

export function filterModelsByParams(minParams?: number, maxParams?: number): ModelPreset[] {
  return MODELS.filter(model => {
    if (minParams && model.parameters < minParams) return false
    if (maxParams && model.parameters > maxParams) return false
    return true
  }).sort((a, b) => b.parameters - a.parameters)
}

export function getCompatibleModelsForGpu(gpu: GpuSpec): ModelPreset[] {
  return MODELS.filter(model => model.minGpuVram <= gpu.vram)
}

export function estimateTokensPerSecond(model: ModelPreset, gpu: GpuSpec, partitions: Partition[]): number {
  if (partitions.length === 0) return 0
  const totalParams = partitions.reduce((sum, p) => {
    const m = MODELS.find(m => m.id === p.modelId)
    return sum + (m?.parameters || 0)
  }, 0)
  const normalizedVram = gpu.vram / Math.max(totalParams * 0.5, 1)
  const bandwidthFactor = gpu.memoryBandwidth / 1000
  return Math.round(normalizedVram * bandwidthFactor * 10)
}
