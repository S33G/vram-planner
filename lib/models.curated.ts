import type { ModelPreset } from './db'

// KV cache per 1k tokens derived from: num_heads * head_dim * num_layers * 2 (K+V) * bytes_per_element / 1e9 * 1000
// Approximations scaled by quantization and parameter count.
// Headroom covers CUDA context, activations, framework overhead (~1–4 GB typical).
export const CURATED_MODELS: ModelPreset[] = [
  // Qwen
  { id: 'qwen2.5-72b',    name: 'Qwen 2.5 72B',          family: 'Qwen',    parameters: 72,   quantization: 'bf16', weightsGb: 144,  kvCachePer1kTokens: 0.34, headroomGb: 4, contextLength: 131072, recommendedGpuVram: 160, minGpuVram: 144, tags: ['chat', 'code', 'long-context'] },
  { id: 'qwen2.5-32b',    name: 'Qwen 2.5 32B',          family: 'Qwen',    parameters: 32,   quantization: 'bf16', weightsGb: 64,   kvCachePer1kTokens: 0.16, headroomGb: 3, contextLength: 131072, recommendedGpuVram: 64,  minGpuVram: 64,  tags: ['chat', 'code'] },
  { id: 'qwen2.5-14b',    name: 'Qwen 2.5 14B',          family: 'Qwen',    parameters: 14,   quantization: 'bf16', weightsGb: 28,   kvCachePer1kTokens: 0.08, headroomGb: 2, contextLength: 131072, recommendedGpuVram: 32,  minGpuVram: 28,  tags: ['chat', 'code'] },
  { id: 'qwen2.5-7b',     name: 'Qwen 2.5 7B',           family: 'Qwen',    parameters: 7,    quantization: 'bf16', weightsGb: 14,   kvCachePer1kTokens: 0.06, headroomGb: 2, contextLength: 131072, recommendedGpuVram: 16,  minGpuVram: 14,  tags: ['chat', 'fast'] },
  { id: 'qwen2.5-3b',     name: 'Qwen 2.5 3B',           family: 'Qwen',    parameters: 3,    quantization: 'bf16', weightsGb: 6,    kvCachePer1kTokens: 0.03, headroomGb: 1, contextLength: 131072, recommendedGpuVram: 8,   minGpuVram: 6,   tags: ['chat', 'fast', 'edge'] },
  { id: 'qwen2.5-1.5b',   name: 'Qwen 2.5 1.5B',         family: 'Qwen',    parameters: 1.5,  quantization: 'bf16', weightsGb: 3,    kvCachePer1kTokens: 0.02, headroomGb: 1, contextLength: 131072, recommendedGpuVram: 4,   minGpuVram: 3,   tags: ['chat', 'fast', 'edge'] },
  { id: 'qwen2.5-72b-q4', name: 'Qwen 2.5 72B (Q4_K_M)', family: 'Qwen',    parameters: 72,   quantization: 'int4', weightsGb: 40,   kvCachePer1kTokens: 0.17, headroomGb: 3, contextLength: 131072, recommendedGpuVram: 48,  minGpuVram: 40,  tags: ['chat', 'code', 'efficient'] },
  { id: 'qwen2.5-32b-q4', name: 'Qwen 2.5 32B (Q4_K_M)', family: 'Qwen',    parameters: 32,   quantization: 'int4', weightsGb: 20,   kvCachePer1kTokens: 0.08, headroomGb: 2, contextLength: 131072, recommendedGpuVram: 24,  minGpuVram: 20,  tags: ['chat', 'code', 'efficient'] },

  // Llama
  { id: 'llama3.3-70b',   name: 'Llama 3.3 70B',         family: 'Llama',   parameters: 70,   quantization: 'bf16', weightsGb: 140,  kvCachePer1kTokens: 0.34, headroomGb: 4, contextLength: 131072, recommendedGpuVram: 160, minGpuVram: 140, tags: ['chat', 'reasoning'] },
  { id: 'llama3.1-8b',    name: 'Llama 3.1 8B',          family: 'Llama',   parameters: 8,    quantization: 'bf16', weightsGb: 16,   kvCachePer1kTokens: 0.06, headroomGb: 2, contextLength: 131072, recommendedGpuVram: 16,  minGpuVram: 16,  tags: ['chat', 'fast'] },
  { id: 'llama3.1-70b',   name: 'Llama 3.1 70B',         family: 'Llama',   parameters: 70,   quantization: 'bf16', weightsGb: 140,  kvCachePer1kTokens: 0.34, headroomGb: 4, contextLength: 131072, recommendedGpuVram: 160, minGpuVram: 140, tags: ['chat', 'reasoning'] },
  { id: 'llama3.1-70b-q4',name: 'Llama 3.1 70B (Q4_K_M)',family: 'Llama',   parameters: 70,   quantization: 'int4', weightsGb: 38,   kvCachePer1kTokens: 0.17, headroomGb: 3, contextLength: 131072, recommendedGpuVram: 48,  minGpuVram: 38,  tags: ['chat', 'efficient'] },
  { id: 'llama3.1-8b-q4', name: 'Llama 3.1 8B (Q4_K_M)', family: 'Llama',   parameters: 8,    quantization: 'int4', weightsGb: 5,    kvCachePer1kTokens: 0.03, headroomGb: 1, contextLength: 131072, recommendedGpuVram: 8,   minGpuVram: 5,   tags: ['chat', 'fast', 'edge'] },
  { id: 'llama3-8b',      name: 'Llama 3 8B',            family: 'Llama',   parameters: 8,    quantization: 'bf16', weightsGb: 16,   kvCachePer1kTokens: 0.06, headroomGb: 2, contextLength: 8192,   recommendedGpuVram: 16,  minGpuVram: 16,  tags: ['chat', 'fast'] },

  // Mistral
  { id: 'mistral-large',  name: 'Mistral Large',          family: 'Mistral', parameters: 123,  quantization: 'bf16', weightsGb: 246,  kvCachePer1kTokens: 0.50, headroomGb: 5, contextLength: 131072, recommendedGpuVram: 256, minGpuVram: 246, tags: ['chat', 'reasoning'] },
  { id: 'mistral-small',  name: 'Mistral Small',          family: 'Mistral', parameters: 22,   quantization: 'bf16', weightsGb: 44,   kvCachePer1kTokens: 0.12, headroomGb: 2, contextLength: 32768,  recommendedGpuVram: 48,  minGpuVram: 44,  tags: ['chat'] },
  { id: 'mistral-7b',     name: 'Mistral 7B',             family: 'Mistral', parameters: 7,    quantization: 'bf16', weightsGb: 14,   kvCachePer1kTokens: 0.06, headroomGb: 2, contextLength: 32768,  recommendedGpuVram: 16,  minGpuVram: 14,  tags: ['chat', 'fast'] },
  { id: 'mistral-7b-q4',  name: 'Mistral 7B (Q4_K_M)',    family: 'Mistral', parameters: 7,    quantization: 'int4', weightsGb: 4,    kvCachePer1kTokens: 0.03, headroomGb: 1, contextLength: 32768,  recommendedGpuVram: 8,   minGpuVram: 4,   tags: ['chat', 'fast', 'edge'] },
  { id: 'mixtral-8x7b',   name: 'Mixtral 8x7B',           family: 'Mistral', parameters: 47,   quantization: 'bf16', weightsGb: 94,   kvCachePer1kTokens: 0.24, headroomGb: 4, contextLength: 32768,  recommendedGpuVram: 96,  minGpuVram: 94,  tags: ['chat', 'reasoning'] },
  { id: 'mixtral-8x22b',  name: 'Mixtral 8x22B',          family: 'Mistral', parameters: 141,  quantization: 'bf16', weightsGb: 282,  kvCachePer1kTokens: 0.50, headroomGb: 5, contextLength: 65536,  recommendedGpuVram: 288, minGpuVram: 282, tags: ['chat', 'reasoning'] },

  // Phi
  { id: 'phi3-mini',      name: 'Phi-3 Mini',             family: 'Phi',     parameters: 3.8,  quantization: 'bf16', weightsGb: 7.6,  kvCachePer1kTokens: 0.03, headroomGb: 1, contextLength: 128000, recommendedGpuVram: 8,   minGpuVram: 8,   tags: ['chat', 'fast'] },
  { id: 'phi3-small',     name: 'Phi-3 Small',            family: 'Phi',     parameters: 7,    quantization: 'bf16', weightsGb: 14,   kvCachePer1kTokens: 0.06, headroomGb: 2, contextLength: 128000, recommendedGpuVram: 16,  minGpuVram: 14,  tags: ['chat'] },
  { id: 'phi3-medium',    name: 'Phi-3 Medium',           family: 'Phi',     parameters: 14,   quantization: 'bf16', weightsGb: 28,   kvCachePer1kTokens: 0.08, headroomGb: 2, contextLength: 128000, recommendedGpuVram: 32,  minGpuVram: 28,  tags: ['chat'] },
  { id: 'phi3.5-mini',    name: 'Phi-3.5 Mini',           family: 'Phi',     parameters: 3.8,  quantization: 'bf16', weightsGb: 8,    kvCachePer1kTokens: 0.03, headroomGb: 1, contextLength: 128000, recommendedGpuVram: 8,   minGpuVram: 8,   tags: ['chat', 'fast', 'multilingual'] },

  // Gemma
  { id: 'gemma2-27b',     name: 'Gemma 2 27B',            family: 'Gemma',   parameters: 27,   quantization: 'bf16', weightsGb: 54,   kvCachePer1kTokens: 0.14, headroomGb: 3, contextLength: 8192,   recommendedGpuVram: 56,  minGpuVram: 54,  tags: ['chat', 'reasoning'] },
  { id: 'gemma2-9b',      name: 'Gemma 2 9B',             family: 'Gemma',   parameters: 9,    quantization: 'bf16', weightsGb: 18,   kvCachePer1kTokens: 0.07, headroomGb: 2, contextLength: 8192,   recommendedGpuVram: 24,  minGpuVram: 18,  tags: ['chat', 'fast'] },
  { id: 'gemma2-2b',      name: 'Gemma 2 2B',             family: 'Gemma',   parameters: 2.6,  quantization: 'bf16', weightsGb: 5.2,  kvCachePer1kTokens: 0.02, headroomGb: 1, contextLength: 4096,   recommendedGpuVram: 8,   minGpuVram: 5,   tags: ['chat', 'fast', 'edge'] },

  // Stable
  { id: 'stable-code-3b', name: 'Stable Code 3B',         family: 'Stable',  parameters: 3,    quantization: 'bf16', weightsGb: 6,    kvCachePer1kTokens: 0.02, headroomGb: 1, contextLength: 16384,  recommendedGpuVram: 8,   minGpuVram: 6,   tags: ['code', 'fast'] },
  { id: 'stable-llama-12b',name: 'Stable LM 12B',         family: 'Stable',  parameters: 12,   quantization: 'bf16', weightsGb: 24,   kvCachePer1kTokens: 0.07, headroomGb: 2, contextLength: 4096,   recommendedGpuVram: 24,  minGpuVram: 24,  tags: ['chat', 'code'] },

  // Dolly / RedPajama
  { id: 'dolly-12b',      name: 'Dolly 12B',              family: 'Dolly',   parameters: 12,   quantization: 'bf16', weightsGb: 24,   kvCachePer1kTokens: 0.07, headroomGb: 2, contextLength: 2048,   recommendedGpuVram: 24,  minGpuVram: 24,  tags: ['chat'] },
  { id: 'redpajama-12b',  name: 'RedPajama 12B',          family: 'RedPajama',parameters: 12,  quantization: 'bf16', weightsGb: 24,   kvCachePer1kTokens: 0.07, headroomGb: 2, contextLength: 2048,   recommendedGpuVram: 24,  minGpuVram: 24,  tags: ['chat'] },

  // Code
  { id: 'codellama-34b',  name: 'Code Llama 34B',         family: 'CodeLlama',parameters: 34,  quantization: 'bf16', weightsGb: 68,   kvCachePer1kTokens: 0.20, headroomGb: 3, contextLength: 16384,  recommendedGpuVram: 80,  minGpuVram: 68,  tags: ['code'] },
  { id: 'codellama-13b',  name: 'Code Llama 13B',         family: 'CodeLlama',parameters: 13,  quantization: 'bf16', weightsGb: 26,   kvCachePer1kTokens: 0.10, headroomGb: 2, contextLength: 16384,  recommendedGpuVram: 32,  minGpuVram: 26,  tags: ['code'] },
  { id: 'codellama-7b',   name: 'Code Llama 7B',          family: 'CodeLlama',parameters: 7,   quantization: 'bf16', weightsGb: 14,   kvCachePer1kTokens: 0.06, headroomGb: 2, contextLength: 16384,  recommendedGpuVram: 16,  minGpuVram: 14,  tags: ['code', 'fast'] },
  { id: 'deepseek-coder-33b',name: 'DeepSeek Coder 33B',  family: 'DeepSeek', parameters: 33,  quantization: 'bf16', weightsGb: 66,   kvCachePer1kTokens: 0.20, headroomGb: 3, contextLength: 16384,  recommendedGpuVram: 80,  minGpuVram: 66,  tags: ['code'] },
  { id: 'deepseek-coder-6.7b',name: 'DeepSeek Coder 6.7B',family: 'DeepSeek', parameters: 6.7, quantization: 'bf16', weightsGb: 13.4, kvCachePer1kTokens: 0.06, headroomGb: 2, contextLength: 16384,  recommendedGpuVram: 16,  minGpuVram: 14,  tags: ['code', 'fast'] },

  // Chat
  { id: 'alpaca-7b',      name: 'Alpaca 7B',              family: 'Alpaca',  parameters: 7,    quantization: 'bf16', weightsGb: 14,   kvCachePer1kTokens: 0.06, headroomGb: 2, contextLength: 512,    recommendedGpuVram: 16,  minGpuVram: 14,  tags: ['chat'] },
  { id: 'guanaco-7b',     name: 'Guanaco 7B',             family: 'Guanaco', parameters: 7,    quantization: 'bf16', weightsGb: 14,   kvCachePer1kTokens: 0.06, headroomGb: 2, contextLength: 2048,   recommendedGpuVram: 16,  minGpuVram: 14,  tags: ['chat'] },
  { id: 'falcon-40b',     name: 'Falcon 40B',             family: 'Falcon',  parameters: 40,   quantization: 'bf16', weightsGb: 80,   kvCachePer1kTokens: 0.20, headroomGb: 3, contextLength: 2048,   recommendedGpuVram: 80,  minGpuVram: 80,  tags: ['chat'] },
  { id: 'falcon-7b',      name: 'Falcon 7B',              family: 'Falcon',  parameters: 7,    quantization: 'bf16', weightsGb: 14,   kvCachePer1kTokens: 0.06, headroomGb: 2, contextLength: 2048,   recommendedGpuVram: 16,  minGpuVram: 14,  tags: ['chat', 'fast'] },

  // Embedding
  { id: 'bert-base',          name: 'BERT Base',           family: 'BERT',              parameters: 0.11, quantization: 'fp16', weightsGb: 0.25, kvCachePer1kTokens: 0,    headroomGb: 0.5, contextLength: 512, recommendedGpuVram: 1,  minGpuVram: 0.5, tags: ['embedding', 'nlp'] },
  { id: 'roberta-large',      name: 'RoBERTa Large',        family: 'RoBERTa',           parameters: 0.34, quantization: 'fp16', weightsGb: 0.7,  kvCachePer1kTokens: 0,    headroomGb: 0.5, contextLength: 512, recommendedGpuVram: 2,  minGpuVram: 1,   tags: ['embedding', 'nlp'] },
  { id: 'sentence-transformers',name: 'Sentence Transformers',family: 'SentenceTransformers',parameters: 0.11,quantization: 'fp16',weightsGb: 0.3,  kvCachePer1kTokens: 0,    headroomGb: 0.5, contextLength: 128, recommendedGpuVram: 1,  minGpuVram: 0.5, tags: ['embedding', 'similarity'] },

  // Diffusion
  { id: 'sd-xl',       name: 'Stable Diffusion XL',  family: 'StableDiffusion', parameters: 6600, quantization: 'fp16', weightsGb: 8,  kvCachePer1kTokens: 0, headroomGb: 2, contextLength: 0, recommendedGpuVram: 12, minGpuVram: 6,  tags: ['image-gen', 'diffusion'] },
  { id: 'sd-1.5',      name: 'Stable Diffusion 1.5', family: 'StableDiffusion', parameters: 860,  quantization: 'fp16', weightsGb: 4,  kvCachePer1kTokens: 0, headroomGb: 1, contextLength: 0, recommendedGpuVram: 6,  minGpuVram: 4,  tags: ['image-gen', 'diffusion'] },
  { id: 'sd3',         name: 'Stable Diffusion 3',   family: 'StableDiffusion', parameters: 2000, quantization: 'fp16', weightsGb: 12, kvCachePer1kTokens: 0, headroomGb: 2, contextLength: 0, recommendedGpuVram: 16, minGpuVram: 10, tags: ['image-gen', 'diffusion'] },
  { id: 'flux',        name: 'FLUX.1',               family: 'FLUX',            parameters: 8000, quantization: 'fp8',  weightsGb: 18, kvCachePer1kTokens: 0, headroomGb: 3, contextLength: 0, recommendedGpuVram: 24, minGpuVram: 16, tags: ['image-gen', 'diffusion'] },
  { id: 'flux-dev',    name: 'FLUX.1 Dev',           family: 'FLUX',            parameters: 8000, quantization: 'bf16', weightsGb: 36, kvCachePer1kTokens: 0, headroomGb: 4, contextLength: 0, recommendedGpuVram: 48, minGpuVram: 32, tags: ['image-gen', 'diffusion'] },
  { id: 'flux-schnell',name: 'FLUX.1 Schnell',       family: 'FLUX',            parameters: 8000, quantization: 'bf16', weightsGb: 16, kvCachePer1kTokens: 0, headroomGb: 3, contextLength: 0, recommendedGpuVram: 24, minGpuVram: 16, tags: ['image-gen', 'diffusion', 'fast'] },
]
