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
  /** Loaded weight footprint in GB */
  weightsGb: number
  /** KV cache cost in GB per 1k tokens (at default context length) */
  kvCachePer1kTokens: number
  /** Runtime headroom: CUDA context, activations, scratch buffers, etc. */
  headroomGb: number
  contextLength: number
  recommendedGpuVram: number
  minGpuVram: number
  tags: string[]
}

export type QuantizationType = 'fp16' | 'bf16' | 'fp8' | 'int8' | 'int4' | 'nf4' | 'mixed'
export type PerformanceBucket = 'realtime' | 'fast' | 'moderate' | 'slow' | 'infeasible'

export type SpillPolicy = 'avoid' | 'allow'
export type FitStatus = 'fits' | 'spills' | 'no-fit'

export interface VramBreakdown {
  weights: number
  kvCache: number      // total KV across all slots at given context
  headroom: number
  total: number
  overflow: number     // how much exceeds gpuVram (0 if fits)
  gpuUsed: number      // min(total, gpuVram)
  free: number         // max(0, gpuVram - total)
  ramSpill: number     // overflow if spillPolicy=allow, else 0
  fits: boolean
  fitStatus: FitStatus
}

export interface PerfShape {
  prefill: string
  tokenGen: string
  risk: string
  effectiveCtx: string
}

export interface Recommendation {
  title: string
  text: string
}

export interface Partition {
  id: string
  name: string
  modelId: string
  slot: number
  slots: number         // parallel inference slots
  contextLength: number // user-overridable context
  vramAllocated: number
  breakdown: VramBreakdown
  perf: PerfShape
  recommendations: Recommendation[]
  performance: PerformanceBucket
}

export interface GpuState {
  gpu: GpuSpec
  partitions: Partition[]
  totalUsedVram: number
  totalFreeVram: number
}

/**
 * Compute the realistic VRAM breakdown for a model on a given GPU.
 * Formula mirrors the reference HTML: weights + (kvPer1k * ctx/1k * slots) + headroom = required.
 */
export function calculateVramBreakdown(
  model: ModelPreset,
  gpuVram: number,
  opts: { slots?: number; contextLength?: number; systemRamGb?: number; spillPolicy?: SpillPolicy } = {}
): VramBreakdown {
  const slots = opts.slots ?? 1
  const ctx = opts.contextLength ?? model.contextLength
  const systemRam = opts.systemRamGb ?? 64
  const spillPolicy = opts.spillPolicy ?? 'avoid'

  const kvCache = ctx > 0 ? (ctx / 1024) * model.kvCachePer1kTokens * slots : 0
  const total = model.weightsGb + kvCache + model.headroomGb
  const overflow = Math.max(0, total - gpuVram)
  const gpuUsed = Math.min(total, gpuVram)
  const free = Math.max(0, gpuVram - total)
  const ramSpill = spillPolicy === 'allow' ? overflow : 0
  const fits = overflow === 0

  let fitStatus: FitStatus
  if (fits) {
    fitStatus = 'fits'
  } else if (spillPolicy === 'allow' && overflow <= systemRam) {
    fitStatus = 'spills'
  } else {
    fitStatus = 'no-fit'
  }

  return { weights: model.weightsGb, kvCache, headroom: model.headroomGb, total, overflow, gpuUsed, free, ramSpill, fits, fitStatus }
}

export function calcPerfShape(breakdown: VramBreakdown, contextLength: number, slots: number): PerfShape {
  const totalCtx = contextLength * slots
  const usageRatio = breakdown.gpuUsed / Math.max(breakdown.gpuUsed + breakdown.free, 1)

  let prefill = 'Fast'
  let tokenGen = 'Fast'
  let risk = 'Low'

  if (totalCtx > 65536 || usageRatio > 0.9)   { prefill = 'Moderate'; risk = 'Moderate' }
  if (totalCtx > 131072 || usageRatio > 0.98)  { prefill = 'Slow' }
  if (breakdown.overflow > 0 && breakdown.fitStatus === 'spills') { prefill = 'Slow'; tokenGen = 'Moderate'; risk = 'High' }
  if (breakdown.overflow > 2 && breakdown.fitStatus === 'spills') { tokenGen = 'Slow' }
  if (breakdown.fitStatus === 'no-fit')         { tokenGen = 'Blocked'; risk = 'Does not fit' }

  const fmt = (n: number) => n >= 1000 ? `${(n / 1000).toFixed(0)}k` : `${n}`
  return { prefill, tokenGen, risk, effectiveCtx: `${fmt(totalCtx)} tokens` }
}

export function calcRecommendations(breakdown: VramBreakdown, contextLength: number, slots: number, gpuVram: number, systemRamGb: number): Recommendation[] {
  const usageRatio = breakdown.gpuUsed / Math.max(gpuVram, 1)
  const recs: Recommendation[] = []

  if (!breakdown.overflow && usageRatio < 0.85)
    recs.push({ title: 'Healthy margin', text: 'You still have headroom for a larger batch size, more slots, or a slightly longer context.' })
  if (usageRatio >= 0.85 && !breakdown.overflow)
    recs.push({ title: 'Tight fit', text: 'Stay on-GPU if possible, but avoid adding many slots unless you reduce context or choose a smaller quant.' })
  if (breakdown.overflow && breakdown.fitStatus === 'spills')
    recs.push({ title: 'RAM fallback', text: `About ${breakdown.overflow.toFixed(1)} GB spills beyond VRAM. Expect slower prompt processing and weaker responsiveness.` })
  if (breakdown.fitStatus === 'no-fit')
    recs.push({ title: 'Resize partitions', text: 'Reduce context, reduce slots, or pick a smaller model/quant until the plan fits entirely inside VRAM.' })
  if (slots > 1)
    recs.push({ title: 'Concurrency cost', text: 'Each slot gets its own context allocation. Parallel requests multiply KV pressure.' })
  if (contextLength >= 32768)
    recs.push({ title: 'Long context tax', text: 'Large windows mainly hurt prefill. Keep long context only for workflows that really need it.' })
  if (breakdown.weights > gpuVram * 0.8)
    recs.push({ title: 'Weights dominate', text: 'This model leaves little room for KV cache. A smaller quant usually buys more than shaving headroom.' })
  if (systemRamGb < breakdown.overflow && breakdown.overflow > 0 && breakdown.fitStatus === 'spills')
    recs.push({ title: 'Host RAM limit', text: 'Even the spill path is undersized. Increase system RAM or downsize the plan.' })
  if (!recs.length)
    recs.push({ title: 'Balanced plan', text: 'This layout is broadly sensible for interactive use.' })

  return recs
}

export const GPUs: GpuSpec[] = [
  // RTX 50 series (Blackwell, GDDR7, PCIe 5.0)
  { id: 'rtx5090-32',    name: 'NVIDIA RTX 5090 32GB',         vendor: 'nvidia', vram: 32,  memoryBandwidth: 1792, pcie: '5.0', releaseYear: 2025, tdp: 575, recommended: false },
  { id: 'rtx5080-16',    name: 'NVIDIA RTX 5080 16GB',         vendor: 'nvidia', vram: 16,  memoryBandwidth: 960,  pcie: '5.0', releaseYear: 2025, tdp: 360, recommended: false },
  { id: 'rtx5070ti-16',  name: 'NVIDIA RTX 5070 Ti 16GB',      vendor: 'nvidia', vram: 16,  memoryBandwidth: 896,  pcie: '5.0', releaseYear: 2025, tdp: 300, recommended: false },
  { id: 'rtx5070-12',    name: 'NVIDIA RTX 5070 12GB',         vendor: 'nvidia', vram: 12,  memoryBandwidth: 896,  pcie: '5.0', releaseYear: 2025, tdp: 250, recommended: false },
  { id: 'rtx5060ti-16',  name: 'NVIDIA RTX 5060 Ti 16GB',      vendor: 'nvidia', vram: 16,  memoryBandwidth: 672,  pcie: '5.0', releaseYear: 2025, tdp: 180, recommended: false },
  { id: 'rtx5060ti-8',   name: 'NVIDIA RTX 5060 Ti 8GB',       vendor: 'nvidia', vram: 8,   memoryBandwidth: 672,  pcie: '5.0', releaseYear: 2025, tdp: 180, recommended: false },
  { id: 'rtx5060-8',     name: 'NVIDIA RTX 5060 8GB',          vendor: 'nvidia', vram: 8,   memoryBandwidth: 448,  pcie: '5.0', releaseYear: 2025, tdp: 145, recommended: false },
  { id: 'rtx5050-8',     name: 'NVIDIA RTX 5050 8GB',          vendor: 'nvidia', vram: 8,   memoryBandwidth: 320,  pcie: '5.0', releaseYear: 2025, tdp: 130, recommended: false },
  // Data center / workstation
  { id: 'h100-80',     name: 'NVIDIA H100 80GB',           vendor: 'nvidia', vram: 80,  memoryBandwidth: 3350, pcie: '4.0', releaseYear: 2022, tdp: 700, recommended: true },
  { id: 'h100-40',     name: 'NVIDIA H100 40GB SXM',       vendor: 'nvidia', vram: 40,  memoryBandwidth: 3350, pcie: '4.0', releaseYear: 2022, tdp: 350, recommended: true },
  { id: 'a100-80',     name: 'NVIDIA A100 80GB',           vendor: 'nvidia', vram: 80,  memoryBandwidth: 2000, pcie: '4.0', releaseYear: 2020, tdp: 400, recommended: true },
  { id: 'a100-40',     name: 'NVIDIA A100 40GB',           vendor: 'nvidia', vram: 40,  memoryBandwidth: 2000, pcie: '4.0', releaseYear: 2020, tdp: 300, recommended: false },
  { id: 'a6000-48',    name: 'NVIDIA RTX Ada 6000 48GB',   vendor: 'nvidia', vram: 48,  memoryBandwidth: 900,  pcie: '4.0', releaseYear: 2022, tdp: 300, recommended: false },
  { id: 'rtx4090-24',  name: 'NVIDIA RTX 4090 24GB',       vendor: 'nvidia', vram: 24,  memoryBandwidth: 1008, pcie: '4.0', releaseYear: 2022, tdp: 450, recommended: true },
  { id: 'rtx4080s-16',   name: 'NVIDIA RTX 4080 Super 16GB',    vendor: 'nvidia', vram: 16,  memoryBandwidth: 736,  pcie: '4.0', releaseYear: 2024, tdp: 320, recommended: false },
  { id: 'rtx4080-16',   name: 'NVIDIA RTX 4080 16GB',          vendor: 'nvidia', vram: 16,  memoryBandwidth: 717,  pcie: '4.0', releaseYear: 2022, tdp: 320, recommended: false },
  { id: 'rtx4070tis-16',name: 'NVIDIA RTX 4070 Ti Super 16GB', vendor: 'nvidia', vram: 16,  memoryBandwidth: 672,  pcie: '4.0', releaseYear: 2024, tdp: 285, recommended: false },
  { id: 'rtx4070ti-12', name: 'NVIDIA RTX 4070 Ti 12GB',       vendor: 'nvidia', vram: 12,  memoryBandwidth: 504,  pcie: '4.0', releaseYear: 2023, tdp: 285, recommended: false },
  { id: 'rtx4070s-12',  name: 'NVIDIA RTX 4070 Super 12GB',    vendor: 'nvidia', vram: 12,  memoryBandwidth: 504,  pcie: '4.0', releaseYear: 2024, tdp: 220, recommended: false },
  { id: 'rtx4070-12',   name: 'NVIDIA RTX 4070 12GB',          vendor: 'nvidia', vram: 12,  memoryBandwidth: 504,  pcie: '4.0', releaseYear: 2023, tdp: 200, recommended: false },
  { id: 'rtx4060ti-16', name: 'NVIDIA RTX 4060 Ti 16GB',       vendor: 'nvidia', vram: 16,  memoryBandwidth: 288,  pcie: '4.0', releaseYear: 2023, tdp: 165, recommended: false },
  { id: 'rtx4060ti-8',  name: 'NVIDIA RTX 4060 Ti 8GB',        vendor: 'nvidia', vram: 8,   memoryBandwidth: 288,  pcie: '4.0', releaseYear: 2023, tdp: 160, recommended: false },
  { id: 'rtx4060-8',    name: 'NVIDIA RTX 4060 8GB',           vendor: 'nvidia', vram: 8,   memoryBandwidth: 272,  pcie: '4.0', releaseYear: 2023, tdp: 115, recommended: false },
  { id: 'rtx3090ti-24', name: 'NVIDIA RTX 3090 Ti 24GB',    vendor: 'nvidia', vram: 24,  memoryBandwidth: 1008, pcie: '4.0', releaseYear: 2022, tdp: 450, recommended: false },
  { id: 'rtx3090-24',  name: 'NVIDIA RTX 3090 24GB',       vendor: 'nvidia', vram: 24,  memoryBandwidth: 936,  pcie: '4.0', releaseYear: 2020, tdp: 350, recommended: false },
  { id: 'rtx3080ti-12',name: 'NVIDIA RTX 3080 Ti 12GB',    vendor: 'nvidia', vram: 12,  memoryBandwidth: 912,  pcie: '4.0', releaseYear: 2021, tdp: 350, recommended: false },
  { id: 'rtx3080-12',  name: 'NVIDIA RTX 3080 12GB',       vendor: 'nvidia', vram: 12,  memoryBandwidth: 912,  pcie: '4.0', releaseYear: 2022, tdp: 350, recommended: false },
  { id: 'rtx3080-10',  name: 'NVIDIA RTX 3080 10GB',       vendor: 'nvidia', vram: 10,  memoryBandwidth: 760,  pcie: '4.0', releaseYear: 2020, tdp: 320, recommended: false },
  { id: 'rtx3070ti-8', name: 'NVIDIA RTX 3070 Ti 8GB',     vendor: 'nvidia', vram: 8,   memoryBandwidth: 608,  pcie: '4.0', releaseYear: 2021, tdp: 290, recommended: false },
  { id: 'rtx3070-8',   name: 'NVIDIA RTX 3070 8GB',        vendor: 'nvidia', vram: 8,   memoryBandwidth: 448,  pcie: '4.0', releaseYear: 2020, tdp: 220, recommended: false },
  { id: 'rtx3060ti-8', name: 'NVIDIA RTX 3060 Ti 8GB',     vendor: 'nvidia', vram: 8,   memoryBandwidth: 448,  pcie: '4.0', releaseYear: 2020, tdp: 200, recommended: false },
  { id: 'rtx3060-12',  name: 'NVIDIA RTX 3060 12GB',       vendor: 'nvidia', vram: 12,  memoryBandwidth: 360,  pcie: '4.0', releaseYear: 2021, tdp: 170, recommended: false },
  { id: 'rtx3060-12v2',name: 'NVIDIA RTX 3060 12GB (rev)', vendor: 'nvidia', vram: 12,  memoryBandwidth: 360,  pcie: '3.0', releaseYear: 2022, tdp: 170, recommended: false },
  { id: 'l40s-48',     name: 'NVIDIA L40S 48GB',           vendor: 'nvidia', vram: 48,  memoryBandwidth: 1812, pcie: '4.0', releaseYear: 2023, tdp: 300, recommended: false },
  { id: 'l4-24',       name: 'NVIDIA L4 24GB',             vendor: 'nvidia', vram: 24,  memoryBandwidth: 864,  pcie: '4.0', releaseYear: 2023, tdp: 72,  recommended: false },
  { id: 't4-16',       name: 'NVIDIA T4 16GB',             vendor: 'nvidia', vram: 16,  memoryBandwidth: 320,  pcie: '3.0', releaseYear: 2018, tdp: 70,  recommended: false },
  { id: 'v100-32',     name: 'NVIDIA V100 32GB',           vendor: 'nvidia', vram: 32,  memoryBandwidth: 900,  pcie: '3.0', releaseYear: 2017, tdp: 250, recommended: false },
  { id: 'v100-16',     name: 'NVIDIA V100 16GB',           vendor: 'nvidia', vram: 16,  memoryBandwidth: 900,  pcie: '3.0', releaseYear: 2019, tdp: 250, recommended: false },
  { id: 'v100-12',     name: 'NVIDIA V100 12GB',           vendor: 'nvidia', vram: 12,  memoryBandwidth: 900,  pcie: '3.0', releaseYear: 2017, tdp: 250, recommended: false },
  { id: 'mi300x-192',  name: 'AMD MI300X 192GB',           vendor: 'amd',    vram: 192, memoryBandwidth: 1965, pcie: '5.0', releaseYear: 2024, tdp: 940, recommended: true },
  { id: 'mi250x-128',  name: 'AMD MI250X 128GB',           vendor: 'amd',    vram: 128, memoryBandwidth: 1140, pcie: '4.0', releaseYear: 2022, tdp: 350, recommended: false },
  { id: 'mi250-64',    name: 'AMD MI250 64GB',             vendor: 'amd',    vram: 64,  memoryBandwidth: 1140, pcie: '4.0', releaseYear: 2022, tdp: 280, recommended: false },
  { id: 'mi100-32',    name: 'AMD MI100 32GB',             vendor: 'amd',    vram: 32,  memoryBandwidth: 1229, pcie: '4.0', releaseYear: 2019, tdp: 250, recommended: false },
  { id: 'rx7900xtx-24',name: 'AMD RX 7900 XTX 24GB',      vendor: 'amd',    vram: 24,  memoryBandwidth: 1228, pcie: '4.0', releaseYear: 2022, tdp: 355, recommended: false },
  { id: 'rx7900xt-20', name: 'AMD RX 7900 XT 20GB',       vendor: 'amd',    vram: 20,  memoryBandwidth: 935,  pcie: '4.0', releaseYear: 2022, tdp: 300, recommended: false },
  { id: 'rx7900gre-16',name: 'AMD RX 7900 GRE 16GB',      vendor: 'amd',    vram: 16,  memoryBandwidth: 869,  pcie: '4.0', releaseYear: 2023, tdp: 260, recommended: false },
  { id: 'rx7800xt-16', name: 'AMD RX 7800 XT 16GB',       vendor: 'amd',    vram: 16,  memoryBandwidth: 756,  pcie: '4.0', releaseYear: 2023, tdp: 263, recommended: false },
  { id: 'rx7700xt-12', name: 'AMD RX 7700 XT 12GB',       vendor: 'amd',    vram: 12,  memoryBandwidth: 575,  pcie: '4.0', releaseYear: 2023, tdp: 245, recommended: false },
  { id: 'rx6900xt-16', name: 'AMD RX 6900 XT 16GB',       vendor: 'amd',    vram: 16,  memoryBandwidth: 1024, pcie: '4.0', releaseYear: 2020, tdp: 300, recommended: false },
  { id: 'rx6800xt-16', name: 'AMD RX 6800 XT 16GB',       vendor: 'amd',    vram: 16,  memoryBandwidth: 840,  pcie: '4.0', releaseYear: 2020, tdp: 300, recommended: false },
  { id: 'm4-24',       name: 'Apple M4 Max 24GB',          vendor: 'apple',  vram: 24,  memoryBandwidth: 410,  pcie: 'N/A', releaseYear: 2024, tdp: 45,  recommended: false },
  { id: 'm4-32',       name: 'Apple M4 Max 32GB',          vendor: 'apple',  vram: 32,  memoryBandwidth: 410,  pcie: 'N/A', releaseYear: 2024, tdp: 45,  recommended: false },
  { id: 'm3-18',       name: 'Apple M3 Max 18GB',          vendor: 'apple',  vram: 18,  memoryBandwidth: 400,  pcie: 'N/A', releaseYear: 2023, tdp: 30,  recommended: false },
  { id: 'm3-36',       name: 'Apple M3 Max 36GB',          vendor: 'apple',  vram: 36,  memoryBandwidth: 400,  pcie: 'N/A', releaseYear: 2023, tdp: 30,  recommended: false },
  { id: 'm2-16',       name: 'Apple M2 Ultra 16GB',        vendor: 'apple',  vram: 16,  memoryBandwidth: 800,  pcie: 'N/A', releaseYear: 2023, tdp: 100, recommended: false },
  { id: 'm2-64',       name: 'Apple M2 Ultra 64GB',        vendor: 'apple',  vram: 64,  memoryBandwidth: 800,  pcie: 'N/A', releaseYear: 2023, tdp: 100, recommended: false },
  { id: 'm2-192',      name: 'Apple M2 Ultra 192GB',       vendor: 'apple',  vram: 192, memoryBandwidth: 800,  pcie: 'N/A', releaseYear: 2023, tdp: 100, recommended: false },
]

// KV cache per 1k tokens derived from: num_heads * head_dim * num_layers * 2 (K+V) * bytes_per_element / 1e9 * 1000
// Approximations scaled by quantization and parameter count.
// Headroom covers CUDA context, activations, framework overhead (~1–4 GB typical).
export const MODELS: ModelPreset[] = [
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

export function calculatePerformance(breakdown: VramBreakdown, gpuVram: number): PerformanceBucket {
  if (breakdown.fitStatus === 'no-fit') return 'infeasible'
  const ratio = breakdown.gpuUsed / gpuVram
  if (ratio <= 0.70) return 'realtime'
  if (ratio <= 0.85) return 'fast'
  if (ratio <= 0.95) return 'moderate'
  return 'slow'
}

export function getPerformanceColor(bucket: PerformanceBucket): string {
  switch (bucket) {
    case 'realtime':   return 'text-green-400'
    case 'fast':       return 'text-emerald-400'
    case 'moderate':   return 'text-yellow-400'
    case 'slow':       return 'text-orange-400'
    case 'infeasible': return 'text-red-400'
  }
}

export function getPartitionBarColor(bucket: PerformanceBucket): string {
  switch (bucket) {
    case 'realtime':   return 'bg-green-500'
    case 'fast':       return 'bg-emerald-500'
    case 'moderate':   return 'bg-yellow-500'
    case 'slow':       return 'bg-orange-500'
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
