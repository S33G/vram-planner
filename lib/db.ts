import { CURATED_MODELS } from './models.curated'
import generatedModelsJson from './models.generated.json'

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

// MODELS is composed from a hand-curated list (lib/models.curated.ts) plus an
// auto-generated list produced by `npm run sync:models` (scripts/sync-hf-models.ts).
// Curated entries win on `id` collisions so manual tuning is preserved.
const generatedModels = generatedModelsJson as ModelPreset[]

function mergeModels(curated: ModelPreset[], generated: ModelPreset[]): ModelPreset[] {
  const seen = new Set(curated.map(m => m.id))
  const extras = generated.filter(m => !seen.has(m.id))
  return [...curated, ...extras]
}

export const MODELS: ModelPreset[] = mergeModels(CURATED_MODELS, generatedModels)


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
