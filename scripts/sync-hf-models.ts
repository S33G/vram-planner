/**
 * scripts/sync-hf-models.ts
 *
 * Pulls a curated allow-list of Hugging Face repos and derives ModelPreset rows
 * for the static VRAM planner. Output is written to lib/models.generated.json.
 *
 * Usage:
 *   npm run sync:models
 *   HF_TOKEN=hf_xxx npm run sync:models   # raises HF API rate limits
 *
 * Curated entries in lib/models.curated.ts win on id collisions; this script
 * never touches that file. Run it locally or via the scheduled GitHub Action
 * (.github/workflows/sync-models.yml) which opens a PR with the diff.
 */
import fs from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

// Avoid importing from lib/db.ts (which imports JSON not yet present in fresh
// checkouts). Re-declare the minimal shape we emit.
type Quant = 'fp16' | 'bf16' | 'fp8' | 'int8' | 'int4' | 'nf4' | 'mixed'

interface ModelPreset {
  id: string
  name: string
  family: string
  parameters: number
  quantization: Quant
  weightsGb: number
  kvCachePer1kTokens: number
  headroomGb: number
  contextLength: number
  recommendedGpuVram: number
  minGpuVram: number
  tags: string[]
}

interface SourceFamily {
  family: string
  repos: string[]
  quantizations?: Quant[]
}

interface SourcesFile {
  families: SourceFamily[]
}

interface HfSafetensorsBlock {
  parameters?: Record<string, number>
  total?: number
}

interface HfModelInfo {
  id: string
  modelId?: string
  pipeline_tag?: string
  tags?: string[]
  config?: {
    hidden_size?: number
    num_hidden_layers?: number
    num_attention_heads?: number
    num_key_value_heads?: number
    max_position_embeddings?: number
    torch_dtype?: string
    architectures?: string[]
    head_dim?: number
  }
  safetensors?: HfSafetensorsBlock
}

const STANDARD_VRAM_TIERS = [4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512] as const
const BYTES_PER_PARAM: Record<Quant, number> = {
  fp16: 2,
  bf16: 2,
  fp8: 1,
  int8: 1,
  int4: 0.5,
  nf4: 0.5,
  mixed: 1.5,
}
const KV_BYTES_PER_QUANT: Record<Quant, number> = {
  fp16: 2,
  bf16: 2,
  fp8: 1,
  int8: 1,
  // KV cache typically stays at fp16/bf16 even when weights are quantized.
  int4: 2,
  nf4: 2,
  mixed: 2,
}

/** Round up a GB number to the next standard VRAM tier (defaults to ceil if exceeds list). */
export function roundToVramTier(gb: number): number {
  for (const t of STANDARD_VRAM_TIERS) if (t >= gb) return t
  return Math.ceil(gb)
}

/** Tiered headroom in GB based on parameter count (in billions). */
export function headroomForParams(paramsB: number): number {
  if (paramsB <= 3) return 1
  if (paramsB <= 14) return 2
  if (paramsB <= 34) return 3
  if (paramsB <= 80) return 4
  return 5
}

/** Compute weight footprint in GB for a (re)quantized variant. */
export function computeWeightsGb(paramsB: number, quant: Quant): number {
  const bytesPerParam = BYTES_PER_PARAM[quant]
  const overhead = 1.05 // small framework overhead beyond raw bytes
  const gb = paramsB * 1e9 * bytesPerParam * overhead / 1e9
  return Number(gb.toFixed(2))
}

/**
 * KV cache per 1k tokens (single batch slot), in GB.
 *   bytes = 2 (K+V) * num_layers * num_kv_heads * head_dim * bytes_per_kv
 * Honors GQA via num_key_value_heads.
 */
export function computeKvCachePer1kTokens(args: {
  numLayers: number
  numKvHeads: number
  headDim: number
  quant: Quant
}): number {
  const { numLayers, numKvHeads, headDim, quant } = args
  const bytesKv = KV_BYTES_PER_QUANT[quant]
  const bytesPerToken = 2 * numLayers * numKvHeads * headDim * bytesKv
  const gbPer1k = (bytesPerToken * 1024) / 1e9
  return Number(gbPer1k.toFixed(3))
}

/** Parse a parameter count out of a repo id (e.g. "Llama-3.1-70B-Instruct" -> 70). */
export function parseParamsFromName(repoId: string): number | null {
  const m = repoId.match(/(?<![A-Za-z])(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])/)
  if (!m) return null
  const n = parseFloat(m[1])
  return Number.isFinite(n) && n > 0 ? n : null
}

/** Pick the dominant quantization from safetensors dtype block (fall back to torch_dtype). */
export function detectBaseQuant(info: HfModelInfo): Quant {
  const params = info.safetensors?.parameters
  if (params && Object.keys(params).length) {
    let best: [string, number] = ['', 0]
    for (const [dtype, count] of Object.entries(params)) {
      if (count > best[1]) best = [dtype, count]
    }
    const dtype = best[0].toLowerCase()
    if (dtype.includes('bf16')) return 'bf16'
    if (dtype.includes('fp16') || dtype.includes('f16')) return 'fp16'
    if (dtype.includes('fp8') || dtype.includes('f8')) return 'fp8'
    if (dtype.includes('int8') || dtype === 'i8') return 'int8'
    if (dtype.includes('int4') || dtype === 'i4' || dtype.includes('nf4')) return 'int4'
  }
  const td = (info.config?.torch_dtype || '').toLowerCase()
  if (td.includes('bfloat16')) return 'bf16'
  if (td.includes('float16') || td === 'fp16') return 'fp16'
  if (td.includes('float8') || td.includes('fp8')) return 'fp8'
  return 'bf16'
}

/** Total parameters in billions, derived from safetensors total or repo name. */
export function detectParameters(info: HfModelInfo): number | null {
  const total = info.safetensors?.total
  if (typeof total === 'number' && total > 0) return Number((total / 1e9).toFixed(2))
  return parseParamsFromName(info.id || info.modelId || '')
}

/** Best-effort tag derivation. */
export function deriveTags(info: HfModelInfo, paramsB: number): string[] {
  const out = new Set<string>()
  const id = (info.id || '').toLowerCase()
  const tags = (info.tags || []).map(t => t.toLowerCase())

  if (id.includes('coder') || id.includes('code')) out.add('code')
  if (id.includes('instruct') || id.includes('chat') || id.includes('-it')) out.add('chat')
  if (tags.some(t => t.includes('multilingual'))) out.add('multilingual')
  if (paramsB <= 8) out.add('fast')
  if (paramsB <= 3) out.add('edge')
  if ((info.config?.max_position_embeddings ?? 0) >= 65536) out.add('long-context')
  if (out.size === 0) out.add('chat')
  return [...out]
}

/**
 * Thrown when HF returns 401/403, which always means missing/invalid HF_TOKEN
 * or a license that has not been accepted on the Hugging Face account that
 * minted the token. These are configuration errors, not transient failures,
 * so we surface them as fatal up the call stack.
 */
class HfAuthError extends Error {
  status: number
  url: string
  constructor(status: number, statusText: string, url: string) {
    super(`HF API ${status} ${statusText} for ${url}`)
    this.name = 'HfAuthError'
    this.status = status
    this.url = url
  }
}

/** Fetch JSON from HF with optional bearer token. */
async function hfGet<T>(url: string): Promise<T> {
  const headers: Record<string, string> = { Accept: 'application/json' }
  if (process.env.HF_TOKEN) headers.Authorization = `Bearer ${process.env.HF_TOKEN}`
  const res = await fetch(url, { headers, redirect: 'follow' })
  if (!res.ok) {
    if (res.status === 401 || res.status === 403) {
      throw new HfAuthError(res.status, res.statusText, url)
    }
    throw new Error(`HF API ${res.status} ${res.statusText} for ${url}`)
  }
  return res.json() as Promise<T>
}

/** Fetch the raw transformers config.json from a repo's main branch. */
async function fetchRepoConfig(repo: string): Promise<HfModelInfo['config']> {
  try {
    return await hfGet<HfModelInfo['config']>(`https://huggingface.co/${repo}/resolve/main/config.json`)
  } catch (e) {
    if (e instanceof HfAuthError) throw e
    console.warn(`[warn] ${repo}: no config.json (${(e as Error).message})`)
    return undefined
  }
}

function slugify(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')
}

function friendlyName(repoId: string): string {
  // Preserve original casing of the repo tail; just turn separators into spaces.
  const tail = repoId.split('/').pop() || repoId
  return tail.replace(/[-_]+/g, ' ').replace(/\s+/g, ' ').trim()
}

function quantSuffix(q: Quant): string {
  switch (q) {
    case 'bf16': return ''
    case 'fp16': return ' (FP16)'
    case 'fp8':  return ' (FP8)'
    case 'int8': return ' (INT8)'
    case 'int4': return ' (Q4)'
    case 'nf4':  return ' (NF4)'
    case 'mixed':return ' (Mixed)'
  }
}

/** Build one or more ModelPreset rows for a given HF repo. */
export function buildPresets(info: HfModelInfo, family: string, requestedQuants?: Quant[]): ModelPreset[] {
  const paramsB = detectParameters(info)
  if (!paramsB || paramsB <= 0) {
    console.warn(`[skip] ${info.id}: cannot determine parameter count`)
    return []
  }
  const cfg = info.config || {}
  const numLayers = cfg.num_hidden_layers ?? 0
  const numHeads = cfg.num_attention_heads ?? 0
  const numKvHeads = cfg.num_key_value_heads ?? numHeads
  const hidden = cfg.hidden_size ?? 0
  const headDim = cfg.head_dim ?? (numHeads > 0 ? hidden / numHeads : 0)
  const ctx = cfg.max_position_embeddings ?? 8192

  if (!numLayers || !numKvHeads || !headDim) {
    console.warn(`[skip] ${info.id}: missing transformer config (layers=${numLayers}, kv=${numKvHeads}, head=${headDim})`)
    return []
  }

  const baseQuant = detectBaseQuant(info)
  const quants: Quant[] = requestedQuants && requestedQuants.length ? requestedQuants : [baseQuant]
  const tags = deriveTags(info, paramsB)
  const headroom = headroomForParams(paramsB)

  const rows: ModelPreset[] = []
  for (const q of quants) {
    const weightsGb = computeWeightsGb(paramsB, q)
    const kvPer1k = computeKvCachePer1kTokens({ numLayers, numKvHeads, headDim, quant: q })
    const ctxKv = (Math.min(ctx, 8192) / 1024) * kvPer1k
    const required = weightsGb + ctxKv + headroom
    const recommended = roundToVramTier(required)
    const min = roundToVramTier(weightsGb + headroom)

    rows.push({
      id: `${slugify(info.id)}${q === 'bf16' ? '' : '-' + q}`,
      name: `${friendlyName(info.id)}${quantSuffix(q)}`,
      family,
      parameters: paramsB,
      quantization: q,
      weightsGb,
      kvCachePer1kTokens: kvPer1k,
      headroomGb: headroom,
      contextLength: ctx,
      recommendedGpuVram: recommended,
      minGpuVram: min,
      tags,
    })
  }
  return rows
}

function validatePreset(p: ModelPreset): string | null {
  if (p.parameters <= 0) return 'parameters <= 0'
  if (p.weightsGb <= 0) return 'weightsGb <= 0'
  if (p.kvCachePer1kTokens < 0) return 'kvCachePer1kTokens < 0'
  if (p.contextLength <= 0) return 'contextLength <= 0'
  if (p.minGpuVram <= 0) return 'minGpuVram <= 0'
  if (p.recommendedGpuVram < p.minGpuVram) return 'recommendedGpuVram < minGpuVram'
  return null
}

async function main() {
  const here = path.dirname(fileURLToPath(import.meta.url))
  const root = path.resolve(here, '..')
  const sourcesPath = path.join(root, 'scripts', 'hf-sources.json')
  const outPath = path.join(root, 'lib', 'models.generated.json')

  const sources = JSON.parse(await fs.readFile(sourcesPath, 'utf8')) as SourcesFile
  const all: ModelPreset[] = []
  let okRepos = 0
  let failedRepos = 0
  let gatedRepos = 0

  for (const fam of sources.families) {
    for (const repo of fam.repos) {
      try {
        const info = await hfGet<HfModelInfo>(`https://huggingface.co/api/models/${repo}?expand=safetensors&expand=tags`)
        info.id = info.id || info.modelId || repo
        info.config = await fetchRepoConfig(repo)
        const rows = buildPresets(info, fam.family, fam.quantizations)
        for (const r of rows) {
          const err = validatePreset(r)
          if (err) {
            console.warn(`[skip] ${r.id}: ${err}`)
            continue
          }
          all.push(r)
        }
        okRepos++
        console.log(`[ok]  ${repo} -> ${rows.length} preset(s)`)
      } catch (e) {
        if (e instanceof HfAuthError) {
          const tokenSet = Boolean(process.env.HF_TOKEN)
          const reason = tokenSet
            ? `HF_TOKEN is set but lacks access (license not accepted by token owner). Visit https://huggingface.co/${repo} and accept the terms.`
            : `HF_TOKEN is not set. Create one at https://huggingface.co/settings/tokens, accept the license at https://huggingface.co/${repo}, and add it as the HF_TOKEN repo secret.`
          console.warn(`[gated] ${repo}: HF ${e.status} \u2014 skipping. ${reason}`)
          gatedRepos++
          continue
        }
        failedRepos++
        console.error(`[fail] ${repo}: ${(e as Error).message}`)
      }
    }
  }

  // Stable sort for clean diffs.
  all.sort((a, b) => a.id.localeCompare(b.id))

  await fs.writeFile(outPath, JSON.stringify(all, null, 2) + '\n', 'utf8')
  console.log(`\nWrote ${all.length} entries to ${path.relative(root, outPath)} (repos ok=${okRepos}, gated=${gatedRepos}, failed=${failedRepos}).`)
}

// Only run main when invoked directly (not when imported by tests).
const invokedDirectly = (() => {
  try {
    return process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])
  } catch {
    return false
  }
})()

if (invokedDirectly) {
  main().catch(err => {
    console.error(err)
    process.exit(1)
  })
}
