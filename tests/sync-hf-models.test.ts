import { describe, it } from 'node:test'
import assert from 'node:assert/strict'

import {
  roundToVramTier,
  headroomForParams,
  computeWeightsGb,
  computeKvCachePer1kTokens,
  parseParamsFromName,
  detectBaseQuant,
  detectParameters,
  buildPresets,
} from '../scripts/sync-hf-models'

describe('roundToVramTier', () => {
  it('returns the next standard tier', () => {
    assert.equal(roundToVramTier(7), 8)
    assert.equal(roundToVramTier(8), 8)
    assert.equal(roundToVramTier(13), 16)
    assert.equal(roundToVramTier(141), 160)
  })
  it('falls back to ceil for above-tier values', () => {
    assert.equal(roundToVramTier(700), 700)
  })
})

describe('headroomForParams', () => {
  it('tiers headroom by parameter count', () => {
    assert.equal(headroomForParams(1), 1)
    assert.equal(headroomForParams(7), 2)
    assert.equal(headroomForParams(30), 3)
    assert.equal(headroomForParams(70), 4)
    assert.equal(headroomForParams(123), 5)
  })
})

describe('computeWeightsGb', () => {
  it('sizes bf16 at ~2 bytes/param', () => {
    // 7B * 2 bytes * 1.05 overhead = 14.7 GB
    assert.ok(Math.abs(computeWeightsGb(7, 'bf16') - 14.7) < 0.05)
  })
  it('sizes int4 at ~0.5 bytes/param', () => {
    // 70B * 0.5 * 1.05 = 36.75
    assert.ok(Math.abs(computeWeightsGb(70, 'int4') - 36.75) < 0.05)
  })
  it('sizes fp8 at ~1 byte/param', () => {
    assert.ok(Math.abs(computeWeightsGb(8, 'fp8') - 8.4) < 0.05)
  })
})

describe('computeKvCachePer1kTokens', () => {
  it('matches the known formula', () => {
    // 2 * 32 layers * 8 kv heads * 128 head_dim * 2 bytes = 131072 bytes/token
    // * 1024 tokens / 1e9 = 0.134 GB per 1k
    const v = computeKvCachePer1kTokens({ numLayers: 32, numKvHeads: 8, headDim: 128, quant: 'bf16' })
    assert.ok(Math.abs(v - 0.134) < 0.005)
  })
  it('honors GQA via num_key_value_heads', () => {
    const mha = computeKvCachePer1kTokens({ numLayers: 32, numKvHeads: 32, headDim: 128, quant: 'bf16' })
    const gqa = computeKvCachePer1kTokens({ numLayers: 32, numKvHeads: 8, headDim: 128, quant: 'bf16' })
    assert.ok(mha > gqa * 3.5)
  })
  it('keeps KV at fp16 sizing even when weights are int4', () => {
    const a = computeKvCachePer1kTokens({ numLayers: 32, numKvHeads: 8, headDim: 128, quant: 'bf16' })
    const b = computeKvCachePer1kTokens({ numLayers: 32, numKvHeads: 8, headDim: 128, quant: 'int4' })
    assert.equal(a, b)
  })
})

describe('parseParamsFromName', () => {
  it('extracts B suffix', () => {
    assert.equal(parseParamsFromName('Llama-3.1-70B-Instruct'), 70)
    assert.equal(parseParamsFromName('Qwen2.5-7B'), 7)
    assert.equal(parseParamsFromName('phi-3.5-mini'), null)
  })
  it('handles decimal params', () => {
    assert.equal(parseParamsFromName('DeepSeek-Coder-6.7b-instruct'), 6.7)
  })
})

describe('detectBaseQuant', () => {
  it('reads dtype from safetensors block', () => {
    assert.equal(detectBaseQuant({ id: 'x', safetensors: { parameters: { BF16: 1e9 } } }), 'bf16')
    assert.equal(detectBaseQuant({ id: 'x', safetensors: { parameters: { F16: 1e9 } } }), 'fp16')
  })
  it('falls back to torch_dtype', () => {
    assert.equal(detectBaseQuant({ id: 'x', config: { torch_dtype: 'bfloat16' } }), 'bf16')
  })
  it('defaults to bf16 when unknown', () => {
    assert.equal(detectBaseQuant({ id: 'x' }), 'bf16')
  })
})

describe('detectParameters', () => {
  it('prefers safetensors total', () => {
    assert.equal(detectParameters({ id: 'whatever', safetensors: { total: 7_615_616_512 } }), 7.62)
  })
  it('falls back to repo name', () => {
    assert.equal(detectParameters({ id: 'org/Foo-13B' }), 13)
  })
  it('returns null when nothing works', () => {
    assert.equal(detectParameters({ id: 'org/Foo' }), null)
  })
})

describe('buildPresets', () => {
  const baseInfo = {
    id: 'Qwen/Qwen2.5-7B-Instruct',
    safetensors: { total: 7_615_616_512, parameters: { BF16: 7_615_616_512 } },
    config: {
      hidden_size: 3584,
      num_hidden_layers: 28,
      num_attention_heads: 28,
      num_key_value_heads: 4,
      max_position_embeddings: 32768,
      torch_dtype: 'bfloat16',
    },
  }

  it('produces one row per requested quant', () => {
    const rows = buildPresets(baseInfo, 'Qwen', ['bf16', 'fp8', 'int4'])
    assert.equal(rows.length, 3)
    const ids = rows.map(r => r.id)
    assert.ok(ids.some(i => i.endsWith('fp8')))
    assert.ok(ids.some(i => i.endsWith('int4')))
  })

  it('skips entries without transformer config', () => {
    const rows = buildPresets({ id: 'org/Foo-7B' }, 'Foo')
    assert.equal(rows.length, 0)
  })

  it('skips entries without parameter count', () => {
    const rows = buildPresets({ id: 'org/Foo', config: baseInfo.config }, 'Foo')
    assert.equal(rows.length, 0)
  })

  it('produces sane min/recommended VRAM ordering', () => {
    const rows = buildPresets(baseInfo, 'Qwen', ['bf16'])
    const r = rows[0]
    assert.ok(r.recommendedGpuVram >= r.minGpuVram)
    assert.ok(r.weightsGb > 0)
    assert.ok(r.kvCachePer1kTokens > 0)
  })
})
