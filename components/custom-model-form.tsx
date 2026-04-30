"use client"

import { useState } from "react"
import type { ModelPreset, QuantizationType } from "@/lib/db"

const QUANT_OPTIONS: QuantizationType[] = ["fp16", "bf16", "fp8", "int8", "int4", "nf4", "mixed"]

interface Props {
  onAdd: (model: ModelPreset) => void
  onCancel: () => void
}

export function CustomModelForm({ onAdd, onCancel }: Props) {
  const [name, setName]                     = useState("")
  const [family, setFamily]                 = useState("")
  const [parameters, setParameters]         = useState("")
  const [quantization, setQuantization]     = useState<QuantizationType>("bf16")
  const [weightsGb, setWeightsGb]           = useState("")
  const [kvPer1k, setKvPer1k]               = useState("")
  const [headroomGb, setHeadroomGb]         = useState("2")
  const [contextLength, setContextLength]   = useState("")

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const weights = parseFloat(weightsGb)
    const params  = parseFloat(parameters)
    if (!name.trim() || isNaN(weights) || weights <= 0 || isNaN(params)) return
    const model: ModelPreset = {
      id:                   `custom-${Date.now()}`,
      name:                 name.trim(),
      family:               family.trim() || "Custom",
      parameters:           params,
      quantization,
      weightsGb:            weights,
      kvCachePer1kTokens:   parseFloat(kvPer1k) || 0,
      headroomGb:           parseFloat(headroomGb) || 1,
      contextLength:        parseInt(contextLength) || 0,
      recommendedGpuVram:   weights + (parseFloat(headroomGb) || 1),
      minGpuVram:           weights,
      tags:                 ["custom"],
    }
    onAdd(model)
  }

  return (
    <form onSubmit={handleSubmit} className="mt-3 flex flex-col gap-3">
      <div className="flex flex-col gap-1">
        <label className="text-xs text-zinc-400">Name</label>
        <input
          required
          className="rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-white outline-none placeholder:text-zinc-600 focus:border-cyan-400"
          placeholder="e.g. My Fine-tune 7B"
          value={name}
          onChange={e => setName(e.target.value)}
        />
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="flex flex-col gap-1">
          <label className="text-xs text-zinc-400">Family</label>
          <input
            className="rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-white outline-none placeholder:text-zinc-600 focus:border-cyan-400"
            placeholder="Optional"
            value={family}
            onChange={e => setFamily(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-zinc-400">Params (B)</label>
          <input
            required
            type="number"
            min="0"
            step="any"
            className="rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-white outline-none placeholder:text-zinc-600 focus:border-cyan-400"
            placeholder="7"
            value={parameters}
            onChange={e => setParameters(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-zinc-400">Quantization</label>
          <select
            className="rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-white outline-none focus:border-cyan-400"
            value={quantization}
            onChange={e => setQuantization(e.target.value as QuantizationType)}
          >
            {QUANT_OPTIONS.map(q => <option key={q} value={q}>{q}</option>)}
          </select>
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-zinc-400">Weights (GB)</label>
          <input
            required
            type="number"
            min="0.1"
            step="any"
            className="rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-white outline-none placeholder:text-zinc-600 focus:border-cyan-400"
            placeholder="14"
            value={weightsGb}
            onChange={e => setWeightsGb(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-zinc-400">KV cache / 1k tokens (GB)</label>
          <input
            type="number"
            min="0"
            step="0.01"
            className="rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-white outline-none placeholder:text-zinc-600 focus:border-cyan-400"
            placeholder="0.06"
            value={kvPer1k}
            onChange={e => setKvPer1k(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-zinc-400">Headroom (GB)</label>
          <input
            type="number"
            min="0"
            step="0.5"
            className="rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-white outline-none placeholder:text-zinc-600 focus:border-cyan-400"
            placeholder="2"
            value={headroomGb}
            onChange={e => setHeadroomGb(e.target.value)}
          />
        </div>
        <div className="col-span-2 flex flex-col gap-1">
          <label className="text-xs text-zinc-400">Context length (tokens)</label>
          <input
            type="number"
            min="0"
            step="1024"
            className="rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-white outline-none placeholder:text-zinc-600 focus:border-cyan-400"
            placeholder="e.g. 32768"
            value={contextLength}
            onChange={e => setContextLength(e.target.value)}
          />
        </div>
      </div>
      <div className="flex gap-2 pt-1">
        <button
          type="submit"
          className="flex-1 rounded-xl bg-cyan-500 px-3 py-2 text-sm font-semibold text-zinc-950 transition hover:bg-cyan-400"
        >
          Add model
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="rounded-xl border border-zinc-700 px-3 py-2 text-sm text-zinc-400 transition hover:border-zinc-500 hover:text-white"
        >
          Cancel
        </button>
      </div>
    </form>
  )
}
