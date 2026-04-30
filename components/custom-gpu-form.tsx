"use client"

import { useState } from "react"
import type { GpuSpec } from "@/lib/db"

interface Props {
  onAdd: (gpu: GpuSpec) => void
  onCancel: () => void
}

export function CustomGpuForm({ onAdd, onCancel }: Props) {
  const [name, setName] = useState("")
  const [vendor, setVendor] = useState<GpuSpec["vendor"]>("nvidia")
  const [vram, setVram] = useState("")
  const [bandwidth, setBandwidth] = useState("")
  const [tdp, setTdp] = useState("")

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const vramNum = parseFloat(vram)
    if (!name.trim() || isNaN(vramNum) || vramNum <= 0) return
    const gpu: GpuSpec = {
      id: `custom-${Date.now()}`,
      name: name.trim(),
      vendor,
      vram: vramNum,
      memoryBandwidth: parseFloat(bandwidth) || 0,
      pcie: "N/A",
      releaseYear: new Date().getFullYear(),
      tdp: parseFloat(tdp) || 0,
      recommended: false,
    }
    onAdd(gpu)
  }

  return (
    <form onSubmit={handleSubmit} className="mt-3 flex flex-col gap-3">
      <div className="flex flex-col gap-1">
        <label className="text-xs text-zinc-400">Name</label>
        <input
          required
          className="rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-white outline-none placeholder:text-zinc-600 focus:border-cyan-400"
          placeholder="e.g. My RTX 5090"
          value={name}
          onChange={e => setName(e.target.value)}
        />
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="flex flex-col gap-1">
          <label className="text-xs text-zinc-400">Vendor</label>
          <select
            className="rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-white outline-none focus:border-cyan-400"
            value={vendor}
            onChange={e => setVendor(e.target.value as GpuSpec["vendor"])}
          >
            <option value="nvidia">NVIDIA</option>
            <option value="amd">AMD</option>
            <option value="apple">Apple</option>
          </select>
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-zinc-400">VRAM (GB)</label>
          <input
            required
            type="number"
            min="1"
            step="any"
            className="rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-white outline-none placeholder:text-zinc-600 focus:border-cyan-400"
            placeholder="24"
            value={vram}
            onChange={e => setVram(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-zinc-400">Bandwidth (GB/s)</label>
          <input
            type="number"
            min="0"
            step="any"
            className="rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-white outline-none placeholder:text-zinc-600 focus:border-cyan-400"
            placeholder="Optional"
            value={bandwidth}
            onChange={e => setBandwidth(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-zinc-400">TDP (W)</label>
          <input
            type="number"
            min="0"
            step="any"
            className="rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-white outline-none placeholder:text-zinc-600 focus:border-cyan-400"
            placeholder="Optional"
            value={tdp}
            onChange={e => setTdp(e.target.value)}
          />
        </div>
      </div>
      <div className="flex gap-2 pt-1">
        <button
          type="submit"
          className="flex-1 rounded-xl bg-cyan-500 px-3 py-2 text-sm font-semibold text-zinc-950 transition hover:bg-cyan-400"
        >
          Add GPU
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
