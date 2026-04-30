"use client"

import { GPUs, MODELS } from "@/lib/db"
import type { GpuSpec, ModelPreset } from "@/lib/db"

interface LoadedModel extends ModelPreset {
  instanceId: string
  vramAllocated: number
}

interface SidebarProps {
  gpu: GpuSpec | null
  setGpu: (gpu: GpuSpec | null) => void
  filter: string
  setFilter: (value: string) => void
  familyFilter: string
  setFamilyFilter: (value: string) => void
  onAddModel: (model: ModelPreset) => void
  loadedModels: LoadedModel[]
  removeModel: (instanceId: string) => void
  totalUsedVram: number
  totalFreeVram: number
}

const families = Array.from(new Set(MODELS.map((model) => model.family))).sort()

export function Sidebar({
  gpu,
  setGpu,
  filter,
  setFilter,
  familyFilter,
  setFamilyFilter,
  onAddModel,
  loadedModels,
  removeModel,
  totalUsedVram,
  totalFreeVram,
}: SidebarProps) {
  const filteredModels = MODELS.filter((model) => {
    const matchesText = `${model.name} ${model.family} ${model.tags.join(" ")}`
      .toLowerCase()
      .includes(filter.toLowerCase())
    const matchesFamily = familyFilter === "all" || model.family === familyFilter
    return matchesText && matchesFamily
  })

  return (
    <aside className="flex w-full flex-col gap-5 overflow-y-auto border-b border-zinc-800 bg-zinc-950/95 p-5 lg:h-dvh lg:w-[420px] lg:border-b-0 lg:border-r">
      <section>
        <p className="text-xs font-semibold uppercase tracking-[0.24em] text-cyan-300">VRAM Planner</p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white">Plan model placement before loading weights.</h1>
        <p className="mt-3 text-sm leading-6 text-zinc-400">
          Pick a target GPU, add models, and compare the simulated plan with live NVIDIA GPU memory when available.
        </p>
      </section>

      <section className="rounded-2xl border border-zinc-800 bg-zinc-900/70 p-4">
        <label className="text-sm font-medium text-zinc-200" htmlFor="gpu-select">Target GPU</label>
        <select
          id="gpu-select"
          className="mt-2 w-full rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-white outline-none focus:border-cyan-400"
          value={gpu?.id ?? ""}
          onChange={(event) => setGpu(GPUs.find((item) => item.id === event.target.value) ?? null)}
        >
          <option value="">Select a GPU</option>
          {GPUs.map((item) => (
            <option key={item.id} value={item.id}>{item.name}</option>
          ))}
        </select>
        {gpu ? (
          <div className="mt-4 grid grid-cols-3 gap-2 text-center text-xs">
            <Metric label="VRAM" value={`${gpu.vram} GB`} />
            <Metric label="Used" value={`${totalUsedVram.toFixed(1)} GB`} />
            <Metric label="Free" value={`${totalFreeVram.toFixed(1)} GB`} tone={totalFreeVram < 0 ? "bad" : "good"} />
          </div>
        ) : null}
      </section>

      <section className="flex min-h-0 flex-1 flex-col rounded-2xl border border-zinc-800 bg-zinc-900/70 p-4">
        <div className="flex items-center justify-between gap-3">
          <h2 className="text-sm font-semibold text-zinc-100">Model Catalog</h2>
          <span className="text-xs text-zinc-500">{filteredModels.length} matches</span>
        </div>
        <input
          className="mt-3 rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-white outline-none placeholder:text-zinc-600 focus:border-cyan-400"
          placeholder="Search model, family, tag"
          value={filter}
          onChange={(event) => setFilter(event.target.value)}
        />
        <select
          className="mt-2 rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-white outline-none focus:border-cyan-400"
          value={familyFilter}
          onChange={(event) => setFamilyFilter(event.target.value)}
        >
          <option value="all">All families</option>
          {families.map((family) => (
            <option key={family} value={family}>{family}</option>
          ))}
        </select>
        <div className="mt-3 flex flex-col gap-2 overflow-y-auto pr-1">
          {filteredModels.map((model) => (
            <button
              key={model.id}
              type="button"
              disabled={!gpu}
              onClick={() => onAddModel(model)}
              className="rounded-xl border border-zinc-800 bg-zinc-950 p-3 text-left transition hover:border-cyan-500 disabled:cursor-not-allowed disabled:opacity-45"
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-sm font-medium text-white">{model.name}</p>
                  <p className="mt-1 text-xs text-zinc-500">{model.family} / {model.quantization} / {model.contextLength.toLocaleString()} ctx</p>
                </div>
                <span className="rounded-full bg-cyan-400/10 px-2 py-1 text-xs font-semibold text-cyan-200">{model.vramRequired} GB</span>
              </div>
            </button>
          ))}
        </div>
      </section>

      <section className="rounded-2xl border border-zinc-800 bg-zinc-900/70 p-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-zinc-100">Planned Loads</h2>
          <span className="text-xs text-zinc-500">{loadedModels.length}</span>
        </div>
        <div className="mt-3 space-y-2">
          {loadedModels.length === 0 ? (
            <p className="text-sm text-zinc-500">Select a GPU, then add models from the catalog.</p>
          ) : loadedModels.map((model) => (
            <div key={model.instanceId} className="flex items-center justify-between gap-3 rounded-xl bg-zinc-950 px-3 py-2">
              <div>
                <p className="text-sm text-white">{model.name}</p>
                <p className="text-xs text-zinc-500">{model.vramAllocated.toFixed(1)} GB allocated</p>
              </div>
              <button
                type="button"
                className="rounded-lg border border-zinc-700 px-2 py-1 text-xs text-zinc-300 hover:border-red-400 hover:text-red-200"
                onClick={() => removeModel(model.instanceId)}
              >
                Remove
              </button>
            </div>
          ))}
        </div>
      </section>
    </aside>
  )
}

function Metric({ label, value, tone }: { label: string; value: string; tone?: "good" | "bad" }) {
  const toneClass = tone === "bad" ? "text-red-300" : tone === "good" ? "text-emerald-300" : "text-white"
  return (
    <div className="rounded-xl bg-zinc-950 px-2 py-3">
      <p className="text-[10px] uppercase tracking-widest text-zinc-500">{label}</p>
      <p className={`mt-1 font-semibold ${toneClass}`}>{value}</p>
    </div>
  )
}
