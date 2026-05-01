"use client"

import { useState } from "react"
import { GPUs, MODELS, calculateVramBreakdown } from "@/lib/db"
import type { GpuSpec, ModelPreset, SpillPolicy } from "@/lib/db"
import { useLocalStorage } from "@/lib/use-local-storage"
import { CustomGpuForm } from "@/components/custom-gpu-form"
import { CustomModelForm } from "@/components/custom-model-form"

interface LoadedModel extends ModelPreset {
  instanceId: string
  slots: number
  contextLength: number
}

interface SidebarProps {
  gpu: GpuSpec | null
  setGpu: (gpu: GpuSpec | null) => void
  allGpus: GpuSpec[]
  onAddCustomGpu: (gpu: GpuSpec) => void
  onRemoveCustomGpu: (id: string) => void
  filter: string
  setFilter: (value: string) => void
  familyFilter: string
  setFamilyFilter: (value: string) => void
  onAddModel: (model: ModelPreset) => void
  loadedModels: LoadedModel[]
  removeModel: (instanceId: string) => void
  totalUsedVram: number
  totalFreeVram: number
  systemRamGb: number
  setSystemRamGb: (v: number) => void
  spillPolicy: SpillPolicy
  setSpillPolicy: (v: SpillPolicy) => void
}

export function Sidebar({
  gpu,
  setGpu,
  allGpus,
  onAddCustomGpu,
  onRemoveCustomGpu,
  filter,
  setFilter,
  familyFilter,
  setFamilyFilter,
  onAddModel,
  loadedModels,
  removeModel,
  totalUsedVram,
  totalFreeVram,
  systemRamGb,
  setSystemRamGb,
  spillPolicy,
  setSpillPolicy,
}: SidebarProps) {
  const [customModels, setCustomModels, modelsHydrated] = useLocalStorage<ModelPreset[]>("custom-models", [])
  const [showGpuForm, setShowGpuForm] = useState(false)
  const [showModelForm, setShowModelForm] = useState(false)
  const [hideNoFit, setHideNoFit] = useState(true)

  const allModels = [...MODELS, ...(modelsHydrated ? customModels : [])]
  const customGpus = allGpus.filter(g => !GPUs.some(cg => cg.id === g.id))

  const families = Array.from(new Set(allModels.map(m => m.family))).sort()

  const filteredModels = allModels
    .filter((model) => {
      const matchesText = `${model.name} ${model.family} ${model.tags.join(" ")}`
        .toLowerCase()
        .includes(filter.toLowerCase())
      const matchesFamily = familyFilter === "all" || model.family === familyFilter
      if (!matchesText || !matchesFamily) return false
      if (hideNoFit && gpu) {
        const breakdown = calculateVramBreakdown(model, gpu.vram, { spillPolicy, systemRamGb })
        if (!breakdown.fits) return false
      }
      return true
    })
    .sort((a, b) => {
      // Sort by total VRAM descending; if no GPU selected fall back to weightsGb
      if (gpu) {
        const bdA = calculateVramBreakdown(a, gpu.vram, { spillPolicy, systemRamGb })
        const bdB = calculateVramBreakdown(b, gpu.vram, { spillPolicy, systemRamGb })
        // Sink won't-fit to bottom when shown
        if (!hideNoFit) {
          if (!bdA.fits && bdB.fits) return 1
          if (bdA.fits && !bdB.fits) return -1
        }
        return bdB.total - bdA.total
      }
      return b.weightsGb - a.weightsGb
    })

  function handleAddCustomGpu(newGpu: GpuSpec) {
    onAddCustomGpu(newGpu)
    setGpu(newGpu)
    setShowGpuForm(false)
  }

  function handleRemoveCustomGpu(id: string) {
    onRemoveCustomGpu(id)
    if (gpu?.id === id) setGpu(null)
  }

  function handleAddCustomModel(model: ModelPreset) {
    setCustomModels(prev => [...prev, model])
    setShowModelForm(false)
  }

  function handleRemoveCustomModel(id: string) {
    setCustomModels(prev => prev.filter(m => m.id !== id))
  }

  return (
    <aside className="flex w-full flex-col gap-5 overflow-y-auto border-b border-zinc-800 bg-zinc-950/95 p-5 lg:h-dvh lg:w-[420px] lg:border-b-0 lg:border-r">
      <section>
        <p className="text-xs font-semibold uppercase tracking-[0.24em] text-cyan-300">VRAM Planner</p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white">Plan model placement before loading weights.</h1>
        <p className="mt-3 text-sm leading-6 text-zinc-400">
          Pick a target GPU and add models to plan VRAM usage before loading weights.
        </p>
      </section>

      {/* GPU selector */}
      <section className="rounded-2xl border border-zinc-800 bg-zinc-900/70 p-4">
        <label className="text-sm font-medium text-zinc-200" htmlFor="gpu-select">Target GPU</label>
        <select
          id="gpu-select"
          className="mt-2 w-full rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-white outline-none focus:border-cyan-400"
          value={gpu?.id ?? ""}
          onChange={(e) => setGpu(allGpus.find((g) => g.id === e.target.value) ?? null)}
        >
          <option value="">Select a GPU</option>
          {GPUs.length > 0 && (
            <optgroup label="Catalog">
              {GPUs.map((g) => (
                <option key={g.id} value={g.id}>{g.name}</option>
              ))}
            </optgroup>
          )}
          {customGpus.length > 0 && (
            <optgroup label="Custom">
              {customGpus.map((g) => (
                <option key={g.id} value={g.id}>{g.name}</option>
              ))}
            </optgroup>
          )}
        </select>

        {gpu ? (
          <div className="mt-4 grid grid-cols-3 gap-2 text-center text-xs">
            <Metric label="VRAM" value={`${gpu.vram} GB`} />
            <Metric label="Used" value={`${totalUsedVram.toFixed(1)} GB`} />
            <Metric label="Free" value={`${totalFreeVram.toFixed(1)} GB`} tone={totalFreeVram < 0 ? "bad" : "good"} />
          </div>
        ) : null}

        {/* System RAM + spill policy */}
        <div className="mt-4 grid grid-cols-2 gap-3">
          <div className="flex flex-col gap-1">
            <label className="text-xs text-zinc-400">System RAM (GB)</label>
            <input
              type="number"
              min="8"
              max="2048"
              step="8"
              className="rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-white outline-none focus:border-cyan-400"
              value={systemRamGb}
              onChange={e => setSystemRamGb(Number(e.target.value))}
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-zinc-400">Spill strategy</label>
            <select
              className="rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-white outline-none focus:border-cyan-400"
              value={spillPolicy}
              onChange={e => setSpillPolicy(e.target.value as SpillPolicy)}
            >
              <option value="avoid">Avoid spill</option>
              <option value="allow">Allow RAM spill</option>
            </select>
          </div>
        </div>

        {showGpuForm ? (
          <CustomGpuForm onAdd={handleAddCustomGpu} onCancel={() => setShowGpuForm(false)} />
        ) : (
          <button
            type="button"
            onClick={() => setShowGpuForm(true)}
            className="mt-3 w-full rounded-xl border border-dashed border-zinc-700 px-3 py-2 text-xs text-zinc-500 transition hover:border-cyan-600 hover:text-cyan-400"
          >
            + Add custom GPU
          </button>
        )}

        {customGpus.length > 0 && !showGpuForm && (
          <div className="mt-3 flex flex-col gap-1">
            {customGpus.map(g => (
              <div key={g.id} className="flex items-center justify-between rounded-lg bg-zinc-950 px-3 py-1.5">
                <span className="text-xs text-zinc-300">{g.name} — {g.vram} GB</span>
                <button
                  type="button"
                  onClick={() => handleRemoveCustomGpu(g.id)}
                  className="text-xs text-zinc-600 transition hover:text-red-400"
                >
                  Remove
                </button>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Model catalog */}
      <section className="flex min-h-0 flex-1 flex-col rounded-2xl border border-zinc-800 bg-zinc-900/70 p-4">
        <div className="flex items-center justify-between gap-3">
          <h2 className="text-sm font-semibold text-zinc-100">Model Catalog</h2>
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-500">{filteredModels.length} models</span>
            {gpu && (
              <button
                type="button"
                onClick={() => setHideNoFit(v => !v)}
                className={[
                  "rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide transition",
                  hideNoFit
                    ? "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
                    : "bg-red-900/40 text-red-400 hover:bg-red-900/60",
                ].join(" ")}
                title={hideNoFit ? "Show models that won't fit" : "Hide models that won't fit"}
              >
                {hideNoFit ? "fits only" : "showing all"}
              </button>
            )}
          </div>
        </div>
        <input
          className="mt-3 rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-white outline-none placeholder:text-zinc-600 focus:border-cyan-400"
          placeholder="Search model, family, tag"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
        <select
          className="mt-2 rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-white outline-none focus:border-cyan-400"
          value={familyFilter}
          onChange={(e) => setFamilyFilter(e.target.value)}
        >
          <option value="all">All families</option>
          {families.map((f) => (
            <option key={f} value={f}>{f}</option>
          ))}
        </select>

        <div className="mt-3 flex flex-col gap-2 overflow-y-auto pr-1">
          {filteredModels.map((model) => {
            const breakdown = gpu ? calculateVramBreakdown(model, gpu.vram) : null
            const doesntFit = breakdown !== null && !breakdown.fits
            return (
              <div key={model.id} className="flex items-stretch gap-1">
                <button
                  type="button"
                  disabled={!gpu}
                  onClick={() => onAddModel(model)}
                  className={[
                    "min-w-0 flex-1 rounded-xl border bg-zinc-950 p-3 text-left transition disabled:cursor-not-allowed disabled:opacity-45",
                    doesntFit
                      ? "border-red-900/60 hover:border-red-600"
                      : "border-zinc-800 hover:border-cyan-500",
                  ].join(" ")}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <p className="truncate text-sm font-medium text-white">{model.name}</p>
                        {doesntFit && (
                          <span className="shrink-0 rounded-full bg-red-900/50 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-red-400">
                            won't fit
                          </span>
                        )}
                      </div>
                      <p className="mt-1 text-xs text-zinc-500">
                        {model.family} / {model.quantization} / {model.contextLength.toLocaleString()} ctx
                      </p>
                      {breakdown && (
                        <p className="mt-1 text-xs text-zinc-600">
                          {breakdown.weights.toFixed(1)} weights
                          {breakdown.kvCache > 0 ? ` · ${breakdown.kvCache.toFixed(1)} KV` : ""}
                          {` · ${breakdown.headroom.toFixed(1)} headroom`}
                          {" = "}
                          <span className={doesntFit ? "text-red-400" : "text-zinc-400"}>
                            {breakdown.total.toFixed(1)} GB total
                          </span>
                        </p>
                      )}
                    </div>
                    <span className={[
                      "shrink-0 rounded-full px-2 py-1 text-xs font-semibold",
                      doesntFit
                        ? "bg-red-900/30 text-red-400"
                        : "bg-cyan-400/10 text-cyan-200",
                    ].join(" ")}>
                      {breakdown ? `${breakdown.total.toFixed(1)} GB` : `${model.weightsGb} GB`}
                    </span>
                  </div>
                </button>
                {model.tags.includes("custom") && (
                  <button
                    type="button"
                    onClick={() => handleRemoveCustomModel(model.id)}
                    className="rounded-xl border border-zinc-800 bg-zinc-950 px-2 text-zinc-600 transition hover:border-red-800 hover:text-red-400"
                    title="Remove custom model"
                  >
                    ✕
                  </button>
                )}
              </div>
            )
          })}
        </div>

        {showModelForm ? (
          <div className="mt-3 shrink-0 rounded-xl border border-zinc-700 bg-zinc-950 p-3">
            <p className="text-xs font-semibold text-zinc-300">New custom model</p>
            <CustomModelForm onAdd={handleAddCustomModel} onCancel={() => setShowModelForm(false)} />
          </div>
        ) : (
          <button
            type="button"
            onClick={() => setShowModelForm(true)}
            className="mt-3 shrink-0 w-full rounded-xl border border-dashed border-zinc-700 px-3 py-2 text-xs text-zinc-500 transition hover:border-cyan-600 hover:text-cyan-400"
          >
            + Add custom model
          </button>
        )}
      </section>

      {/* Planned loads */}
      <section className="rounded-2xl border border-zinc-800 bg-zinc-900/70 p-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-zinc-100">Planned Loads</h2>
          <span className="text-xs text-zinc-500">{loadedModels.length}</span>
        </div>
        <div className="mt-3 space-y-2">
          {loadedModels.length === 0 ? (
            <p className="text-sm text-zinc-500">Select a GPU, then add models from the catalog.</p>
          ) : loadedModels.map((model) => {
            const breakdown = gpu ? calculateVramBreakdown(model, gpu.vram, {
              slots: model.slots,
              contextLength: model.contextLength,
              systemRamGb,
              spillPolicy,
            }) : null
            return (
              <div key={model.instanceId} className="flex items-center justify-between gap-3 rounded-xl bg-zinc-950 px-3 py-2">
                <div className="min-w-0">
                  <p className="truncate text-sm text-white">{model.name}</p>
                  <p className="text-xs text-zinc-500">
                    {breakdown ? `${breakdown.gpuUsed.toFixed(1)} GB` : "—"}
                    {" · "}
                    {model.slots} slot{model.slots !== 1 ? "s" : ""}
                    {" · "}
                    {model.contextLength >= 1000 ? `${(model.contextLength / 1000).toFixed(0)}k` : model.contextLength} ctx
                  </p>
                </div>
                <button
                  type="button"
                  className="shrink-0 rounded-lg border border-zinc-700 px-2 py-1 text-xs text-zinc-300 hover:border-red-400 hover:text-red-200"
                  onClick={() => removeModel(model.instanceId)}
                >
                  Remove
                </button>
              </div>
            )
          })}
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
