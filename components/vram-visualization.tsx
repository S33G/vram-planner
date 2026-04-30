"use client"

import { useEffect, useState } from "react"
import type { ReactNode } from "react"
import type { GpuSpec, Partition, SpillPolicy } from "@/lib/db"

interface LiveGpu {
  index: number
  name: string
  memoryTotalMiB: number
  memoryUsedMiB: number
  memoryFreeMiB: number
  utilizationGpu: number | null
  temperature: number | null
}

interface LiveGpuResponse {
  available: boolean
  gpus: LiveGpu[]
  error?: string
  updatedAt: string
}

interface VramVisualizationProps {
  gpu: GpuSpec | null
  partitions: Partition[]
  totalUsedVram: number
  totalFreeVram: number
  totalRamSpill: number
  systemRamGb: number
  spillPolicy: SpillPolicy
  updateModelSlots: (instanceId: string, slots: number) => void
  updateModelContext: (instanceId: string, ctx: number) => void
}

export function VramVisualization({
  gpu, partitions, totalUsedVram, totalFreeVram, totalRamSpill,
  updateModelSlots, updateModelContext,
}: VramVisualizationProps) {
  const [view, setView] = useState<"plan" | "live">("plan")
  const [liveState, setLiveState] = useState<LiveGpuResponse | null>(null)
  const [isLoadingLive, setIsLoadingLive] = useState(false)

  useEffect(() => {
    if (view !== "live") return
    let cancelled = false
    async function loadGpuState() {
      setIsLoadingLive(true)
      try {
        const response = await fetch("/api/gpus", { cache: "no-store" })
        const data = await response.json() as LiveGpuResponse
        if (!cancelled) setLiveState(data)
      } catch (error) {
        if (!cancelled) setLiveState({
          available: false, gpus: [],
          error: error instanceof Error ? error.message : "Unable to load GPU state",
          updatedAt: new Date().toISOString(),
        })
      } finally {
        if (!cancelled) setIsLoadingLive(false)
      }
    }
    loadGpuState()
    const interval = window.setInterval(loadGpuState, 5000)
    return () => { cancelled = true; window.clearInterval(interval) }
  }, [view])

  return (
    <div className="flex h-dvh flex-1 flex-col overflow-y-auto bg-[radial-gradient(circle_at_top_right,_rgba(34,211,238,0.16),_transparent_28rem),#09090b] p-5 md:p-8">
      <div className="flex flex-col justify-between gap-4 md:flex-row md:items-start">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.24em] text-zinc-500">Machine View</p>
           <h2 className="mt-2 flex items-center gap-3 text-3xl font-semibold tracking-tight text-white md:text-5xl">
             {view === "plan" ? "Simulated VRAM allocation" : "Live NVIDIA memory state"}
             <span className="rounded-full bg-amber-500/20 px-3 py-1 text-xs font-semibold uppercase tracking-wider text-amber-400">Beta</span>
           </h2>
          <p className="mt-3 max-w-2xl text-sm leading-6 text-zinc-400">
            Use planning mode for proposed model placement. Switch to live mode to inspect the current host if the container can execute <code className="text-cyan-200">nvidia-smi</code>.
          </p>
        </div>
        <div className="flex rounded-2xl border border-zinc-800 bg-zinc-950 p-1">
          <ToggleButton active={view === "plan"} onClick={() => setView("plan")}>Plan</ToggleButton>
          <ToggleButton active={view === "live"} onClick={() => setView("live")}>Live host</ToggleButton>
        </div>
      </div>

      {view === "plan" ? (
        <PlanView
          gpu={gpu}
          partitions={partitions}
          totalUsedVram={totalUsedVram}
          totalFreeVram={totalFreeVram}
          totalRamSpill={totalRamSpill}
          updateModelSlots={updateModelSlots}
          updateModelContext={updateModelContext}
        />
      ) : (
        <LiveView state={liveState} isLoading={isLoadingLive} />
      )}
    </div>
  )
}

function PlanView({
  gpu, partitions, totalUsedVram, totalFreeVram, totalRamSpill,
  updateModelSlots, updateModelContext,
}: Omit<VramVisualizationProps, 'systemRamGb' | 'spillPolicy'>) {
  if (!gpu) {
    return (
      <div className="mt-10 flex flex-1 items-center justify-center rounded-3xl border border-dashed border-zinc-800 bg-zinc-950/50 p-10 text-center">
        <div>
          <p className="text-lg font-medium text-white">Select a GPU to start planning.</p>
          <p className="mt-2 text-sm text-zinc-500">Each model is broken down into weights, KV cache, and headroom.</p>
        </div>
      </div>
    )
  }

  const usedPercent   = Math.min((totalUsedVram / gpu.vram) * 100, 100)
  const freePercent   = Math.max((totalFreeVram / gpu.vram) * 100, 0)
  const overflowing   = totalUsedVram > gpu.vram
  const totalWeights  = partitions.reduce((s, p) => s + p.breakdown.weights, 0)
  const totalKv       = partitions.reduce((s, p) => s + p.breakdown.kvCache, 0)
  const totalHeadroom = partitions.reduce((s, p) => s + p.breakdown.headroom, 0)

  // Overall fit status
  const anyNoFit  = partitions.some(p => p.breakdown.fitStatus === 'no-fit')
  const anySpills = partitions.some(p => p.breakdown.fitStatus === 'spills')
  const fitLabel  = anyNoFit ? 'Does not fit' : anySpills ? 'Spills to RAM' : partitions.length ? 'Fits in VRAM' : '—'
  const fitColor  = anyNoFit ? 'bg-red-900/40 text-red-400' : anySpills ? 'bg-amber-900/40 text-amber-400' : 'bg-emerald-900/40 text-emerald-400'

  return (
    <div className="mt-8 flex flex-col gap-5">
      {/* KPI row */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <KpiCard label="GPU used"   value={`${totalUsedVram.toFixed(1)} GB`} />
        <KpiCard label="GPU free"   value={`${totalFreeVram.toFixed(1)} GB`} tone={overflowing ? 'bad' : 'normal'} />
        <KpiCard label="RAM spill"  value={`${totalRamSpill.toFixed(1)} GB`} tone={totalRamSpill > 0 ? 'warn' : 'normal'} />
        <KpiCard label="Fit status" value={fitLabel} raw fitColor={fitColor} />
      </div>

      <div className="grid flex-1 gap-5 xl:grid-cols-[1.35fr_0.65fr]">
        {/* Left column: bar + per-partition details */}
        <div className="flex flex-col gap-5">
          {/* VRAM map */}
          <section className="rounded-3xl border border-zinc-800 bg-zinc-950/80 p-5 md:p-7">
            <div className="flex flex-col justify-between gap-3 md:flex-row md:items-end">
              <div>
                <h3 className="text-xl font-semibold text-white">{gpu.name}</h3>
                <p className="mt-1 text-sm text-zinc-500">{gpu.vram} GB VRAM · {gpu.memoryBandwidth} GB/s · {gpu.tdp} W TDP</p>
              </div>
              <span className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-bold ${fitColor}`}>
                {fitLabel}
              </span>
            </div>

            {/* Segmented bar */}
            <div className="mt-6 overflow-hidden rounded-3xl border border-zinc-800 bg-zinc-900 p-3">
              <div className="flex h-20 overflow-hidden rounded-2xl bg-zinc-950">
                {partitions.map((p) => {
                  const base = Math.max(totalUsedVram, gpu.vram)
                  const wPct  = (p.breakdown.weights  / base) * 100
                  const kvPct = (p.breakdown.kvCache  / base) * 100
                  const hPct  = (p.breakdown.headroom / base) * 100
                  const totalPct = Math.max(((p.breakdown.gpuUsed / base) * 100), 3)
                  return (
                    <div key={p.id} className="flex min-w-0 flex-col overflow-hidden border-r border-zinc-950" style={{ width: `${totalPct}%` }} title={`${p.name}: ${p.vramAllocated.toFixed(1)} GB`}>
                      <div className="overflow-hidden bg-cyan-600 px-1 pb-0.5 text-[10px] font-bold text-zinc-950" style={{ height: `${wPct}%` }} />
                      {p.breakdown.kvCache > 0 && (
                        <div className="overflow-hidden bg-blue-600" style={{ height: `${kvPct}%` }} />
                      )}
                      <div className="flex flex-1 items-center justify-center overflow-hidden bg-amber-700/70 text-[10px] font-bold text-amber-200" style={{ height: `${hPct}%` }}>
                        <span className="line-clamp-1 px-1 leading-tight">{p.name}</span>
                      </div>
                    </div>
                  )
                })}
                {freePercent > 0 && (
                  <div className="flex flex-1 items-center justify-center bg-emerald-500/15 text-xs font-medium text-emerald-200" style={{ width: `${freePercent}%` }}>
                    {freePercent > 8 ? "Free" : ""}
                  </div>
                )}
              </div>
              <div className="mt-3 flex flex-wrap gap-3 text-xs text-zinc-400">
                <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-sm bg-cyan-600" />Weights</span>
                <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-sm bg-blue-600" />KV cache</span>
                <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-sm bg-amber-700/70" />Headroom</span>
                <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-sm bg-emerald-500/40" />Free</span>
              </div>
            </div>

            <div className="mt-4 grid grid-cols-4 gap-3">
              <SummaryCard label="Capacity"    value={`${gpu.vram} GB`} />
              <SummaryCard label="Models"      value={`${partitions.length}`} />
              <SummaryCard label="Utilization" value={`${usedPercent.toFixed(0)}%`} tone={overflowing ? 'bad' : 'normal'} />
              <SummaryCard label="Allocated"   value={`${totalUsedVram.toFixed(1)} GB`} />
            </div>
            {partitions.length > 0 && (
              <div className="mt-3 grid grid-cols-3 gap-3">
                <SummaryCard label="Weights"  value={`${totalWeights.toFixed(1)} GB`} />
                <SummaryCard label="KV cache" value={`${totalKv.toFixed(1)} GB`} />
                <SummaryCard label="Headroom" value={`${totalHeadroom.toFixed(1)} GB`} />
              </div>
            )}
          </section>

          {/* Per-partition fit + slot planner */}
          {partitions.map(p => (
            <PartitionDetail
              key={p.id}
              partition={p}
              gpuVram={gpu.vram}
              onSlotsChange={slots => updateModelSlots(p.id, slots)}
              onContextChange={ctx => updateModelContext(p.id, ctx)}
            />
          ))}
        </div>

        {/* Right column: perf shape + recommendations (first partition or aggregate) */}
        <div className="flex flex-col gap-5">
          {partitions.length === 0 ? (
            <div className="rounded-3xl border border-dashed border-zinc-800 bg-zinc-950/50 p-7 text-center">
              <p className="text-sm text-zinc-500">Add models to see fit analysis, performance shape, and recommendations.</p>
            </div>
          ) : partitions.map(p => (
            <section key={p.id} className="rounded-3xl border border-zinc-800 bg-zinc-950/80 p-5 md:p-7">
              <p className="text-xs font-semibold uppercase tracking-widest text-zinc-500">{p.name}</p>

              <h4 className="mt-3 text-sm font-semibold text-zinc-100">Performance shape</h4>
              <div className="mt-2 divide-y divide-zinc-800 rounded-2xl border border-zinc-800 overflow-hidden">
                <PerfRow label="Prefill latency"      value={p.perf.prefill} />
                <PerfRow label="Token generation"     value={p.perf.tokenGen} />
                <PerfRow label="Risk"                 value={p.perf.risk} />
                <PerfRow label="Effective context"    value={p.perf.effectiveCtx} />
              </div>

              <h4 className="mt-5 text-sm font-semibold text-zinc-100">Recommendations</h4>
              <div className="mt-2 flex flex-col gap-2">
                {p.recommendations.map((r, i) => (
                  <div key={i} className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-3">
                    <p className="text-xs font-semibold text-zinc-200">{r.title}</p>
                    <p className="mt-1 text-xs leading-5 text-zinc-500">{r.text}</p>
                  </div>
                ))}
              </div>
            </section>
          ))}
        </div>
      </div>
    </div>
  )
}

function PartitionDetail({
  partition, gpuVram, onSlotsChange, onContextChange,
}: {
  partition: Partition
  gpuVram: number
  onSlotsChange: (v: number) => void
  onContextChange: (v: number) => void
}) {
  const { breakdown, name, slots, contextLength } = partition
  const perSlotKv = slots > 0 ? breakdown.kvCache / slots : 0

  const fitBadgeColor = breakdown.fitStatus === 'fits'
    ? 'bg-emerald-900/40 text-emerald-400'
    : breakdown.fitStatus === 'spills'
    ? 'bg-amber-900/40 text-amber-400'
    : 'bg-red-900/40 text-red-400'
  const fitBadgeLabel = breakdown.fitStatus === 'fits' ? 'Fits' : breakdown.fitStatus === 'spills' ? 'Spills to RAM' : 'Does not fit'

  return (
    <section className="rounded-3xl border border-zinc-800 bg-zinc-950/80 p-5 md:p-7">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-base font-semibold text-white">{name}</h3>
        <span className={`rounded-full px-2.5 py-0.5 text-xs font-bold ${fitBadgeColor}`}>{fitBadgeLabel}</span>
      </div>

      {/* Controls: slots + context */}
      <div className="mt-4 grid grid-cols-2 gap-3">
        <div className="flex flex-col gap-1">
          <label className="text-xs text-zinc-400">Parallel slots</label>
          <input
            type="number"
            min="1"
            max="32"
            className="rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-white outline-none focus:border-cyan-400"
            value={slots}
            onChange={e => onSlotsChange(Math.max(1, parseInt(e.target.value) || 1))}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-zinc-400">Context (tokens)</label>
          <input
            type="number"
            min="512"
            max="1048576"
            step="512"
            className="rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-white outline-none focus:border-cyan-400"
            value={contextLength}
            onChange={e => onContextChange(Math.max(512, parseInt(e.target.value) || 512))}
          />
        </div>
      </div>

      {/* Segmented mini-bar */}
      <div className="mt-4 flex h-2 overflow-hidden rounded-full bg-zinc-800">
        <div className="h-full bg-cyan-600"   style={{ width: `${Math.min((breakdown.weights  / gpuVram) * 100, 100)}%` }} />
        <div className="h-full bg-blue-600"   style={{ width: `${Math.min((breakdown.kvCache  / gpuVram) * 100, 100)}%` }} />
        <div className="h-full bg-amber-700"  style={{ width: `${Math.min((breakdown.headroom / gpuVram) * 100, 100)}%` }} />
      </div>
      <div className="mt-2 flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-zinc-500">
        <span>Weights {breakdown.weights.toFixed(1)} GB</span>
        {breakdown.kvCache > 0 && <span>KV {breakdown.kvCache.toFixed(1)} GB ({slots} slot{slots !== 1 ? 's' : ''})</span>}
        <span>Headroom {breakdown.headroom.toFixed(1)} GB</span>
        <span className={breakdown.fits ? "text-zinc-400" : "font-semibold text-red-400"}>
          = {breakdown.total.toFixed(1)} GB total
        </span>
        {breakdown.overflow > 0 && (
          <span className="font-semibold text-red-400">
            ({breakdown.overflow.toFixed(1)} GB over)
          </span>
        )}
      </div>

      {/* Slot planner */}
      {slots > 0 && (
        <div className="mt-4">
          <p className="text-xs font-semibold text-zinc-400">Slot planner</p>
          <p className="mt-0.5 text-xs text-zinc-600">Every slot consumes its own KV allocation. More slots multiply KV pressure.</p>
          <div className="mt-3 flex flex-col gap-2">
            {Array.from({ length: Math.min(slots, 8) }).map((_, i) => (
              <div key={i} className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-3">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold text-zinc-300">Slot {i + 1}</span>
                  <span className="text-zinc-500">
                    {contextLength >= 1000 ? `${(contextLength / 1000).toFixed(0)}k` : contextLength} tokens
                  </span>
                </div>
                <div className="mt-2 flex h-2 overflow-hidden rounded-full bg-zinc-800">
                  {/* KV share */}
                  <div className="h-full bg-blue-600" style={{ width: `${Math.min((perSlotKv / gpuVram) * 100 * 4, 100)}%` }} />
                  {/* Free headroom (visual) */}
                  <div className="h-full bg-emerald-600/30" style={{ width: `${Math.max(0, 40 - Math.min((perSlotKv / gpuVram) * 100 * 4, 40))}%` }} />
                </div>
                <p className="mt-1 text-xs text-zinc-600">KV share ≈ {perSlotKv.toFixed(2)} GB</p>
              </div>
            ))}
            {slots > 8 && (
              <p className="text-xs text-zinc-600">…and {slots - 8} more slots</p>
            )}
          </div>
        </div>
      )}
    </section>
  )
}

function LiveView({ state, isLoading }: { state: LiveGpuResponse | null; isLoading: boolean }) {
  if (!state && isLoading) return <StatusPanel title="Checking host GPUs" description="Running nvidia-smi from the server container..." />
  if (!state?.available) return (
    <StatusPanel
      title="No live NVIDIA state available"
      description={state?.error ?? "nvidia-smi was not found or no GPUs were reported. Run the container with NVIDIA runtime support to enable this view."}
    />
  )
  return (
    <div className="mt-8 grid gap-5 xl:grid-cols-2">
      {state.gpus.map((gpu) => {
        const usedPercent = gpu.memoryTotalMiB > 0 ? (gpu.memoryUsedMiB / gpu.memoryTotalMiB) * 100 : 0
        return (
          <section key={gpu.index} className="rounded-3xl border border-zinc-800 bg-zinc-950/80 p-5 md:p-7">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">GPU {gpu.index}</p>
                <h3 className="mt-2 text-xl font-semibold text-white">{gpu.name}</h3>
              </div>
              <span className="rounded-full bg-cyan-400/10 px-3 py-1 text-sm font-semibold text-cyan-200">{usedPercent.toFixed(0)}% VRAM</span>
            </div>
            <div className="mt-8 h-8 overflow-hidden rounded-full bg-zinc-800">
              <div className="h-full rounded-full bg-gradient-to-r from-cyan-400 to-fuchsia-400" style={{ width: `${Math.min(usedPercent, 100)}%` }} />
            </div>
            <div className="mt-6 grid gap-3 sm:grid-cols-4">
              <SummaryCard label="Used"     value={`${mibToGib(gpu.memoryUsedMiB)} GiB`} />
              <SummaryCard label="Free"     value={`${mibToGib(gpu.memoryFreeMiB)} GiB`} />
              <SummaryCard label="Total"    value={`${mibToGib(gpu.memoryTotalMiB)} GiB`} />
              <SummaryCard label="GPU util" value={gpu.utilizationGpu == null ? "n/a" : `${gpu.utilizationGpu}%`} />
            </div>
            <p className="mt-5 text-xs text-zinc-500">Updated {new Date(state.updatedAt).toLocaleTimeString()}</p>
          </section>
        )
      })}
    </div>
  )
}

// ---- Primitives ----

function ToggleButton({ active, children, onClick }: { active: boolean; children: ReactNode; onClick: () => void }) {
  return (
    <button type="button" onClick={onClick} className={`rounded-xl px-4 py-2 text-sm font-medium transition ${active ? "bg-cyan-400 text-zinc-950" : "text-zinc-400 hover:text-white"}`}>
      {children}
    </button>
  )
}

function KpiCard({ label, value, tone, raw, fitColor }: { label: string; value: string; tone?: 'normal' | 'bad' | 'warn'; raw?: boolean; fitColor?: string }) {
  const valueClass = raw && fitColor
    ? `mt-2 text-lg font-black ${fitColor.split(' ')[1]}`
    : `mt-2 text-lg font-black ${tone === 'bad' ? 'text-red-300' : tone === 'warn' ? 'text-amber-400' : 'text-white'}`
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-4">
      <p className="text-xs uppercase tracking-widest text-zinc-500">{label}</p>
      <p className={valueClass}>{value}</p>
    </div>
  )
}

function SummaryCard({ label, value, tone }: { label: string; value: string; tone?: 'normal' | 'bad' }) {
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/80 p-4">
      <p className="text-xs uppercase tracking-widest text-zinc-500">{label}</p>
      <p className={`mt-2 text-lg font-semibold ${tone === 'bad' ? 'text-red-300' : 'text-white'}`}>{value}</p>
    </div>
  )
}

function PerfRow({ label, value }: { label: string; value: string }) {
  const isGood    = ['Fast', 'Low'].includes(value)
  const isBad     = ['Slow', 'Blocked', 'Does not fit', 'High'].includes(value)
  const isWarn    = ['Moderate'].includes(value)
  const valClass  = isBad ? 'text-red-400' : isWarn ? 'text-amber-400' : isGood ? 'text-emerald-400' : 'text-zinc-300'
  return (
    <div className="flex items-center justify-between gap-3 px-3 py-2.5">
      <span className="text-xs text-zinc-500">{label}</span>
      <span className={`text-xs font-semibold ${valClass}`}>{value}</span>
    </div>
  )
}

function StatusPanel({ title, description }: { title: string; description: string }) {
  return (
    <div className="mt-10 flex flex-1 items-center justify-center rounded-3xl border border-dashed border-zinc-800 bg-zinc-950/50 p-10 text-center">
      <div className="max-w-lg">
        <p className="text-lg font-medium text-white">{title}</p>
        <p className="mt-2 text-sm leading-6 text-zinc-500">{description}</p>
      </div>
    </div>
  )
}

function mibToGib(value: number) { return (value / 1024).toFixed(1) }
