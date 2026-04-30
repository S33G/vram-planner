"use client"

import { useEffect, useRef, useState } from "react"
import type { ReactNode } from "react"
import type { GpuSpec, Partition, SpillPolicy } from "@/lib/db"
import { GPUs } from "@/lib/db"
import { CustomGpuForm } from "@/components/custom-gpu-form"

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
  setGpu: (gpu: GpuSpec | null) => void
  allGpus: GpuSpec[]
  onAddCustomGpu: (gpu: GpuSpec) => void
  partitions: Partition[]
  totalUsedVram: number
  totalFreeVram: number
  totalRamSpill: number
  systemRamGb: number
  spillPolicy: SpillPolicy
  updateModelSlots: (instanceId: string, slots: number) => void
  updateModelContext: (instanceId: string, ctx: number) => void
}

const liveModeEnabled = process.env.NEXT_PUBLIC_ENABLE_LIVE_MODE !== "false"

// ---- Tooltip primitives ----

/**
 * A floating tooltip panel. Appears above its trigger by default.
 * `children` is the trigger element; `content` is what shows in the bubble.
 */
function Tooltip({
  content,
  children,
  className = "",
  style,
}: {
  content: ReactNode
  children: ReactNode
  className?: string
  style?: React.CSSProperties
}) {
  const [visible, setVisible] = useState(false)
  const hideTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  function show() {
    if (hideTimer.current) clearTimeout(hideTimer.current)
    setVisible(true)
  }
  function hide() {
    hideTimer.current = setTimeout(() => setVisible(false), 80)
  }

  return (
    <span
      className={`relative inline-flex ${className}`}
      style={style}
      onMouseEnter={show}
      onMouseLeave={hide}
      onFocus={show}
      onBlur={hide}
    >
      {children}
      {visible && (
        <span
          role="tooltip"
          className="pointer-events-none absolute bottom-full left-1/2 z-50 mb-2 w-64 -translate-x-1/2 rounded-2xl border border-zinc-700 bg-zinc-900 px-3 py-2.5 text-left text-xs leading-5 text-zinc-300 shadow-xl"
        >
          {content}
          {/* Arrow */}
          <span className="absolute left-1/2 top-full -translate-x-1/2 border-4 border-transparent border-t-zinc-700" />
        </span>
      )}
    </span>
  )
}

/**
 * A small `?` icon that shows an explanatory tooltip on hover.
 * Used next to labels to explain concepts.
 */
function InfoTooltip({ content }: { content: ReactNode }) {
  return (
    <Tooltip content={content}>
      <span
        aria-label="More information"
        className="inline-flex h-3.5 w-3.5 cursor-help items-center justify-center rounded-full border border-zinc-600 text-[9px] font-bold text-zinc-500 transition hover:border-zinc-400 hover:text-zinc-300"
      >
        ?
      </span>
    </Tooltip>
  )
}

// ---- Tooltip copy ----

const TOOLTIPS = {
  parallelSlots: (
    <>
      <strong className="text-zinc-100">Parallel slots</strong> — how many concurrent
      inference requests this model serves simultaneously.
      <br /><br />
      Each slot needs its own <em>KV cache</em> buffer, so doubling slots doubles KV
      memory. More slots → higher throughput, but higher VRAM pressure.
    </>
  ),
  contextLength: (
    <>
      <strong className="text-zinc-100">Context length</strong> — the maximum number of
      tokens (roughly words) the model can &ldquo;see&rdquo; at once.
      <br /><br />
      Longer context means a larger KV cache, which uses more VRAM. 128k context uses
      roughly 256× the KV memory of 512 tokens.
    </>
  ),
  weights: (
    <>
      <strong className="text-zinc-100">Weights</strong> — the model parameters loaded
      into VRAM. This is fixed regardless of how many requests you serve or how long the
      context is.
    </>
  ),
  kvCache: (
    <>
      <strong className="text-zinc-100">KV cache</strong> — key/value attention tensors
      computed per token. Grows with both context length and the number of parallel slots.
      <br /><br />
      Formula: <code className="rounded bg-zinc-800 px-1">ctx ÷ 1k × kvPer1k × slots</code>
    </>
  ),
  headroom: (
    <>
      <strong className="text-zinc-100">Headroom</strong> — a fixed overhead budget for
      CUDA kernels, activations, and framework bookkeeping. Typically 0.5–2 GB depending
      on the model.
    </>
  ),
  freeVram: (
    <>
      <strong className="text-zinc-100">Free VRAM</strong> — unallocated GPU memory
      after all planned model loads. Larger free headroom allows loading additional models
      or running larger contexts.
    </>
  ),
  gpuUsed: (
    <>
      <strong className="text-zinc-100">GPU used</strong> — total VRAM allocated across
      all loaded models (weights + KV cache + headroom).
    </>
  ),
  gpuFree: (
    <>
      <strong className="text-zinc-100">GPU free</strong> — VRAM remaining after all
      planned loads. Negative when the plan overflows the GPU capacity.
    </>
  ),
  ramSpill: (
    <>
      <strong className="text-zinc-100">RAM spill</strong> — model layers that cannot
      fit in VRAM and fall back to system RAM. Dramatically slows inference because RAM
      bandwidth (~50 GB/s) is far below GPU bandwidth (600–3000 GB/s).
    </>
  ),
  fitStatus: (
    <>
      <strong className="text-zinc-100">Fit status</strong> — whether the entire plan
      fits in VRAM, spills to RAM, or does not fit at all.
      <br /><br />
      <em>Fits</em> — all layers in VRAM.<br />
      <em>Spills to RAM</em> — partial offload to system RAM.<br />
      <em>Does not fit</em> — even with RAM spill the model is too large.
    </>
  ),
  prefill: (
    <>
      <strong className="text-zinc-100">Prefill latency</strong> — time to process the
      input prompt. Compute-bound; dominated by model size and GPU TFLOPS.
      Long contexts make this slow because every input token must be processed.
    </>
  ),
  tokenGen: (
    <>
      <strong className="text-zinc-100">Token generation</strong> — speed of producing
      output tokens one-by-one. Memory-bandwidth-bound; dominated by GPU bandwidth
      (GB/s), not compute.
    </>
  ),
  risk: (
    <>
      <strong className="text-zinc-100">Risk</strong> — overall pressure on VRAM:
      how close the allocation is to the GPU capacity limit. High risk means small
      increases in context or batch size may cause OOM errors.
    </>
  ),
  effectiveCtx: (
    <>
      <strong className="text-zinc-100">Effective context</strong> — whether the
      requested context length is sustainable given the KV cache budget. &ldquo;Full&rdquo;
      means the complete context fits; &ldquo;Reduced&rdquo; signals KV pressure.
    </>
  ),
  utilization: (
    <>
      <strong className="text-zinc-100">Utilization</strong> — percentage of GPU VRAM
      consumed by this plan. Above ~90% leaves little room for runtime overhead.
    </>
  ),
  capacity: (
    <>
      <strong className="text-zinc-100">Capacity</strong> — total VRAM available on the
      selected GPU.
    </>
  ),
  allocated: (
    <>
      <strong className="text-zinc-100">Allocated</strong> — total VRAM assigned across
      all planned model loads.
    </>
  ),
  slotPlanner: (
    <>
      <strong className="text-zinc-100">Slot planner</strong> — visual breakdown of how
      much KV cache each parallel slot consumes relative to total GPU VRAM. Each stripe
      represents one concurrent inference slot.
    </>
  ),
}

// ---- Main export ----

export function VramVisualization({
  gpu, setGpu, allGpus, onAddCustomGpu,
  partitions, totalUsedVram, totalFreeVram, totalRamSpill,
  updateModelSlots, updateModelContext,
}: VramVisualizationProps) {
  const [view, setView] = useState<"plan" | "live">("plan")
  const [liveState, setLiveState] = useState<LiveGpuResponse | null>(null)
  const [isLoadingLive, setIsLoadingLive] = useState(false)

  useEffect(() => {
    if (!liveModeEnabled || view !== "live") return
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
             {liveModeEnabled && view === "live" ? "Live NVIDIA memory state" : "Simulated VRAM allocation"}
             <span className="rounded-full bg-amber-500/20 px-3 py-1 text-xs font-semibold uppercase tracking-wider text-amber-400">Beta</span>
           </h2>
        </div>
        {liveModeEnabled && (
          <div className="flex rounded-2xl border border-zinc-800 bg-zinc-950 p-1">
            <ToggleButton active={view === "plan"} onClick={() => setView("plan")}>Plan</ToggleButton>
            <ToggleButton active={view === "live"} onClick={() => setView("live")}>Live host</ToggleButton>
          </div>
        )}
      </div>

      {!liveModeEnabled || view === "plan" ? (
        <PlanView
          gpu={gpu}
          setGpu={setGpu}
          allGpus={allGpus}
          onAddCustomGpu={onAddCustomGpu}
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
  gpu, setGpu, allGpus, onAddCustomGpu,
  partitions, totalUsedVram, totalFreeVram, totalRamSpill,
  updateModelSlots, updateModelContext,
}: Omit<VramVisualizationProps, 'systemRamGb' | 'spillPolicy'>) {
  const [showCustomForm, setShowCustomForm] = useState(false)

  if (!gpu) {
    const catalogGpus = allGpus.filter(g => GPUs.some(cg => cg.id === g.id))
    const customGpus  = allGpus.filter(g => !GPUs.some(cg => cg.id === g.id))

    function handleAddCustom(newGpu: GpuSpec) {
      onAddCustomGpu(newGpu)
      setGpu(newGpu)
      setShowCustomForm(false)
    }

    return (
      <div className="mt-10 flex flex-1 items-center justify-center rounded-3xl border border-dashed border-zinc-800 bg-zinc-950/50 p-10">
        <div className="w-full max-w-sm">
          <p className="text-center text-lg font-medium text-white">Select a GPU to start planning.</p>
          <p className="mt-2 text-center text-sm text-zinc-500">Each model is broken down into weights, KV cache, and headroom.</p>

          <div className="mt-6">
            <label className="text-sm font-medium text-zinc-200" htmlFor="plan-gpu-select">Target GPU</label>
            <select
              id="plan-gpu-select"
              className="mt-2 w-full rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-white outline-none focus:border-cyan-400"
              value=""
              onChange={e => setGpu(allGpus.find(g => g.id === e.target.value) ?? null)}
            >
              <option value="">Choose a GPU…</option>
              {catalogGpus.length > 0 && (
                <optgroup label="Catalog">
                  {catalogGpus.map(g => <option key={g.id} value={g.id}>{g.name}</option>)}
                </optgroup>
              )}
              {customGpus.length > 0 && (
                <optgroup label="Custom">
                  {customGpus.map(g => <option key={g.id} value={g.id}>{g.name} — {g.vram} GB</option>)}
                </optgroup>
              )}
            </select>
          </div>

          {showCustomForm ? (
            <div className="mt-4 rounded-2xl border border-zinc-700 bg-zinc-950 p-4">
              <p className="text-xs font-semibold text-zinc-300">New custom GPU</p>
              <CustomGpuForm onAdd={handleAddCustom} onCancel={() => setShowCustomForm(false)} />
            </div>
          ) : (
            <button
              type="button"
              onClick={() => setShowCustomForm(true)}
              className="mt-3 w-full rounded-xl border border-dashed border-zinc-700 px-3 py-2 text-xs text-zinc-500 transition hover:border-cyan-600 hover:text-cyan-400"
            >
              + Add custom GPU
            </button>
          )}
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
        <KpiCard label="GPU used"   value={`${totalUsedVram.toFixed(1)} GB`}  tooltip={TOOLTIPS.gpuUsed} />
        <KpiCard label="GPU free"   value={`${totalFreeVram.toFixed(1)} GB`}  tone={overflowing ? 'bad' : 'normal'} tooltip={TOOLTIPS.gpuFree} />
        <KpiCard label="RAM spill"  value={`${totalRamSpill.toFixed(1)} GB`}  tone={totalRamSpill > 0 ? 'warn' : 'normal'} tooltip={TOOLTIPS.ramSpill} />
        <KpiCard label="Fit status" value={fitLabel} raw fitColor={fitColor}  tooltip={TOOLTIPS.fitStatus} />
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

            {/* Segmented bar — horizontal */}
            <div className="mt-6 overflow-hidden rounded-3xl border border-zinc-800 bg-zinc-900 p-3">
              <div className="flex h-14 w-full overflow-hidden rounded-2xl bg-zinc-950">
                {partitions.map((p) => {
                  const base    = Math.max(totalUsedVram, gpu.vram)
                  const wPct    = (p.breakdown.weights  / base) * 100
                  const kvPct   = (p.breakdown.kvCache  / base) * 100
                  const hPct    = (p.breakdown.headroom / base) * 100
                  return (
                    <div key={p.id} className="group relative flex h-full min-w-0 flex-row overflow-hidden border-r-2 border-zinc-950 last:border-r-0" style={{ width: `${wPct + kvPct + hPct}%` }}>
                      <Tooltip
                        content={
                          <>
                            <strong className="text-zinc-100">{p.name}</strong>
                            <div className="mt-1.5 space-y-1">
                              <div className="flex justify-between gap-4"><span className="text-zinc-400">Weights</span><span>{p.breakdown.weights.toFixed(1)} GB</span></div>
                              {p.breakdown.kvCache > 0 && <div className="flex justify-between gap-4"><span className="text-zinc-400">KV cache</span><span>{p.breakdown.kvCache.toFixed(1)} GB</span></div>}
                              <div className="flex justify-between gap-4"><span className="text-zinc-400">Headroom</span><span>{p.breakdown.headroom.toFixed(1)} GB</span></div>
                              <div className="flex justify-between gap-4 border-t border-zinc-700 pt-1"><span className="text-zinc-300 font-medium">Total</span><span className="font-semibold">{p.vramAllocated.toFixed(1)} GB</span></div>
                            </div>
                          </>
                        }
                        className="flex h-full w-full"
                      >
                        <div className="flex h-full w-full">
                          <div className="h-full bg-cyan-600"     style={{ width: `${wPct  / (wPct + kvPct + hPct) * 100}%` }} />
                          {p.breakdown.kvCache > 0 && (
                            <div className="h-full bg-blue-600"   style={{ width: `${kvPct / (wPct + kvPct + hPct) * 100}%` }} />
                          )}
                          <div className="h-full bg-amber-700/70" style={{ width: `${hPct  / (wPct + kvPct + hPct) * 100}%` }} />
                        </div>
                      </Tooltip>
                      <span className="pointer-events-none absolute inset-0 flex items-center justify-center text-[10px] font-semibold text-white/70 opacity-0 transition group-hover:opacity-100">
                        {p.name}
                      </span>
                    </div>
                  )
                })}
                {freePercent > 0 && (
                  <Tooltip
                    content={
                      <>
                        <strong className="text-zinc-100">Free VRAM</strong>
                        <div className="mt-1.5 space-y-1">
                          <div className="flex justify-between gap-4"><span className="text-zinc-400">Available</span><span>{totalFreeVram.toFixed(1)} GB</span></div>
                          <div className="flex justify-between gap-4"><span className="text-zinc-400">Total capacity</span><span>{gpu.vram} GB</span></div>
                          <div className="flex justify-between gap-4"><span className="text-zinc-400">Used</span><span>{totalUsedVram.toFixed(1)} GB</span></div>
                        </div>
                      </>
                    }
                    className="flex h-full flex-1 items-center justify-center"
                  >
                    <div className="flex h-full w-full items-center justify-center bg-emerald-500/15 text-xs font-medium text-emerald-200" style={{ width: `${freePercent}%` }}>
                      {freePercent > 8 ? "Free" : ""}
                    </div>
                  </Tooltip>
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
              <SummaryCard label="Capacity"    value={`${gpu.vram} GB`}                          tooltip={TOOLTIPS.capacity} />
              <SummaryCard label="Models"      value={`${partitions.length}`} />
              <SummaryCard label="Utilization" value={`${usedPercent.toFixed(0)}%`}              tone={overflowing ? 'bad' : 'normal'} tooltip={TOOLTIPS.utilization} />
              <SummaryCard label="Allocated"   value={`${totalUsedVram.toFixed(1)} GB`}          tooltip={TOOLTIPS.allocated} />
            </div>
            {partitions.length > 0 && (
              <div className="mt-3 grid grid-cols-3 gap-3">
                <SummaryCard label="Weights"  value={`${totalWeights.toFixed(1)} GB`}   tooltip={TOOLTIPS.weights} />
                <SummaryCard label="KV cache" value={`${totalKv.toFixed(1)} GB`}        tooltip={TOOLTIPS.kvCache} />
                <SummaryCard label="Headroom" value={`${totalHeadroom.toFixed(1)} GB`}  tooltip={TOOLTIPS.headroom} />
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

        {/* Right column: perf shape + recommendations */}
        <div className="flex flex-col gap-5">
          {partitions.length === 0 ? (
            <div className="rounded-3xl border border-dashed border-zinc-800 bg-zinc-950/50 p-7 text-center">
              <p className="text-sm text-zinc-500">Add models to see fit analysis, performance shape, and recommendations.</p>
            </div>
          ) : partitions.map(p => (
            <section key={p.id} className="rounded-3xl border border-zinc-800 bg-zinc-950/80 p-5 md:p-7">
              <p className="text-xs font-semibold uppercase tracking-widest text-zinc-500">{p.name}</p>

              <h4 className="mt-3 text-sm font-semibold text-zinc-100">Performance shape</h4>
              <div className="mt-2 divide-y divide-zinc-800 rounded-2xl border border-zinc-800">
                <PerfRow label="Prefill latency"   value={p.perf.prefill}      tooltip={TOOLTIPS.prefill} />
                <PerfRow label="Token generation"  value={p.perf.tokenGen}     tooltip={TOOLTIPS.tokenGen} />
                <PerfRow label="Risk"              value={p.perf.risk}         tooltip={TOOLTIPS.risk} />
                <PerfRow label="Effective context" value={p.perf.effectiveCtx} tooltip={TOOLTIPS.effectiveCtx} />
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

  // Bar scale: use the larger of total footprint vs gpu capacity so overflow is visible
  const barScale = Math.max(breakdown.total, gpuVram)

  return (
    <section className="rounded-3xl border border-zinc-800 bg-zinc-950/80 p-5 md:p-7">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-base font-semibold text-white">{name}</h3>
        <span className={`rounded-full px-2.5 py-0.5 text-xs font-bold ${fitBadgeColor}`}>{fitBadgeLabel}</span>
      </div>

      {/* Controls: sliders */}
      <div className="mt-4 flex flex-col gap-4">
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <label className="flex items-center gap-1.5 text-xs text-zinc-400">
              Parallel slots
              <InfoTooltip content={TOOLTIPS.parallelSlots} />
            </label>
            <span className="text-xs font-semibold text-zinc-200">{slots}</span>
          </div>
          <input
            type="range"
            min="1"
            max="32"
            step="1"
            aria-label="Parallel slots"
            className="h-2 w-full cursor-pointer appearance-none rounded-full bg-zinc-700 accent-cyan-400"
            value={slots}
            onChange={e => onSlotsChange(parseInt(e.target.value))}
          />
          <div className="relative mx-2 h-3 text-[10px] text-zinc-600">
            <span className="absolute -translate-x-1/2" style={{ left: '0%' }}>1</span>
            <span className="absolute -translate-x-1/2" style={{ left: '22.6%' }}>8</span>
            <span className="absolute -translate-x-1/2" style={{ left: '48.4%' }}>16</span>
            <span className="absolute -translate-x-1/2" style={{ left: '100%' }}>32</span>
          </div>
        </div>

        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <label className="flex items-center gap-1.5 text-xs text-zinc-400">
              Context length
              <InfoTooltip content={TOOLTIPS.contextLength} />
            </label>
            <span className="text-xs font-semibold text-zinc-200">
              {contextLength >= 1024 ? `${(contextLength / 1024).toFixed(0)}k` : contextLength} tokens
            </span>
          </div>
          <input
            type="range"
            min="512"
            max="131072"
            step="512"
            aria-label="Context length"
            className="h-2 w-full cursor-pointer appearance-none rounded-full bg-zinc-700 accent-cyan-400"
            value={contextLength}
            onChange={e => onContextChange(parseInt(e.target.value))}
          />
          <div className="relative mx-2 h-3 text-[10px] text-zinc-600">
            <span className="absolute -translate-x-1/2" style={{ left: '0%' }}>512</span>
            <span className="absolute -translate-x-1/2" style={{ left: '12.1%' }}>16k</span>
            <span className="absolute -translate-x-1/2" style={{ left: '49.7%' }}>64k</span>
            <span className="absolute -translate-x-1/2" style={{ left: '100%' }}>128k</span>
          </div>
        </div>
      </div>

      {/* Segmented bar — horizontal, scaled to model footprint */}
      <div className="mt-4 flex h-5 w-full overflow-hidden rounded-full bg-zinc-800">
        <Tooltip
          content={<>{TOOLTIPS.weights}<div className="mt-1.5 text-zinc-400">{breakdown.weights.toFixed(1)} GB for {name}</div></>}
          className="h-full"
          style={{ width: `${(breakdown.weights / barScale) * 100}%` }}
        >
          <div
            className="h-full w-full bg-cyan-600 cursor-default"
            aria-label={`Weights ${breakdown.weights.toFixed(1)} GB`}
          />
        </Tooltip>
        {breakdown.kvCache > 0 && (
          <Tooltip
            content={<>{TOOLTIPS.kvCache}<div className="mt-1.5 text-zinc-400">{breakdown.kvCache.toFixed(1)} GB across {slots} slot{slots !== 1 ? 's' : ''}</div></>}
            className="h-full"
            style={{ width: `${(breakdown.kvCache / barScale) * 100}%` }}
          >
            <div
              className="h-full w-full bg-blue-600 cursor-default"
              aria-label={`KV cache ${breakdown.kvCache.toFixed(1)} GB`}
            />
          </Tooltip>
        )}
        <Tooltip
          content={<>{TOOLTIPS.headroom}<div className="mt-1.5 text-zinc-400">{breakdown.headroom.toFixed(1)} GB for {name}</div></>}
          className="h-full"
          style={{ width: `${(breakdown.headroom / barScale) * 100}%` }}
        >
          <div
            className="h-full w-full bg-amber-700/70 cursor-default"
            aria-label={`Headroom ${breakdown.headroom.toFixed(1)} GB`}
          />
        </Tooltip>
        {breakdown.overflow > 0 && (
          <div className="h-full bg-red-600/70" style={{ width: `${(breakdown.overflow / barScale) * 100}%` }} aria-label={`Over by ${breakdown.overflow.toFixed(1)} GB`} />
        )}
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
          <div className="flex items-baseline justify-between">
            <p className="flex items-center gap-1.5 text-xs font-semibold text-zinc-400">
              Slot planner
              <InfoTooltip content={TOOLTIPS.slotPlanner} />
            </p>
            <p className="text-xs text-zinc-600">{perSlotKv.toFixed(2)} GB KV per slot</p>
          </div>
          <p className="mt-0.5 text-xs text-zinc-600">Each slot adds its own KV allocation — more slots multiply KV pressure.</p>
          <div className="mt-3 flex h-5 w-full overflow-hidden rounded-full bg-zinc-800">
            {Array.from({ length: slots }).map((_, i) => (
              <Tooltip
                key={i}
                content={
                  <>
                    <strong className="text-zinc-100">Slot {i + 1}</strong>
                    <div className="mt-1 text-zinc-400">{perSlotKv.toFixed(2)} GB KV cache</div>
                    <div className="text-zinc-500 text-[10px]">{((perSlotKv / gpuVram) * 100).toFixed(1)}% of GPU VRAM</div>
                  </>
                }
                className="h-full"
                style={{ width: `${Math.min((perSlotKv / gpuVram) * 100, 100 / slots)}%` }}
              >
                <div
                  className={`h-full w-full border-r border-zinc-950 last:border-r-0 cursor-default ${i % 2 === 0 ? 'bg-blue-600' : 'bg-blue-500'}`}
                  aria-label={`Slot ${i + 1}: ${perSlotKv.toFixed(2)} GB`}
                />
              </Tooltip>
            ))}
          </div>
          <div className="mt-1.5 flex justify-between text-xs text-zinc-600">
            <span>Slot 1</span>
            <span>Slot {slots} · {breakdown.kvCache.toFixed(1)} GB total KV</span>
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

function KpiCard({ label, value, tone, raw, fitColor, tooltip }: { label: string; value: string; tone?: 'normal' | 'bad' | 'warn'; raw?: boolean; fitColor?: string; tooltip?: ReactNode }) {
  const valueClass = raw && fitColor
    ? `mt-2 text-lg font-black ${fitColor.split(' ')[1]}`
    : `mt-2 text-lg font-black ${tone === 'bad' ? 'text-red-300' : tone === 'warn' ? 'text-amber-400' : 'text-white'}`
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-4">
      <p className="flex items-center gap-1.5 text-xs uppercase tracking-widest text-zinc-500">
        {label}
        {tooltip && <InfoTooltip content={tooltip} />}
      </p>
      <p className={valueClass}>{value}</p>
    </div>
  )
}

function SummaryCard({ label, value, tone, tooltip }: { label: string; value: string; tone?: 'normal' | 'bad'; tooltip?: ReactNode }) {
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/80 p-4">
      <p className="flex items-center gap-1.5 text-xs uppercase tracking-widest text-zinc-500">
        {label}
        {tooltip && <InfoTooltip content={tooltip} />}
      </p>
      <p className={`mt-2 text-lg font-semibold ${tone === 'bad' ? 'text-red-300' : 'text-white'}`}>{value}</p>
    </div>
  )
}

function PerfRow({ label, value, tooltip }: { label: string; value: string; tooltip?: ReactNode }) {
  const isGood    = ['Fast', 'Low'].includes(value)
  const isBad     = ['Slow', 'Blocked', 'Does not fit', 'High'].includes(value)
  const isWarn    = ['Moderate'].includes(value)
  const valClass  = isBad ? 'text-red-400' : isWarn ? 'text-amber-400' : isGood ? 'text-emerald-400' : 'text-zinc-300'
  return (
    <div className="flex items-center justify-between gap-3 px-3 py-2.5">
      <span className="flex items-center gap-1.5 text-xs text-zinc-500">
        {label}
        {tooltip && <InfoTooltip content={tooltip} />}
      </span>
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
