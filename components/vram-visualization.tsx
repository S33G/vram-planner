"use client"

import { useEffect, useState } from "react"
import type { ReactNode } from "react"
import type { GpuSpec, Partition } from "@/lib/db"

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
}

export function VramVisualization({ gpu, partitions, totalUsedVram, totalFreeVram }: VramVisualizationProps) {
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
        if (!cancelled) {
          setLiveState({
            available: false,
            gpus: [],
            error: error instanceof Error ? error.message : "Unable to load GPU state",
            updatedAt: new Date().toISOString(),
          })
        }
      } finally {
        if (!cancelled) setIsLoadingLive(false)
      }
    }

    loadGpuState()
    const interval = window.setInterval(loadGpuState, 5000)

    return () => {
      cancelled = true
      window.clearInterval(interval)
    }
  }, [view])

  return (
    <div className="flex h-dvh flex-1 flex-col overflow-y-auto bg-[radial-gradient(circle_at_top_right,_rgba(34,211,238,0.16),_transparent_28rem),#09090b] p-5 md:p-8">
      <div className="flex flex-col justify-between gap-4 md:flex-row md:items-start">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.24em] text-zinc-500">Machine View</p>
          <h2 className="mt-2 text-3xl font-semibold tracking-tight text-white md:text-5xl">
            {view === "plan" ? "Simulated VRAM allocation" : "Live NVIDIA memory state"}
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
        <PlanView gpu={gpu} partitions={partitions} totalUsedVram={totalUsedVram} totalFreeVram={totalFreeVram} />
      ) : (
        <LiveView state={liveState} isLoading={isLoadingLive} />
      )}
    </div>
  )
}

function PlanView({ gpu, partitions, totalUsedVram, totalFreeVram }: VramVisualizationProps) {
  if (!gpu) {
    return (
      <div className="mt-10 flex flex-1 items-center justify-center rounded-3xl border border-dashed border-zinc-800 bg-zinc-950/50 p-10 text-center">
        <div>
          <p className="text-lg font-medium text-white">Select a GPU to start planning.</p>
          <p className="mt-2 text-sm text-zinc-500">The planner reserves 5% overhead and stacks model allocations into the remaining VRAM.</p>
        </div>
      </div>
    )
  }

  const overhead = gpu.vram * 0.05
  const usedPercent = Math.min((totalUsedVram / gpu.vram) * 100, 100)
  const overheadPercent = (overhead / gpu.vram) * 100
  const freePercent = Math.max((totalFreeVram / gpu.vram) * 100, 0)

  return (
    <div className="mt-8 grid flex-1 gap-5 xl:grid-cols-[1.35fr_0.65fr]">
      <section className="rounded-3xl border border-zinc-800 bg-zinc-950/80 p-5 md:p-7">
        <div className="flex flex-col justify-between gap-4 md:flex-row md:items-end">
          <div>
            <h3 className="text-xl font-semibold text-white">{gpu.name}</h3>
            <p className="mt-1 text-sm text-zinc-500">{gpu.vram} GB VRAM / {gpu.memoryBandwidth} GB/s bandwidth / {gpu.tdp} W TDP</p>
          </div>
          <p className={`text-sm font-semibold ${totalFreeVram < 0 ? "text-red-300" : "text-emerald-300"}`}>
            {totalFreeVram.toFixed(1)} GB free
          </p>
        </div>

        <div className="mt-8 overflow-hidden rounded-3xl border border-zinc-800 bg-zinc-900 p-3">
          <div className="flex h-28 overflow-hidden rounded-2xl bg-zinc-950">
            {partitions.map((partition, index) => {
              const width = Math.max((partition.vramAllocated / gpu.vram) * 100, 2)
              return (
                <div
                  key={`${partition.id}-${index}`}
                  className="flex min-w-8 items-center justify-center border-r border-zinc-950 bg-cyan-500 px-2 text-center text-xs font-semibold text-zinc-950"
                  style={{ width: `${width}%` }}
                  title={`${partition.name}: ${partition.vramAllocated.toFixed(1)} GB`}
                >
                  <span className="line-clamp-2">{partition.name}</span>
                </div>
              )
            })}
            <div className="bg-zinc-700/70" style={{ width: `${overheadPercent}%` }} title={`Reserved overhead: ${overhead.toFixed(1)} GB`} />
            <div className="flex items-center justify-center bg-emerald-500/15 text-xs font-medium text-emerald-200" style={{ width: `${freePercent}%` }}>
              {freePercent > 10 ? "Free" : ""}
            </div>
          </div>
        </div>

        <div className="mt-6 grid gap-3 md:grid-cols-4">
          <SummaryCard label="Capacity" value={`${gpu.vram} GB`} />
          <SummaryCard label="Models" value={`${partitions.length}`} />
          <SummaryCard label="Allocated" value={`${totalUsedVram.toFixed(1)} GB`} />
          <SummaryCard label="Utilization" value={`${usedPercent.toFixed(0)}%`} tone={usedPercent > 100 ? "bad" : "normal"} />
        </div>
      </section>

      <section className="rounded-3xl border border-zinc-800 bg-zinc-950/80 p-5 md:p-7">
        <h3 className="text-lg font-semibold text-white">Fit Analysis</h3>
        <div className="mt-5 space-y-3">
          {partitions.length === 0 ? (
            <p className="text-sm text-zinc-500">Add models to see per-model allocation and fit status.</p>
          ) : partitions.map((partition, index) => (
            <div key={`${partition.id}-${partition.slot}-${index}`} className="rounded-2xl border border-zinc-800 bg-zinc-900/70 p-4">
              <div className="flex items-center justify-between gap-3">
                <p className="text-sm font-medium text-white">{partition.name}</p>
                <span className="text-sm text-cyan-200">{partition.vramAllocated.toFixed(1)} GB</span>
              </div>
              <div className="mt-3 h-2 overflow-hidden rounded-full bg-zinc-800">
                <div className="h-full rounded-full bg-cyan-400" style={{ width: `${Math.min((partition.vramAllocated / gpu.vram) * 100, 100)}%` }} />
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}

function LiveView({ state, isLoading }: { state: LiveGpuResponse | null; isLoading: boolean }) {
  if (!state && isLoading) {
    return <StatusPanel title="Checking host GPUs" description="Running nvidia-smi from the server container..." />
  }

  if (!state?.available) {
    return (
      <StatusPanel
        title="No live NVIDIA state available"
        description={state?.error ?? "nvidia-smi was not found or no GPUs were reported. Run the container with NVIDIA runtime support to enable this view."}
      />
    )
  }

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
              <SummaryCard label="Used" value={`${mibToGib(gpu.memoryUsedMiB)} GiB`} />
              <SummaryCard label="Free" value={`${mibToGib(gpu.memoryFreeMiB)} GiB`} />
              <SummaryCard label="Total" value={`${mibToGib(gpu.memoryTotalMiB)} GiB`} />
              <SummaryCard label="GPU util" value={gpu.utilizationGpu == null ? "n/a" : `${gpu.utilizationGpu}%`} />
            </div>
            <p className="mt-5 text-xs text-zinc-500">Updated {new Date(state.updatedAt).toLocaleTimeString()}</p>
          </section>
        )
      })}
    </div>
  )
}

function ToggleButton({ active, children, onClick }: { active: boolean; children: ReactNode; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-xl px-4 py-2 text-sm font-medium transition ${active ? "bg-cyan-400 text-zinc-950" : "text-zinc-400 hover:text-white"}`}
    >
      {children}
    </button>
  )
}

function SummaryCard({ label, value, tone }: { label: string; value: string; tone?: "normal" | "bad" }) {
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/80 p-4">
      <p className="text-xs uppercase tracking-widest text-zinc-500">{label}</p>
      <p className={`mt-2 text-lg font-semibold ${tone === "bad" ? "text-red-300" : "text-white"}`}>{value}</p>
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

function mibToGib(value: number) {
  return (value / 1024).toFixed(1)
}
