"use client"

import { useState, useCallback } from "react"
import { Sidebar } from "@/components/sidebar"
import { VramVisualization } from "@/components/vram-visualization"
import {
  calculateVramBreakdown,
  calculatePerformance,
  calcPerfShape,
  calcRecommendations,
} from "@/lib/db"
import type { GpuSpec, ModelPreset, Partition, SpillPolicy } from "@/lib/db"

interface LoadedModel extends ModelPreset {
  instanceId: string
  slots: number
  contextLength: number  // user-overridable
}

export default function Home() {
  const [gpu, setGpu]                 = useState<GpuSpec | null>(null)
  const [loadedModels, setLoadedModels] = useState<LoadedModel[]>([])
  const [filter, setFilter]           = useState("")
  const [familyFilter, setFamilyFilter] = useState<string>("all")
  const [systemRamGb, setSystemRamGb] = useState(64)
  const [spillPolicy, setSpillPolicy] = useState<SpillPolicy>("avoid")

  const addModel = useCallback((model: ModelPreset) => {
    if (!gpu) return
    const instanceId = `${model.id}-${Date.now()}`
    setLoadedModels(prev => [...prev, {
      ...model,
      instanceId,
      slots: 1,
      contextLength: model.contextLength,
    }])
  }, [gpu])

  const removeModel = useCallback((instanceId: string) => {
    setLoadedModels(prev => prev.filter(m => m.instanceId !== instanceId))
  }, [])

  const updateModelSlots = useCallback((instanceId: string, slots: number) => {
    setLoadedModels(prev => prev.map(m => m.instanceId === instanceId ? { ...m, slots } : m))
  }, [])

  const updateModelContext = useCallback((instanceId: string, contextLength: number) => {
    setLoadedModels(prev => prev.map(m => m.instanceId === instanceId ? { ...m, contextLength } : m))
  }, [])

  const partitions: Partition[] = gpu
    ? loadedModels.map((m, idx) => {
        const breakdown = calculateVramBreakdown(m, gpu.vram, {
          slots: m.slots,
          contextLength: m.contextLength,
          systemRamGb,
          spillPolicy,
        })
        return {
          id: m.instanceId,
          name: m.name,
          modelId: m.id,
          slot: idx,
          slots: m.slots,
          contextLength: m.contextLength,
          vramAllocated: breakdown.gpuUsed,
          breakdown,
          perf: calcPerfShape(breakdown, m.contextLength, m.slots),
          recommendations: calcRecommendations(breakdown, m.contextLength, m.slots, gpu.vram, systemRamGb),
          performance: calculatePerformance(breakdown, gpu.vram),
        }
      })
    : []

  const totalUsedVram  = partitions.reduce((s, p) => s + p.breakdown.gpuUsed, 0)
  const totalFreeVram  = gpu ? Math.max(0, gpu.vram - totalUsedVram) : 0
  const totalRamSpill  = partitions.reduce((s, p) => s + p.breakdown.ramSpill, 0)

  return (
    <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
      <Sidebar
        gpu={gpu}
        setGpu={setGpu}
        filter={filter}
        setFilter={setFilter}
        familyFilter={familyFilter}
        setFamilyFilter={setFamilyFilter}
        onAddModel={addModel}
        loadedModels={loadedModels}
        removeModel={removeModel}
        totalUsedVram={totalUsedVram}
        totalFreeVram={totalFreeVram}
        systemRamGb={systemRamGb}
        setSystemRamGb={setSystemRamGb}
        spillPolicy={spillPolicy}
        setSpillPolicy={setSpillPolicy}
      />
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <VramVisualization
          gpu={gpu}
          partitions={partitions}
          totalUsedVram={totalUsedVram}
          totalFreeVram={totalFreeVram}
          totalRamSpill={totalRamSpill}
          systemRamGb={systemRamGb}
          spillPolicy={spillPolicy}
          updateModelSlots={updateModelSlots}
          updateModelContext={updateModelContext}
        />
      </main>
    </div>
  )
}
