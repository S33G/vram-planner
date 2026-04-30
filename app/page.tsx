"use client"

import { useState, useCallback } from "react"
import { Sidebar } from "@/components/sidebar"
import { VramVisualization } from "@/components/vram-visualization"
import type { GpuSpec, ModelPreset, Partition } from "@/lib/db"

interface LoadedModel extends ModelPreset {
  instanceId: string
  vramAllocated: number
}

export default function Home() {
  const [gpu, setGpu] = useState<GpuSpec | null>(null)
  const [loadedModels, setLoadedModels] = useState<LoadedModel[]>([])
  const [filter, setFilter] = useState("")
  const [familyFilter, setFamilyFilter] = useState<string>("all")

  const addModel = useCallback((model: ModelPreset) => {
    if (gpu) {
      const overhead = gpu.vram * 0.05
      const vramAllocated = Math.min(model.vramRequired, gpu.vram - overhead)
      const instanceId = `${model.id}-${Date.now()}`
      setLoadedModels(prev => [...prev, { ...model, instanceId, vramAllocated }])
    }
  }, [gpu])

  const removeModel = useCallback((instanceId: string) => {
    setLoadedModels(prev => prev.filter(m => m.instanceId !== instanceId))
  }, [])

  const totalUsedVram = loadedModels.reduce((sum, m) => sum + m.vramAllocated, 0)
  const reservedVram = gpu ? gpu.vram * 0.05 : 0
  const usableVram = gpu ? gpu.vram - reservedVram : 0
  const totalFreeVram = usableVram - totalUsedVram

  const partitions: Partition[] = loadedModels.map((m, idx) => ({
    id: m.id,
    name: m.name,
    modelId: m.id,
    slot: idx,
    vramAllocated: m.vramAllocated,
    performance: totalUsedVram > usableVram ? 'infeasible' : 'fast'
  }))

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
      />
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <VramVisualization
          gpu={gpu}
          partitions={partitions}
          totalUsedVram={totalUsedVram}
          totalFreeVram={totalFreeVram}
        />
      </main>
    </div>
  )
}
