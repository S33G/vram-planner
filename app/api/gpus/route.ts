import { execFile } from "node:child_process"
import { promisify } from "node:util"
import { NextResponse } from "next/server"

const execFileAsync = promisify(execFile)

export const dynamic = "force-dynamic"

export async function GET() {
  try {
    const { stdout } = await execFileAsync(
      "nvidia-smi",
      [
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
      ],
      { timeout: 3000 },
    )

    const gpus = stdout
      .trim()
      .split("\n")
      .filter(Boolean)
      .map((line) => {
        const [index, name, total, used, free, utilization, temperature] = line.split(",").map((value) => value.trim())
        return {
          index: Number(index),
          name,
          memoryTotalMiB: Number(total),
          memoryUsedMiB: Number(used),
          memoryFreeMiB: Number(free),
          utilizationGpu: parseOptionalNumber(utilization),
          temperature: parseOptionalNumber(temperature),
        }
      })

    return NextResponse.json({
      available: gpus.length > 0,
      gpus,
      updatedAt: new Date().toISOString(),
    })
  } catch (error) {
    return NextResponse.json({
      available: false,
      gpus: [],
      error: error instanceof Error ? error.message : "nvidia-smi is not available",
      updatedAt: new Date().toISOString(),
    })
  }
}

function parseOptionalNumber(value: string) {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}
