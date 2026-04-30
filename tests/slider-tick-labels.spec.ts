import { test, expect } from '@playwright/test'

/**
 * Returns the expected left-percentage for a value on a linear slider.
 *   left% = (value - min) / (max - min) * 100
 */
function sliderPct(value: number, min: number, max: number): number {
  return ((value - min) / (max - min)) * 100
}

test.describe('Slider tick label alignment', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')

    // Select a GPU via its ID value so models become clickable
    await page.locator('#gpu-select').selectOption('rtx4090-24')

    // Scroll the model button into view inside the sidebar and click it via JS
    // (the sidebar is a scrollable overlay that blocks Playwright pointer events)
    await page.evaluate(() => {
      const btn = Array.from(document.querySelectorAll('button')).find(b =>
        b.textContent?.includes('Llama 3.1 8B (Q4_K_M)'),
      )
      if (!btn) throw new Error('Model button not found')
      btn.scrollIntoView({ block: 'center' })
      btn.click()
    })

    // Wait for the sliders to appear in the main panel
    await page.waitForFunction(() => document.querySelectorAll('input[type="range"]').length >= 2)
  })

  /**
   * For each tick, assert its rendered centre X is within `tolerancePx` pixels
   * of where it should fall on the slider track.
   */
  async function assertTickAlignment(
    page: Parameters<typeof test>[1] extends { page: infer P } ? P : never,
    sliderIndex: number,
    min: number,
    max: number,
    ticks: { text: string; value: number }[],
    tolerancePx = 4,
  ) {
    const result = await page.evaluate(
      ({ sliderIndex, min, max, ticks }) => {
        const sliders = document.querySelectorAll('input[type="range"]')
        const slider = sliders[sliderIndex]
        if (!slider) return { error: `No slider at index ${sliderIndex}` }

        const tickContainer = slider.nextElementSibling
        if (!tickContainer) return { error: 'No tick container found after slider' }

        const cBox = tickContainer.getBoundingClientRect()

        return ticks.map(tick => {
          const span = Array.from(tickContainer.querySelectorAll('span')).find(
            s => s.textContent?.trim() === tick.text,
          )
          if (!span) return { text: tick.text, error: 'span not found' }

          const sb = span.getBoundingClientRect()
          const labelCenterX = sb.left + sb.width / 2
          const expectedX = cBox.left + (((tick.value - min) / (max - min)) * cBox.width)

          return {
            text: tick.text,
            labelCenterX,
            expectedX,
            diff: Math.abs(labelCenterX - expectedX),
          }
        })
      },
      { sliderIndex, min, max, ticks },
    )

    if (!Array.isArray(result)) {
      throw new Error((result as { error: string }).error)
    }

    for (const r of result) {
      if ('error' in r) throw new Error(`Tick "${r.text}": ${r.error}`)
      expect(
        r.diff,
        `Tick "${r.text}" center (${r.labelCenterX.toFixed(1)}px) should be within ${tolerancePx}px of expected (${r.expectedX.toFixed(1)}px)`,
      ).toBeLessThanOrEqual(tolerancePx)
    }
  }

  test('slots slider tick labels align with track positions', async ({ page }) => {
    // Slots slider: min=1, max=32, ticks at 1, 8, 16, 32
    await assertTickAlignment(page, 0, 1, 32, [
      { text: '1',  value: 1 },
      { text: '8',  value: 8 },
      { text: '16', value: 16 },
      { text: '32', value: 32 },
    ])
  })

  test('context length slider tick labels align with track positions', async ({ page }) => {
    // Context slider: min=512, max=131072, ticks at 512, 16384, 65536, 131072
    await assertTickAlignment(page, 1, 512, 131072, [
      { text: '512',  value: 512 },
      { text: '16k',  value: 16384 },
      { text: '64k',  value: 65536 },
      { text: '128k', value: 131072 },
    ])
  })
})
