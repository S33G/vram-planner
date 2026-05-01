import { test, expect, type Page } from '@playwright/test'

async function setupWithModel(page: Page) {
  await page.goto('/')
  await page.locator('#gpu-select').selectOption('rtx4090-24')
  await page.evaluate(() => {
    const btn = Array.from(document.querySelectorAll('button')).find(b =>
      b.textContent?.includes('Llama 3.1 8B (Q4_K_M)'),
    )
    if (!btn) throw new Error('Model button not found')
    btn.scrollIntoView({ block: 'center' })
    btn.click()
  })
  await page.waitForFunction(
    () => document.querySelectorAll('input[type="range"]').length >= 2,
  )
}

test('free segment fills the remaining space of the VRAM bar', async ({ page }) => {
  await setupWithModel(page)

  // The main VRAM overview bar.
  const bar = page.locator('.h-14').first()
  await expect(bar).toBeVisible()
  const barBox = await bar.boundingBox()
  expect(barBox).not.toBeNull()

  // Direct children of the bar: partition wrappers + the Free tooltip wrapper (last).
  const children = bar.locator('> *')
  const count = await children.count()
  expect(count).toBeGreaterThan(1)

  // Sum widths of model partition wrappers (all but the last child).
  let modelWidth = 0
  for (let i = 0; i < count - 1; i++) {
    const box = await children.nth(i).boundingBox()
    if (box) modelWidth += box.width
  }

  // The Free wrapper is the last child.
  const freeWrapper = children.nth(count - 1)
  const freeWrapperBox = await freeWrapper.boundingBox()
  expect(freeWrapperBox).not.toBeNull()

  // The visible green fill inside the wrapper.
  const freeFill = freeWrapper.locator('div.bg-emerald-500\\/15').first()
  const freeFillBox = await freeFill.boundingBox()
  expect(freeFillBox).not.toBeNull()

  console.log('bar width', barBox!.width)
  console.log('model widths total', modelWidth)
  console.log('free wrapper width', freeWrapperBox!.width)
  console.log('free fill (bg-emerald) width', freeFillBox!.width)
  console.log('expected free width', barBox!.width - modelWidth)

  // The green fill should fully occupy the free region of the bar
  // (i.e. match the wrapper width, which equals bar width minus model widths).
  expect(freeFillBox!.width).toBeGreaterThan(freeWrapperBox!.width - 1)
})
