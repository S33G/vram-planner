import { test, expect, type Page } from '@playwright/test'

// ---- helpers ----

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

/** Hover an element and assert the tooltip role becomes visible with expected text. */
async function expectTooltip(
  page: Page,
  trigger: Parameters<Page['locator']>[0],
  expectedText: string | RegExp,
) {
  await page.locator(trigger).hover()
  const tooltip = page.locator('[role="tooltip"]')
  await expect(tooltip).toBeVisible({ timeout: 3000 })
  await expect(tooltip).toContainText(expectedText)
}

// ---- tests ----

test.describe('InfoTooltip — ? icon on slider labels', () => {
  test.beforeEach(async ({ page }) => { await setupWithModel(page) })

  test('Parallel slots ? shows concept explanation', async ({ page }) => {
    // The ? is inside the label element for the parallel slots slider
    const icon = page.locator('label').filter({ hasText: 'Parallel slots' })
      .getByLabel('More information')
    await icon.scrollIntoViewIfNeeded()
    await icon.hover()
    const tooltip = page.locator('[role="tooltip"]')
    await expect(tooltip).toBeVisible({ timeout: 3000 })
    await expect(tooltip).toContainText('Parallel slots')
    await expect(tooltip).toContainText('KV cache')
    await expect(tooltip).toContainText('concurrent')
  })

  test('Context length ? shows concept explanation', async ({ page }) => {
    const icon = page.locator('label').filter({ hasText: 'Context length' })
      .getByLabel('More information')
    await icon.scrollIntoViewIfNeeded()
    await icon.hover()
    const tooltip = page.locator('[role="tooltip"]')
    await expect(tooltip).toBeVisible({ timeout: 3000 })
    await expect(tooltip).toContainText('Context length')
    await expect(tooltip).toContainText('tokens')
    await expect(tooltip).toContainText('KV cache')
  })

  test('Slot planner ? shows explanation', async ({ page }) => {
    const icon = page.locator('p').filter({ hasText: 'Slot planner' })
      .getByLabel('More information')
    await icon.scrollIntoViewIfNeeded()
    await icon.hover()
    const tooltip = page.locator('[role="tooltip"]')
    await expect(tooltip).toBeVisible({ timeout: 3000 })
    await expect(tooltip).toContainText('Slot planner')
    await expect(tooltip).toContainText('KV cache')
  })
})

test.describe('InfoTooltip — ? icon on KPI cards', () => {
  test.beforeEach(async ({ page }) => { await setupWithModel(page) })

  test('GPU used card has ? that explains allocation', async ({ page }) => {
    const icon = page.locator('p').filter({ hasText: /GPU used/i })
      .getByLabel('More information')
    await icon.hover()
    const tooltip = page.locator('[role="tooltip"]')
    await expect(tooltip).toBeVisible({ timeout: 3000 })
    await expect(tooltip).toContainText('GPU used')
    await expect(tooltip).toContainText('VRAM')
  })

  test('GPU free card has ? explaining free memory', async ({ page }) => {
    const icon = page.locator('p').filter({ hasText: /GPU free/i })
      .getByLabel('More information')
    await icon.hover()
    const tooltip = page.locator('[role="tooltip"]')
    await expect(tooltip).toBeVisible({ timeout: 3000 })
    await expect(tooltip).toContainText('GPU free')
  })

  test('RAM spill card has ? explaining RAM offload', async ({ page }) => {
    const icon = page.locator('p').filter({ hasText: /RAM spill/i })
      .getByLabel('More information')
    await icon.hover()
    const tooltip = page.locator('[role="tooltip"]')
    await expect(tooltip).toBeVisible({ timeout: 3000 })
    await expect(tooltip).toContainText('RAM spill')
    await expect(tooltip).toContainText('bandwidth')
  })

  test('Fit status card has ? explaining fit categories', async ({ page }) => {
    const icon = page.locator('p').filter({ hasText: /Fit status/i })
      .getByLabel('More information')
    await icon.hover()
    const tooltip = page.locator('[role="tooltip"]')
    await expect(tooltip).toBeVisible({ timeout: 3000 })
    await expect(tooltip).toContainText('Fit status')
    await expect(tooltip).toContainText('Spills to RAM')
  })
})

test.describe('InfoTooltip — ? icon on summary cards', () => {
  test.beforeEach(async ({ page }) => { await setupWithModel(page) })

  test('Weights card has ? explaining model weights', async ({ page }) => {
    // Target the tracking-widest label p that contains "Weights" (not sidebar breakdown text)
    const icon = page.locator('p.tracking-widest').filter({ hasText: 'Weights' })
      .getByLabel('More information')
    await icon.first().scrollIntoViewIfNeeded()
    await icon.first().hover()
    const tooltip = page.locator('[role="tooltip"]')
    await expect(tooltip).toBeVisible({ timeout: 3000 })
    await expect(tooltip).toContainText('Weights')
    await expect(tooltip).toContainText('model parameters')
  })

  test('KV cache card has ? with formula explanation', async ({ page }) => {
    const icon = page.locator('p.tracking-widest').filter({ hasText: 'KV cache' })
      .getByLabel('More information')
    await icon.first().scrollIntoViewIfNeeded()
    await icon.first().hover()
    const tooltip = page.locator('[role="tooltip"]')
    await expect(tooltip).toBeVisible({ timeout: 3000 })
    await expect(tooltip).toContainText('KV cache')
    await expect(tooltip).toContainText('ctx')
  })

  test('Headroom card has ? explaining overhead budget', async ({ page }) => {
    const icon = page.locator('p.tracking-widest').filter({ hasText: 'Headroom' })
      .getByLabel('More information')
    await icon.first().scrollIntoViewIfNeeded()
    await icon.first().hover()
    const tooltip = page.locator('[role="tooltip"]')
    await expect(tooltip).toBeVisible({ timeout: 3000 })
    await expect(tooltip).toContainText('Headroom')
    await expect(tooltip).toContainText('overhead')
  })
})

test.describe('InfoTooltip — ? icon on performance shape rows', () => {
  test.beforeEach(async ({ page }) => { await setupWithModel(page) })

  test('Prefill latency row has ? explaining prefill', async ({ page }) => {
    const icon = page.locator('span').filter({ hasText: 'Prefill latency' })
      .getByLabel('More information')
    await icon.scrollIntoViewIfNeeded()
    await icon.hover()
    const tooltip = page.locator('[role="tooltip"]')
    await expect(tooltip).toBeVisible({ timeout: 3000 })
    await expect(tooltip).toContainText('Prefill latency')
    await expect(tooltip).toContainText('prompt')
  })

  test('Token generation row has ? explaining bandwidth bound', async ({ page }) => {
    const icon = page.locator('span').filter({ hasText: 'Token generation' })
      .getByLabel('More information')
    await icon.scrollIntoViewIfNeeded()
    await icon.hover()
    const tooltip = page.locator('[role="tooltip"]')
    await expect(tooltip).toBeVisible({ timeout: 3000 })
    await expect(tooltip).toContainText('Token generation')
    await expect(tooltip).toContainText('bandwidth')
  })

  test('Risk row has ? explaining VRAM pressure', async ({ page }) => {
    const icon = page.locator('span').filter({ hasText: /^Risk\?$/ })
      .getByLabel('More information')
    await icon.scrollIntoViewIfNeeded()
    await icon.hover()
    const tooltip = page.locator('[role="tooltip"]')
    await expect(tooltip).toBeVisible({ timeout: 3000 })
    await expect(tooltip).toContainText('Risk')
    await expect(tooltip).toContainText('VRAM')
  })

  test('Effective context row has ? explaining KV budget', async ({ page }) => {
    const icon = page.locator('span').filter({ hasText: 'Effective context' })
      .getByLabel('More information')
    await icon.scrollIntoViewIfNeeded()
    await icon.hover()
    const tooltip = page.locator('[role="tooltip"]')
    await expect(tooltip).toBeVisible({ timeout: 3000 })
    await expect(tooltip).toContainText('Effective context')
  })
})

test.describe('Hover tooltips — VRAM bar segments', () => {
  test.beforeEach(async ({ page }) => { await setupWithModel(page) })

  test('Hovering weights segment shows model name and GB', async ({ page }) => {
    // The weights segment has aria-label "Weights X.X GB"
    const seg = page.locator('[aria-label^="Weights"]').first()
    await seg.scrollIntoViewIfNeeded()
    await seg.hover()
    const tooltip = page.locator('[role="tooltip"]')
    await expect(tooltip).toBeVisible({ timeout: 3000 })
    await expect(tooltip).toContainText('Weights')
    await expect(tooltip).toContainText('GB')
  })

  test('Hovering KV cache segment shows slots and GB', async ({ page }) => {
    const seg = page.locator('[aria-label^="KV cache"]').first()
    await seg.scrollIntoViewIfNeeded()
    await seg.hover()
    const tooltip = page.locator('[role="tooltip"]')
    await expect(tooltip).toBeVisible({ timeout: 3000 })
    await expect(tooltip).toContainText('KV cache')
    await expect(tooltip).toContainText('GB')
  })

  test('Hovering headroom segment shows GB', async ({ page }) => {
    const seg = page.locator('[aria-label^="Headroom"]').first()
    await seg.scrollIntoViewIfNeeded()
    await seg.hover()
    const tooltip = page.locator('[role="tooltip"]')
    await expect(tooltip).toBeVisible({ timeout: 3000 })
    await expect(tooltip).toContainText('Headroom')
    await expect(tooltip).toContainText('GB')
  })
})

test.describe('Hover tooltips — slot planner bars', () => {
  test.beforeEach(async ({ page }) => { await setupWithModel(page) })

  test('Hovering slot 1 bar shows slot number and KV size', async ({ page }) => {
    const slotBar = page.locator('[aria-label="Slot 1:"]', { hasText: /GB/ }).or(
      page.locator('[aria-label^="Slot 1:"]'),
    ).first()
    await slotBar.scrollIntoViewIfNeeded()
    await slotBar.hover()
    const tooltip = page.locator('[role="tooltip"]')
    await expect(tooltip).toBeVisible({ timeout: 3000 })
    await expect(tooltip).toContainText('Slot 1')
    await expect(tooltip).toContainText('GB')
  })

  test('Hovering slot bar after increasing slots shows correct slot number', async ({ page }) => {
    // Set slots to 4 via the range input (React requires nativeInputValueSetter trick)
    await page.locator('input[aria-label="Parallel slots"]').evaluate(
      (el: HTMLInputElement) => {
        const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')?.set
        nativeInputValueSetter?.call(el, '4')
        el.dispatchEvent(new Event('input', { bubbles: true }))
        el.dispatchEvent(new Event('change', { bubbles: true }))
      },
    )
    await page.waitForTimeout(200)

    const slotBar = page.locator('[aria-label^="Slot 4:"]').first()
    await slotBar.scrollIntoViewIfNeeded()
    await slotBar.hover()
    const tooltip = page.locator('[role="tooltip"]')
    await expect(tooltip).toBeVisible({ timeout: 3000 })
    await expect(tooltip).toContainText('Slot 4')
  })
})

test.describe('Hover tooltips — main VRAM overview bar', () => {
  test.beforeEach(async ({ page }) => { await setupWithModel(page) })

  test('Hovering the model partition shows breakdown', async ({ page }) => {
    // The top-level VRAM overview bar: first direct child is the model partition Tooltip wrapper
    const bar = page.locator('.h-14').first()
    // The Tooltip wrapper is the first child element of the bar
    const partitionSegment = bar.locator('> *').first()
    await partitionSegment.scrollIntoViewIfNeeded()
    await partitionSegment.hover()
    const tooltip = page.locator('[role="tooltip"]')
    await expect(tooltip).toBeVisible({ timeout: 3000 })
    await expect(tooltip).toContainText('GB')
  })
})

test.describe('Tooltip accessibility', () => {
  test.beforeEach(async ({ page }) => { await setupWithModel(page) })

  test('Tooltip disappears when mouse leaves the trigger', async ({ page }) => {
    const icon = page.locator('label').filter({ hasText: 'Parallel slots' })
      .getByLabel('More information')
    await icon.scrollIntoViewIfNeeded()
    await icon.hover()
    await expect(page.locator('[role="tooltip"]')).toBeVisible({ timeout: 3000 })

    // Move mouse away to body
    await page.mouse.move(10, 10)
    await expect(page.locator('[role="tooltip"]')).toBeHidden({ timeout: 2000 })
  })

  test('Only one tooltip is visible at a time', async ({ page }) => {
    const icons = page.locator('span[aria-label="More information"]')

    // Hover first, then second
    await icons.nth(0).hover()
    await expect(page.locator('[role="tooltip"]')).toHaveCount(1, { timeout: 3000 })

    await icons.nth(1).hover()
    await expect(page.locator('[role="tooltip"]')).toHaveCount(1)
  })
})
