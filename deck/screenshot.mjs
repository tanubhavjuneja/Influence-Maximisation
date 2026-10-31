// Screenshot every slide at 1280x720 using the system Edge browser
// (no browser download needed). Run: node screenshot.mjs  (after `npm run build` + preview)
import { chromium } from 'playwright'
import { spawn } from 'node:child_process'
import { mkdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = dirname(fileURLToPath(import.meta.url))
const out = join(root, 'screens')
mkdirSync(out, { recursive: true })

const server = spawn('npx', ['vite', 'preview', '--port', '4173', '--strictPort'], {
  cwd: root, shell: true, stdio: 'ignore',
})

const wait = (ms) => new Promise((r) => setTimeout(r, ms))
await wait(1500)

const browser = await chromium.launch({ channel: 'msedge', headless: true })
const page = await browser.newPage({ viewport: { width: 1312, height: 806 } })

for (let i = 0; i < 16; i++) {
  await page.goto(`http://localhost:4173/#${i}`, { waitUntil: 'networkidle' })
  await wait(400)
  const el = page.locator('.slide')
  await el.screenshot({ path: join(out, `slide-${String(i + 1).padStart(2, '0')}.png`) })
  console.log('saved slide', i + 1)
}

await browser.close()
server.kill()