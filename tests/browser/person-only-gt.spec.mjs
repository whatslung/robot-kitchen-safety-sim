import { test, expect } from '@playwright/test';

const BASE_URL = process.env.BASE_URL || 'http://localhost:5199';
test.setTimeout(120000);

test.beforeEach(async ({ page }) => {
  await page.route('**/predict', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ tracks: [], worst: null }),
  }));
  await page.goto(`${BASE_URL}/sim.html?layout=island&scenario=none&v=person-only-gt-test`);
  await page.waitForFunction(() => window.__simReady === true, null, { timeout: 60000 });
});

test('ground truth exposes only class 0 person labels', async ({ page }) => {
  const result = await page.evaluate(async () => {
    await window.__sim.setExtraCount(0);
    window.__sim.setFire(false);
    window.__sim.personWander(false);
    // Keep the worker centered in cvN so this test checks labeling, not camera coverage.
    window.__sim.person.node.position.set(0.6, 0, 2.5);
    const gt = await window.__sim.groundTruth('cvN', { noDepth: true });
    const maskImage = new Image();
    maskImage.src = gt.mask;
    await maskImage.decode();
    const canvas = document.createElement('canvas');
    canvas.width = maskImage.width;
    canvas.height = maskImage.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(maskImage, 0, 0);
    const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    const maskColors = new Set();
    let exactPersonPixels = 0;
    for (let i = 0; i < pixels.length; i += 4) {
      maskColors.add((pixels[i] << 16) | (pixels[i + 1] << 8) | pixels[i + 2]);
      if (pixels[i] === 255 && pixels[i + 1] === 0 && pixels[i + 2] === 0) exactPersonPixels++;
    }
    return {
      classes: window.__sim.GT_CLASSES.map(({ id, key, label }) => ({ id, key, label })),
      labels: gt.labels.map(({ id, label, instance }) => ({ id, label, instance })),
      instances: gt.instances.map(({ class: className, instance }) => ({ className, instance })),
      labelText: gt.labelText,
      maskUniqueColors: maskColors.size,
      exactPersonPixels,
    };
  });

  expect(result.classes).toEqual([{ id: 0, key: 'person', label: 'person' }]);
  expect(result.labels.length).toBeGreaterThan(0);
  expect(result.labels.every(label => label.id === 0 && label.label === 'person')).toBe(true);
  expect(result.instances.every(instance =>
    instance.className === 'person' && instance.instance.startsWith('person_'))).toBe(true);
  expect(result.labelText.split('\n').filter(Boolean).every(row => row.startsWith('0 '))).toBe(true);
  expect(result.maskUniqueColors).toBe(2);
  expect(result.exactPersonPixels).toBeGreaterThan(80);
});

test('instance decoder rejects an MSAA blended palette color', async ({ page }) => {
  const result = await page.evaluate(() => {
    const decoder = window.__sim.classifyByExactColor;
    return {
      type: typeof decoder,
      output: typeof decoder === 'function'
        ? Array.from(decoder(new Uint8Array([
            230, 25, 75, 255,
            60, 180, 75, 255,
            145, 103, 75, 255,
          ]), 3, 1, [
            ['person_0', [230, 25, 75]],
            ['person_1', [60, 180, 75]],
          ]))
        : null,
    };
  });

  expect(result.type).toBe('function');
  expect(result.output).toEqual([0, 1, -1]);
});
