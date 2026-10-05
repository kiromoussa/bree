// Drives the reviewer page in headless Chrome and prints what the keyboard flow did, as JSON.
//   BREE_PLAYWRIGHT=<path to node_modules/playwright> node scripts/review/page_keys.mjs http://127.0.0.1:8090
// Used by tests/test_review.py (skipped when BREE_PLAYWRIGHT is not set). Run it on a SYNTHETIC store
// with at least 5 alerts. The reviewer id is "kb".
import { createRequire } from 'node:module';
const { chromium } = createRequire(import.meta.url)(process.env.BREE_PLAYWRIGHT);
const url = process.argv[2];
const browser = await chromium.launch({ channel: 'chrome' });
const page = await browser.newPage();
const outside = [];
page.on('request', r => { if (!r.url().startsWith(url)) outside.push(r.url()); });
await page.goto(url);
await page.fill('#who', 'kb');
await page.press('#who', 'Tab');                       // change event: loads the first alert
await page.waitForSelector('#item');
const id = () => page.evaluate(() => cur && cur.alert_id);
const left = () => page.evaluate(() => parseInt(document.getElementById('left').textContent));
const settled = prev => page.waitForFunction(p => !busy && (!cur || cur.alert_id !== p), prev);
const ready = () => page.waitForTimeout(450);          // the page ignores decisions for 400 ms after an alert appears
const first = await id();

// 1. Too early: a key right after the alert appears is ignored. The key is pressed from inside the page in
//    the same task that marks the alert as just shown (t0), so the check does not depend on how long the
//    browser and this script took to get here (on a loaded machine that was more than the 400 ms).
await page.evaluate(() => { t0 = performance.now(); document.dispatchEvent(new KeyboardEvent('keydown', { key: '1', bubbles: true })); });
await page.waitForTimeout(150);
const tooEarly = (await id()) !== first || await page.evaluate(() => busy);

// 2. Key 3 = wrong item. The "3" is not typed, Enter submits without the mouse.
await ready();
await page.keyboard.press('3');
const afterKey3 = await page.inputValue('#item');      // must be empty
const focused = await page.evaluate(() => document.activeElement.id);
await page.keyboard.type('Chips ');                    // stored as "chips"
await page.keyboard.press('Enter');
await settled(first);

// 3. Text left in the field must not be stored with another decision. Then key 1 HELD for about
//    a second (30 repeat keydowns): exactly one decision, not one per repeat.
const second = await id();
const leftBeforeHold = await left();
await ready();
await page.fill('#item', 'leftover');
await page.evaluate(() => document.activeElement.blur());
for (let i = 0; i < 30; i++) { await page.keyboard.down('1'); await page.waitForTimeout(33); }
await page.keyboard.up('1');
await settled(second);
await page.waitForTimeout(300);
const third = await id();
const leftAfterHold = await left();

// 4. Five separate fast taps: the first decides, the rest land while saving or inside the 400 ms.
await ready();
for (let i = 0; i < 5; i++) await page.keyboard.press('2');
await settled(third);
await page.waitForTimeout(300);
const fourth = await id();
const leftAfterTaps = await left();

// 5. Back (key U) reopens the alert just decided. Deciding again replaces the decision.
await ready();
await page.keyboard.press('u');
await page.waitForFunction(p => !busy && cur && cur.alert_id === p, third);
const backShows = await page.evaluate(() => document.getElementById('main').textContent.includes('You decided: not theft'));
await ready();
await page.keyboard.press('4');
await settled(third);
const afterBack = await id();

// 6. Held key 3: the repeats must not be typed into the item field.
await ready();
for (let i = 0; i < 10; i++) { await page.keyboard.down('3'); await page.waitForTimeout(33); }
await page.keyboard.up('3');
const heldKey3 = await page.inputValue('#item');

// 7. An item that is not in the list needs Enter twice.
await page.keyboard.type('chpis');
await page.keyboard.press('Enter');
await page.waitForTimeout(200);
const unlistedFirstEnter = { moved: (await id()) !== fourth, hint: await page.textContent('#hint') };
await page.keyboard.press('Enter');
await settled(fourth);

console.log(JSON.stringify({ first, second, third, fourth, afterBack, tooEarly, afterKey3, focused, leftBeforeHold,
  leftAfterHold, leftAfterTaps, backShows, heldKey3, unlistedFirstEnter, outside }));
await browser.close();
