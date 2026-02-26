// ThaiOCRBench Evaluation Runner
// Evaluates VLMs on 10 of 13 Thai OCR benchmark tasks
// (Skips Chart/Table/Document parsing which require TED/STEDS)

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { parquetRead } from 'hyparquet';
import { distance as levenshteinDistance } from 'fastest-levenshtein';
import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';
import 'dotenv/config';

const __dirname = dirname(fileURLToPath(import.meta.url));

// ── Types ──────────────────────────────────────────────────────────

interface Sample {
  id: string;
  imageBytes: Uint8Array;
  task: string;
  question: string;
  answer: string;
  category: string;
}

interface Result {
  id: string;
  task: string;
  question: string;
  answer: string;
  prediction: string;
  score: number;
  inputTokens: number;
  outputTokens: number;
  cost: number;
}

interface ApiResult {
  content: string;
  inputTokens: number;
  outputTokens: number;
}

// ── Constants ──────────────────────────────────────────────────────

const CONCURRENCY = 5;
const SAVE_INTERVAL = 10;
const DATA_DIR = join(__dirname, 'data');
const RESULTS_DIR = join(__dirname, 'results');
const SMALL_SAMPLES_PER_TASK = 30;

const PARQUET_FILES = [
  'test-00000-of-00004.parquet',
  'test-00001-of-00004.parquet',
  'test-00002-of-00004.parquet',
  'test-00003-of-00004.parquet',
];

const SKIPPED_TASKS = new Set([
  'Chart parsing',
  'Table parsing',
  'Document parsing',
  'Key information extraction',
  'Key information mapping',
]);

const ANLS_TASKS = new Set([
  'Text recognition',
  'Document classification',
  'Diagram VQA',
  'Cognition VQA',
  'Infographics',
]);

const F1_TASKS = new Set([
  'Key information extraction',
  'Key information mapping',
]);

const BMFL_TASKS = new Set([
  'Fine-grained text recognition',
  'Full-page OCR',
  'Handwritten content extraction',
]);

const TASK_ORDER = [
  'Fine-grained text recognition',
  'Full-page OCR',
  'Handwritten content extraction',
  'Text recognition',
  'Document classification',
  'Diagram VQA',
  'Cognition VQA',
  'Infographics',
];

// USD per token — update as pricing changes
const PRICING: Record<string, { input: number; output: number }> = {
  'gpt-4o':            { input: 2.50 / 1e6, output: 10.00 / 1e6 },
  'gpt-5-mini':        { input: 1.10 / 1e6, output:  4.40 / 1e6 },
  'gpt-5.2':           { input: 2.50 / 1e6, output: 10.00 / 1e6 },
  'claude-opus-4-6':   { input: 15.00 / 1e6, output: 75.00 / 1e6 },
  'claude-haiku-4-5':  { input: 0.80 / 1e6, output:  4.00 / 1e6 },
  'typhoon-ocr':       { input: 0,           output: 0            },
};

// ── Data Loading ───────────────────────────────────────────────────

const textDecoder = new TextDecoder();

function toArrayBuffer(buf: Buffer): ArrayBuffer {
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

function decodeStr(val: any): string {
  if (typeof val === 'string') return val;
  if (val instanceof Uint8Array) return textDecoder.decode(val);
  return String(val ?? '');
}

async function loadParquetFile(filePath: string): Promise<Sample[]> {
  const buf = readFileSync(filePath);
  const ab = toArrayBuffer(buf);
  const samples: Sample[] = [];

  await parquetRead({
    file: ab,
    // Return raw Uint8Array for all BYTE_ARRAY columns (preserves binary image data)
    parsers: { stringFromBytes: (bytes: Uint8Array) => bytes } as any,
    onComplete: (rows: any[][]) => {
      for (const row of rows) {
        try {
          // Schema: Id(0), image{bytes,path}(1), Task(2), question(3), answer(4), category(5)
          const id = decodeStr(row[0]);
          const imageObj = row[1] as { bytes: Uint8Array; path: Uint8Array };
          const imageBytes = imageObj.bytes;
          const task = decodeStr(row[2]);
          const question = decodeStr(row[3]);
          const answer = decodeStr(row[4]);
          const category = row[5] ? decodeStr(row[5]) : '';

          if (!SKIPPED_TASKS.has(task)) {
            samples.push({ id, imageBytes, task, question, answer, category });
          }
        } catch (err: any) {
          console.warn(`  Skipping row: ${err.message}`);
        }
      }
    },
  });

  return samples;
}

async function loadAllSamples(): Promise<Sample[]> {
  const allSamples: Sample[] = [];

  for (const file of PARQUET_FILES) {
    const filePath = join(DATA_DIR, file);
    if (!existsSync(filePath)) {
      console.error(`Missing: ${filePath}`);
      console.error('Run "npx tsx setup.ts" first to download the dataset.');
      process.exit(1);
    }
    console.log(`Loading ${file}...`);
    const samples = await loadParquetFile(filePath);
    allSamples.push(...samples);
  }

  // Print task distribution
  const taskCounts = new Map<string, number>();
  for (const s of allSamples) {
    taskCounts.set(s.task, (taskCounts.get(s.task) || 0) + 1);
  }
  console.log(`\nLoaded ${allSamples.length} samples across ${taskCounts.size} tasks:`);
  for (const task of TASK_ORDER) {
    const count = taskCounts.get(task);
    if (count) console.log(`  ${task}: ${count}`);
  }
  console.log();

  return allSamples;
}

// ── Stratified Sampling ───────────────────────────────────────────

/** Mulberry32 seeded PRNG — returns a function that produces [0, 1) floats */
function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Pick `perTask` samples from each task via Fisher-Yates shuffle with seeded PRNG */
function stratifiedSample(samples: Sample[], perTask: number, seed = 42): Sample[] {
  const rng = mulberry32(seed);
  const byTask = new Map<string, Sample[]>();
  for (const s of samples) {
    let arr = byTask.get(s.task);
    if (!arr) { arr = []; byTask.set(s.task, arr); }
    arr.push(s);
  }

  const result: Sample[] = [];
  for (const [, taskSamples] of byTask) {
    // Fisher-Yates shuffle (partial — only need first `n` elements)
    const n = Math.min(perTask, taskSamples.length);
    for (let i = 0; i < n; i++) {
      const j = i + Math.floor(rng() * (taskSamples.length - i));
      [taskSamples[i], taskSamples[j]] = [taskSamples[j], taskSamples[i]];
    }
    result.push(...taskSamples.slice(0, n));
  }

  return result;
}

// ── Image Utilities ────────────────────────────────────────────────

function detectMimeType(bytes: Uint8Array): string {
  if (bytes[0] === 0x89 && bytes[1] === 0x50) return 'image/png';
  if (bytes[0] === 0xff && bytes[1] === 0xd8) return 'image/jpeg';
  if (bytes[0] === 0x47 && bytes[1] === 0x49) return 'image/gif';
  if (bytes[0] === 0x52 && bytes[1] === 0x49) return 'image/webp';
  return 'image/png';
}

function imageToDataUrl(bytes: Uint8Array): string {
  const mime = detectMimeType(bytes);
  const b64 = Buffer.from(bytes).toString('base64');
  return `data:${mime};base64,${b64}`;
}

// ── Prompt Wrapping (matches reference code) ───────────────────────

function wrapPrompt(question: string): string {
  return (
    'Instruction: ' +
    question.trim() +
    '\nตอบเฉพาะข้อความตามภาพตรง ๆ อย่างกระชับ โดยไม่ใส่คำอธิบายเพิ่มเติม'
  );
}

// ── Thai Tokenization ──────────────────────────────────────────────

const thaiSegmenter = new Intl.Segmenter('th', { granularity: 'word' });

function tokenize(text: string): string[] {
  const hasThai = /[\u0E00-\u0E7F]/.test(text);
  if (hasThai) {
    return [...thaiSegmenter.segment(text)]
      .filter(s => s.isWordLike)
      .map(s => s.segment);
  }
  return text.split(/\s+/).filter(Boolean);
}

// ── N-gram Utilities ───────────────────────────────────────────────

function getNgrams(tokens: string[], n: number): Map<string, number> {
  const ngrams = new Map<string, number>();
  for (let i = 0; i <= tokens.length - n; i++) {
    const ng = tokens.slice(i, i + n).join('\x00');
    ngrams.set(ng, (ngrams.get(ng) || 0) + 1);
  }
  return ngrams;
}

// ── Metric: BLEU (matches NLTK sentence_bleu, no smoothing) ───────

function computeBLEU(reference: string[], hypothesis: string[]): number {
  if (hypothesis.length === 0 || reference.length === 0) return 0;

  const weights = [0.25, 0.25, 0.25, 0.25];
  let logSum = 0;

  for (let n = 1; n <= 4; n++) {
    const hypNg = getNgrams(hypothesis, n);
    const refNg = getNgrams(reference, n);

    let clipped = 0;
    let total = 0;
    for (const [ng, count] of hypNg) {
      total += count;
      clipped += Math.min(count, refNg.get(ng) || 0);
    }

    const precision = total > 0 ? clipped / total : 0;
    if (precision === 0) return 0; // geometric mean: any zero precision → BLEU = 0
    logSum += weights[n - 1] * Math.log(precision);
  }

  const bp =
    hypothesis.length >= reference.length
      ? 1
      : Math.exp(1 - reference.length / hypothesis.length);

  return bp * Math.exp(logSum);
}

// ── Metric: METEOR (exact unigram matching, NLTK defaults) ────────

function computeMETEOR(reference: string[], hypothesis: string[]): number {
  if (hypothesis.length === 0 || reference.length === 0) return 0;

  // Greedy exact unigram matching
  const refUsed = new Set<number>();
  const matchMap = new Map<number, number>(); // hyp_idx -> ref_idx

  for (let i = 0; i < hypothesis.length; i++) {
    const hTok = hypothesis[i].toLowerCase();
    for (let j = 0; j < reference.length; j++) {
      if (!refUsed.has(j) && reference[j].toLowerCase() === hTok) {
        refUsed.add(j);
        matchMap.set(i, j);
        break;
      }
    }
  }

  const matches = matchMap.size;
  if (matches === 0) return 0;

  const P = matches / hypothesis.length;
  const R = matches / reference.length;

  // NLTK defaults: alpha=0.9(recall-biased), beta=3, gamma=0.5
  const alpha = 0.9;
  const denom = alpha * P + (1 - alpha) * R;
  if (denom === 0) return 0;
  const Fmean = (P * R) / denom;

  // Count chunks (contiguous aligned sequences)
  const sortedHypIdx = [...matchMap.keys()].sort((a, b) => a - b);
  let chunks = 1;
  for (let i = 1; i < sortedHypIdx.length; i++) {
    const prevH = sortedHypIdx[i - 1];
    const currH = sortedHypIdx[i];
    const prevR = matchMap.get(prevH)!;
    const currR = matchMap.get(currH)!;
    if (currH !== prevH + 1 || currR !== prevR + 1) {
      chunks++;
    }
  }

  const gamma = 0.5;
  const beta = 3;
  const penalty = gamma * Math.pow(chunks / matches, beta);

  return Fmean * (1 - penalty);
}

// ── Metric: Set F1 (bag-of-words, matches NLTK f_measure) ─────────

function computeSetF1(reference: string[], hypothesis: string[]): number {
  const refSet = new Set(reference);
  const hypSet = new Set(hypothesis);

  if (hypSet.size === 0 || refSet.size === 0) return 0;

  let intersection = 0;
  for (const tok of hypSet) {
    if (refSet.has(tok)) intersection++;
  }

  const precision = intersection / hypSet.size;
  const recall = intersection / refSet.size;

  if (precision + recall === 0) return 0;
  return (2 * precision * recall) / (precision + recall);
}

// ── Metric: NLS (character-level, matches reference) ──────────────

function computeNLS(prediction: string, groundTruth: string): number {
  const maxLen = Math.max(prediction.length, groundTruth.length);
  if (maxLen === 0) return 1;
  return 1 - levenshteinDistance(prediction, groundTruth) / maxLen;
}

// ── Metric: BMFL = (BLEU + METEOR + setF1 + NLS) / 4 ─────────────

function computeBMFL(prediction: string, groundTruth: string): number {
  const refTokens = tokenize(groundTruth);
  const hypTokens = tokenize(prediction);

  const bleu = computeBLEU(refTokens, hypTokens);
  const meteor = computeMETEOR(refTokens, hypTokens);
  const f1 = computeSetF1(refTokens, hypTokens);
  const nls = computeNLS(prediction, groundTruth);

  return (bleu + meteor + f1 + nls) / 4;
}

// ── Metric: ANLS (cn_vqa_evaluation from reference code) ──────────

function computeANLS(prediction: string, answers: string[]): number {
  let maxScore = 0;

  for (const answer of answers) {
    // Normalize: lowercase, trim, replace newlines, remove all spaces
    const pred = prediction
      .toLowerCase()
      .trim()
      .replace(/\n/g, ' ')
      .replace(/ /g, '');
    const ans = answer
      .toLowerCase()
      .trim()
      .replace(/\n/g, ' ')
      .replace(/ /g, '');

    let score = 0;

    // Short answer containment check (< 4 comma-separated parts)
    if (ans.split(',').length < 4 && pred.includes(ans)) {
      score = 1;
    }

    // NLS with threshold >= 0.5
    if (score < 1) {
      const maxLen = Math.max(pred.length, ans.length);
      if (maxLen === 0) {
        score = 1;
      } else {
        const nls = 1 - levenshteinDistance(pred, ans) / maxLen;
        if (nls >= 0.5) {
          score = Math.max(score, nls);
        }
      }
    }

    maxScore = Math.max(maxScore, score);
  }

  return maxScore;
}

// ── Metric: Key-level F1 (matches reference compute_f1_score) ─────

function tryParseDict(text: string): Record<string, string> | null {
  // Try direct JSON parse
  try {
    const parsed = JSON.parse(text);
    if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
      const result: Record<string, string> = {};
      for (const [k, v] of Object.entries(parsed)) {
        result[k] = String(v);
      }
      return result;
    }
  } catch {}

  // Try extracting JSON from markdown code blocks
  const codeBlock = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (codeBlock) {
    try {
      const parsed = JSON.parse(codeBlock[1].trim());
      if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
        const result: Record<string, string> = {};
        for (const [k, v] of Object.entries(parsed)) {
          result[k] = String(v);
        }
        return result;
      }
    } catch {}
  }

  // Try finding JSON object in the text
  const jsonMatch = text.match(/\{[\s\S]*\}/);
  if (jsonMatch) {
    try {
      const parsed = JSON.parse(jsonMatch[0]);
      if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
        const result: Record<string, string> = {};
        for (const [k, v] of Object.entries(parsed)) {
          result[k] = String(v);
        }
        return result;
      }
    } catch {}
  }

  return null;
}

function normalizeValue(value: string): string {
  return value.toLowerCase().trim().replace(/\n/g, ' ').replace(/ /g, '');
}

function computeKeyF1(prediction: string, groundTruth: string): number {
  const gtDict = tryParseDict(groundTruth);
  const predDict = tryParseDict(prediction);

  if (!gtDict) return 0;
  if (!predDict) return 0;

  const allKeys = new Set([...Object.keys(gtDict), ...Object.keys(predDict)]);
  if (allKeys.size === 0) return 0;

  let totalF1 = 0;

  for (const key of allKeys) {
    const gtRaw = gtDict[key];
    const predRaw = predDict[key];

    if (gtRaw === undefined || predRaw === undefined) {
      // Missing key in either → F1 = 0
      continue;
    }

    const normalizedPred = normalizeValue(predRaw);

    // Handle ground truth with multiple valid values (JSON array)
    let gtOptions: string[];
    try {
      const parsed = JSON.parse(gtRaw);
      if (Array.isArray(parsed)) {
        gtOptions = parsed.map(v => normalizeValue(String(v)));
      } else {
        gtOptions = [normalizeValue(gtRaw)];
      }
    } catch {
      gtOptions = [normalizeValue(gtRaw)];
    }

    const matches = gtOptions.some(opt => opt === normalizedPred);
    totalF1 += matches ? 1 : 0;
  }

  return totalF1 / allKeys.size;
}

// ── Score Dispatcher ───────────────────────────────────────────────

function computeScore(task: string, prediction: string, answer: string): number {
  if (ANLS_TASKS.has(task)) return computeANLS(prediction, [answer]);
  if (F1_TASKS.has(task)) return computeKeyF1(prediction, answer);
  if (BMFL_TASKS.has(task)) return computeBMFL(prediction, answer);
  console.warn(`Unknown task: "${task}", using ANLS`);
  return computeANLS(prediction, [answer]);
}

function getMetricName(task: string): string {
  if (ANLS_TASKS.has(task)) return 'ANLS';
  if (F1_TASKS.has(task)) return 'F1';
  if (BMFL_TASKS.has(task)) return 'BMFL';
  return '?';
}

// ── OpenAI API ─────────────────────────────────────────────────────

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function callVLM(
  client: OpenAI,
  model: string,
  imageDataUrl: string,
  prompt: string,
  maxRetries = 5,
): Promise<ApiResult> {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const response = await client.chat.completions.create({
        model,
        messages: [
          {
            role: 'user',
            content: [
              { type: 'image_url', image_url: { url: imageDataUrl } },
              { type: 'text', text: prompt },
            ],
          },
        ],
      });
      return {
        content: response.choices[0]?.message?.content ?? '',
        inputTokens: response.usage?.prompt_tokens ?? 0,
        outputTokens: response.usage?.completion_tokens ?? 0,
      };
    } catch (err: any) {
      const isRateLimit =
        err?.status === 429 || err?.code === 'rate_limit_exceeded';
      const isRetryable = isRateLimit || err?.status >= 500;

      if (isRetryable && attempt < maxRetries - 1) {
        const baseDelay = isRateLimit ? 5000 : 1000;
        const delay = baseDelay * Math.pow(2, attempt) + Math.random() * 1000;
        console.warn(
          `  ${isRateLimit ? 'Rate limited' : 'Server error'}, retry in ${(delay / 1000).toFixed(1)}s...`,
        );
        await sleep(delay);
        continue;
      }

      console.error(`  API error: ${err.message}`);
      return { content: '', inputTokens: 0, outputTokens: 0 };
    }
  }
  return { content: '', inputTokens: 0, outputTokens: 0 };
}

// ── Anthropic API ───────────────────────────────────────────────────

async function callAnthropic(
  client: Anthropic,
  model: string,
  imageBytes: Uint8Array,
  prompt: string,
  maxRetries = 5,
): Promise<ApiResult> {
  const mime = detectMimeType(imageBytes);
  const b64 = Buffer.from(imageBytes).toString('base64');

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const response = await client.messages.create({
        model,
        max_tokens: 4096,
        messages: [
          {
            role: 'user',
            content: [
              { type: 'image', source: { type: 'base64', media_type: mime as any, data: b64 } },
              { type: 'text', text: prompt },
            ],
          },
        ],
      });
      const text = response.content
        .filter((b): b is Anthropic.TextBlock => b.type === 'text')
        .map(b => b.text)
        .join('');
      return {
        content: text,
        inputTokens: response.usage.input_tokens,
        outputTokens: response.usage.output_tokens,
      };
    } catch (err: any) {
      const isRateLimit = err?.status === 429;
      const isRetryable = isRateLimit || err?.status >= 500;

      if (isRetryable && attempt < maxRetries - 1) {
        const baseDelay = isRateLimit ? 5000 : 1000;
        const delay = baseDelay * Math.pow(2, attempt) + Math.random() * 1000;
        console.warn(
          `  ${isRateLimit ? 'Rate limited' : 'Server error'}, retry in ${(delay / 1000).toFixed(1)}s...`,
        );
        await sleep(delay);
        continue;
      }

      console.error(`  API error: ${err.message}`);
      return { content: '', inputTokens: 0, outputTokens: 0 };
    }
  }
  return { content: '', inputTokens: 0, outputTokens: 0 };
}

// ── Typhoon OCR API ─────────────────────────────────────────────────

function mimeExtension(mime: string): string {
  if (mime === 'image/jpeg') return 'jpg';
  if (mime === 'image/png') return 'png';
  if (mime === 'image/gif') return 'gif';
  if (mime === 'image/webp') return 'webp';
  return 'png';
}

async function callTyphoonOCR(
  apiKey: string,
  imageBytes: Uint8Array,
  maxRetries = 5,
): Promise<ApiResult> {
  const mime = detectMimeType(imageBytes);
  const ext = mimeExtension(mime);

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const blob = new Blob([imageBytes], { type: mime });
      const formData = new FormData();
      formData.append('file', blob, `image.${ext}`);
      formData.append('model', 'typhoon-ocr');
      formData.append('task_type', 'default');
      formData.append('max_tokens', '16384');
      formData.append('temperature', '0.1');
      formData.append('top_p', '0.6');
      formData.append('repetition_penalty', '1.2');

      const response = await fetch('https://api.opentyphoon.ai/v1/ocr', {
        method: 'POST',
        headers: { Authorization: `Bearer ${apiKey}` },
        body: formData,
      });

      if (response.ok) {
        const result: any = await response.json();
        const texts: string[] = [];
        let inputTokens = 0;
        let outputTokens = 0;
        for (const pageResult of result.results || []) {
          if (pageResult.success && pageResult.message) {
            let content = pageResult.message.choices[0].message.content;
            const usage = pageResult.message.usage;
            if (usage) {
              inputTokens += usage.prompt_tokens ?? 0;
              outputTokens += usage.completion_tokens ?? 0;
            }
            try {
              const parsed = JSON.parse(content);
              content = parsed.natural_text || content;
            } catch {}
            texts.push(content);
          }
        }
        return { content: texts.join('\n'), inputTokens, outputTokens };
      }

      if (response.status === 429 || response.status >= 500) {
        const baseDelay = response.status === 429 ? 5000 : 1000;
        const delay = baseDelay * Math.pow(2, attempt) + Math.random() * 1000;
        console.warn(
          `  ${response.status === 429 ? 'Rate limited' : 'Server error'} (${response.status}), retry in ${(delay / 1000).toFixed(1)}s...`,
        );
        await sleep(delay);
        continue;
      }

      console.error(`  API error: ${response.status} ${await response.text()}`);
      return { content: '', inputTokens: 0, outputTokens: 0 };
    } catch (err: any) {
      if (attempt < maxRetries - 1) {
        const delay = 1000 * Math.pow(2, attempt) + Math.random() * 1000;
        console.warn(`  Network error, retry in ${(delay / 1000).toFixed(1)}s...`);
        await sleep(delay);
        continue;
      }
      console.error(`  Error: ${err.message}`);
      return { content: '', inputTokens: 0, outputTokens: 0 };
    }
  }
  return { content: '', inputTokens: 0, outputTokens: 0 };
}

// ── Results Management ─────────────────────────────────────────────

function getResultsPath(model: string): string {
  return join(RESULTS_DIR, `${model.replace(/[/\\]/g, '_')}.json`);
}

function loadResults(model: string): Result[] {
  const path = getResultsPath(model);
  if (!existsSync(path)) return [];
  try {
    return JSON.parse(readFileSync(path, 'utf-8'));
  } catch {
    return [];
  }
}

function saveResults(model: string, results: Result[]): void {
  mkdirSync(RESULTS_DIR, { recursive: true });
  writeFileSync(getResultsPath(model), JSON.stringify(results, null, 2));
}

// ── Summary Output ─────────────────────────────────────────────────

function printSummary(resultsKey: string, results: Result[]): void {
  const displayName = resultsKey.replace(/-small$/, '');
  console.log(`\n${'='.repeat(65)}`);
  console.log(`  ThaiOCRBench Results — ${displayName}`);
  console.log(`${'='.repeat(65)}`);
  console.log(
    `${'Task'.padEnd(36)} ${'Metric'.padEnd(8)} ${'N'.padStart(5)}   Score`,
  );
  console.log('-'.repeat(65));

  const taskScores: { task: string; avg: number }[] = [];

  for (const task of TASK_ORDER) {
    const taskResults = results.filter(r => r.task === task);
    if (taskResults.length === 0) continue;

    const avg =
      taskResults.reduce((s, r) => s + r.score, 0) / taskResults.length;
    const metric = getMetricName(task);
    taskScores.push({ task, avg });

    console.log(
      `${task.padEnd(36)} ${metric.padEnd(8)} ${String(taskResults.length).padStart(5)}   ${avg.toFixed(3)}`,
    );
  }

  console.log('-'.repeat(65));

  const overallAvg =
    taskScores.length > 0
      ? taskScores.reduce((s, t) => s + t.avg, 0) / taskScores.length
      : 0;

  console.log(
    `${'Overall Average'.padEnd(36)} ${''.padEnd(8)} ${String(results.length).padStart(5)}   ${overallAvg.toFixed(3)}`,
  );
  console.log('='.repeat(65));

  const totalCost = results.reduce((s, r) => s + (r.cost ?? 0), 0);
  const totalInput = results.reduce((s, r) => s + (r.inputTokens ?? 0), 0);
  const totalOutput = results.reduce((s, r) => s + (r.outputTokens ?? 0), 0);
  const avgCost = results.length > 0 ? totalCost / results.length : 0;

  console.log(`\n  Cost: $${totalCost.toFixed(4)} total, $${avgCost.toFixed(4)}/image`);
  console.log(`  Tokens: ${totalInput.toLocaleString()} input + ${totalOutput.toLocaleString()} output = ${(totalInput + totalOutput).toLocaleString()} total`);
}

// ── Main Runner ────────────────────────────────────────────────────

async function main() {
  const args = process.argv.slice(2).filter(a => !a.startsWith('--'));
  const flags = new Set(process.argv.slice(2).filter(a => a.startsWith('--')));
  const model = args[0] || 'gpt-4o';
  const small = flags.has('--small');

  console.log('ThaiOCRBench Evaluation Runner');
  console.log(`Model: ${model}${small ? ' (small bench)' : ''}\n`);

  // Load dataset
  let samples = await loadAllSamples();

  if (small) {
    samples = stratifiedSample(samples, SMALL_SAMPLES_PER_TASK);
    console.log(`Small bench mode: ${SMALL_SAMPLES_PER_TASK} samples per task (${samples.length} total)\n`);
  }

  // Load existing results for resumability
  const resultsKey = small ? `${model}-small` : model;
  const results = loadResults(resultsKey);
  const completedIds = new Set(results.map(r => r.id));
  const pending = samples.filter(s => !completedIds.has(s.id));

  if (pending.length === 0) {
    console.log('All samples already processed!');
    printSummary(resultsKey, results);
    return;
  }

  console.log(
    `Progress: ${completedIds.size} done, ${pending.length} remaining\n`,
  );

  // Initialize API clients
  const isTyphoon = model.startsWith('typhoon');
  const isClaude = model.startsWith('claude');
  let openaiClient: OpenAI | null = null;
  let anthropicClient: Anthropic | null = null;
  let typhoonKeys: string[] = [];

  if (isTyphoon) {
    typhoonKeys = [1, 2, 3, 4]
      .map(i => process.env[`TYPHOON_API_KEY_${i}`])
      .filter(Boolean) as string[];
    if (typhoonKeys.length === 0) {
      console.error('Set TYPHOON_API_KEY_1..4 in .env');
      process.exit(1);
    }
    console.log(`Using ${typhoonKeys.length} Typhoon API keys\n`);
  } else if (isClaude) {
    anthropicClient = new Anthropic();
  } else {
    openaiClient = new OpenAI();
  }

  let completed = 0;
  let queueIndex = 0;
  let typhoonKeyIdx = 0;

  async function worker() {
    while (true) {
      const idx = queueIndex++;
      if (idx >= pending.length) break;

      const sample = pending[idx];
      let apiResult: ApiResult;

      if (isTyphoon) {
        const key = typhoonKeys[typhoonKeyIdx++ % typhoonKeys.length];
        apiResult = await callTyphoonOCR(key, sample.imageBytes);
      } else if (isClaude) {
        const prompt = wrapPrompt(sample.question);
        apiResult = await callAnthropic(anthropicClient!, model, sample.imageBytes, prompt);
      } else {
        const prompt = wrapPrompt(sample.question);
        const dataUrl = imageToDataUrl(sample.imageBytes);
        apiResult = await callVLM(openaiClient!, model, dataUrl, prompt);
      }

      const score = computeScore(sample.task, apiResult.content, sample.answer);
      const price = PRICING[model] ?? { input: 0, output: 0 };
      const cost = apiResult.inputTokens * price.input + apiResult.outputTokens * price.output;

      results.push({
        id: sample.id,
        task: sample.task,
        question: sample.question,
        answer: sample.answer,
        prediction: apiResult.content,
        score,
        inputTokens: apiResult.inputTokens,
        outputTokens: apiResult.outputTokens,
        cost,
      });

      completed++;
      const pct = ((completed / pending.length) * 100).toFixed(1);
      console.log(
        `[${completed}/${pending.length}] (${pct}%) ${sample.task}: ${score.toFixed(3)}`,
      );

      if (completed % SAVE_INTERVAL === 0) {
        saveResults(resultsKey, results);
      }
    }
  }

  const concurrency = isTyphoon ? typhoonKeys.length * 2 : CONCURRENCY;
  const numWorkers = Math.min(concurrency, pending.length);
  await Promise.all(Array.from({ length: numWorkers }, () => worker()));

  // Final save
  saveResults(resultsKey, results);
  printSummary(resultsKey, results);
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
