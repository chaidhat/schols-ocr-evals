# ThaiOCRBench Small 2026-02-25

Chaidhat Chaimongkol, Kanitta Ket-in, Kanokwan Sae-Tae, Artis Nateephaisan\
Scholarity Co., Ltd. Thailand \
chai@schols.io

Reproduction and extension of the [ThaiOCRBench paper](https://arxiv.org/abs/2511.04479) (Nonesung et al., 2025). We re-run the benchmark on GPT-4o and extend it to gemini-3-pro-preview, gemini-3-flash-preview, gpt-5.2, gpt-5-mini, typhoon-ocr, claude-opus-4-6, and claude-haiku-4-5. This was done in February 25, 2026.

| Model | Fine-grained | Full-page OCR | Handwritten | Text recog. | Doc. class. | Diagram VQA | Cognition VQA | Infographics | Avg | $/img |
|---|---|---|---|---|---|---|---|---|---|---|
| gemini-3-pro | **.614** | .976 | .820 | .903 | .967 | .595 | .829 | .814 | **.815** | — |
| gemini-3-flash | .499 | **.978** | **.856** | **.914** | .967 | **.607** | **.872** | .810 | .813 | .0003 |
| gpt-5-mini | .324 | .848 | .545 | .765 | .967 | .535 | .764 | .803 | .694 | .0052 |
| gpt-5.2 | .229 | .899 | .564 | .749 | .900 | .504 | .816 | **.813** | .684 | .0045 |
| claude-opus-4-6 | .290 | .949 | .626 | .608 | .967 | .513 | .674 | .678 | .663 | .0352 |
| gpt-4o | .259 | .715 | .418 | .732 | **.987** | .479 | .724 | .665 | .626 | — |
| Paper GPT-4o | .259 | .637 | .414 | .729 | .987 | .509 | .717 | .692 | .618 | — |
| typhoon-ocr | .131 | .881 | .331 | .572 | .267 | .400 | .500 | .533 | .452 | — |
| claude-haiku-4-5 | .099 | .474 | .195 | .476 | .855 | .204 | .526 | .417 | .406 | .0018 |

30 samples/task, 240 total. Metrics: BMFL (first 3 tasks), ANLS (rest).

# Table of Contents

- [How to run](#how-to-run)
- [What we contributed](#what-we-contributed)
- [Differences from the paper](#differences-from-the-paper)
- [Results](#results)

# How to run

```bash
# 1. Download the dataset (~1.8 GB)
npx tsx setup.ts

# 2. Run evaluation (defaults to gpt-4o)
npx tsx index.ts

# 3. Run on a different model
npx tsx index.ts gpt-5-mini

# 4. Run small bench (30 samples per task, 300 total — ~7x faster/cheaper)
npx tsx index.ts gpt-4o --small
```


# What we contributed
1. We extended ThaiOCRBench by creating ThaiOCRBench Small.
2. We ran the benchmark in the same way the authors of the paper did but for `gemini-3-pro-preview`, `gemini-3-flash-preview`, `gpt-5.2`, `gpt-5-mini`, `typhoon-ocr`, `claude-opus-4-6`, and `claude-haiku-4-5` as of Feb 25, 2026.
3. We verified that our methodology follows that of the original paper by running the benchmark for `gpt-4o` and were within 9% of the original results.

ThaiOCRBench Small is a stratified random subset: 30 samples per task (240 total across 8 tasks). This gives ~7x cost reduction while maintaining ±9% confidence intervals per task at 95% confidence. Used a fixed seed (`42`) for reproducibility — every run selects the same 240 samples. Our results (see below) fell within this confidence interval except for one category.



# Differences from the paper

We follow the paper's methodology — same dataset, prompt wrapping, greedy decoding, and scoring metrics (ANLS, F1, BMFL) — with the following differences:

1. **5 of 13 tasks skipped:** Chart parsing (TED), Table parsing (TED), and Document parsing (STEDS) require tree edit distance algorithms we did not reimplement in TypeScript. Key information extraction and Key information mapping (F1) are skipped because pure-OCR endpoints (e.g. typhoon-ocr) cannot produce the required JSON output, making cross-model comparison unfair. We evaluate 8 tasks (1,794 of 2,808 samples), so our overall average is not directly comparable to the paper's 13-task average.

2. **Thai tokenization (BMFL tasks only):** We use `Intl.Segmenter('th', { granularity: 'word' })` (ICU) instead of `pythainlp.word_tokenize(engine='newmm')`. This may produce slightly different token boundaries, affecting BLEU, METEOR, and set-F1. The character-level NLS component is unaffected.

3. **METEOR without synonym/stem matching (BMFL tasks only):** We implement exact unigram matching only, versus NLTK's `meteor_score()` which supports synonyms and stemming. Impact is minor since METEOR is averaged with 3 other sub-metrics.

4. **OpenAI API version drift:** The paper's GPT-4o results may come from a different model snapshot. Exact score reproduction is not guaranteed.

For the 5 non-BMFL tasks (ANLS), our scoring is functionally identical to the paper.

# Results

## ThaiOCRBench Small (30 samples/task, 240 total)

| Task | Metric | Paper GPT-4o | gemini-3-pro | gemini-3-flash | gpt-5.2 | gpt-5-mini | claude-opus-4-6 | gpt-4o | typhoon-ocr | claude-haiku-4-5 |
|---|---|---|---|---|---|---|---|---|---|---|
| Fine-grained text recognition | BMFL | 0.259 | **0.614** | 0.499 | 0.229 | 0.324 | 0.290 | 0.259 | 0.131 | 0.099 |
| Full-page OCR | BMFL | 0.637 | 0.976 | **0.978** | 0.899 | 0.848 | 0.949 | 0.715 | 0.881 | 0.474 |
| Handwritten content extraction | BMFL | 0.414 | 0.820 | **0.856** | 0.564 | 0.545 | 0.626 | 0.418 | 0.331 | 0.195 |
| Text recognition | ANLS | 0.729 | 0.903 | **0.914** | 0.749 | 0.765 | 0.608 | 0.732 | 0.572 | 0.476 |
| Document classification | ANLS | 0.987 | 0.967 | 0.967 | 0.900 | 0.967 | 0.967 | **0.987** | 0.267 | 0.855 |
| Diagram VQA | ANLS | 0.509 | 0.595 | **0.607** | 0.504 | 0.535 | 0.513 | 0.479 | 0.400 | 0.204 |
| Cognition VQA | ANLS | 0.717 | 0.829 | **0.872** | 0.816 | 0.764 | 0.674 | 0.724 | 0.500 | 0.526 |
| Infographics | ANLS | 0.692 | **0.814** | 0.810 | 0.813 | 0.803 | 0.678 | 0.665 | 0.533 | 0.417 |
| **Overall Average** | | **0.618** | **0.815** | **0.813** | **0.684** | **0.694** | **0.663** | **0.626** | **0.452** | **0.406** |

### Cost

| Model | Total Cost | Cost/Image | Input Tokens | Output Tokens |
|---|---|---|---|---|
| gemini-3-flash | $0.05 | $0.0002 | 271,824 | 21,726 |
| gemini-3-pro | — | — | 271,824 | 21,945 |
| gpt-5-mini | $1.24 | $0.0052 | 328,134 | 199,161 |
| gpt-5.2 | $1.07 | $0.0045 | 328,134 | 25,316 |
| claude-opus-4-6 | $8.46 | $0.0352 | 302,206 | 52,352 |
| claude-haiku-4-5 | $0.44 | $0.0018 | 302,206 | 48,476 |
| typhoon-ocr | — | — | 597,330 | 123,598 |

Note: ~16 images exceeded Anthropic's 5 MB base64 limit and scored 0 for Claude models. Typhoon OCR had intermittent network errors affecting some Document classification scores.