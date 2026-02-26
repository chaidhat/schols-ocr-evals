# ThaiOCRBench Small 2026-02-25

Chaidhat Chaimongkol, Kanitta Ket-in, Kanokwan Sae-Tae, Artis Nateephaisan\
Scholarity Co., Ltd. Thailand \
chai@schols.io

Reproduction and extension of the [ThaiOCRBench paper](https://arxiv.org/abs/2511.04479) (Nonesung et al., 2025). We re-run the benchmark on GPT-4o and extend it to gpt-5-mini, typhoon-ocr, claude-opus-4-6, and claude-haiku-4-5. This was done in February 25, 2026.

| Model | Fine-grained | Full-page OCR | Handwritten | Text recog. | Doc. class. | Diagram VQA | Cognition VQA | Infographics | Avg | $/img |
|---|---|---|---|---|---|---|---|---|---|---|
| claude-opus-4-6 | .290 | **.949** | .626 | .608 | .967 | .513 | .674 | .678 | .663 | .0352 |
| typhoon-ocr | .131 | .881 | .331 | .572 | .267 | .400 | .500 | .533 | .452 | — |
| gpt-5-mini | .324 | .848 | .545 | .765 | .967 | .535 | .764 | .803 | **.694** | .0052 |
| gpt-4o | .259 | .715 | .418 | .732 | **.987** | .479 | .724 | .665 | .626 | — |
| Paper GPT-4o | .259 | .637 | .414 | .729 | .987 | .509 | .717 | .692 | .618 | — |
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
2. We ran the benchmark in the same way the authors of the paper did but for `gpt-5-mini`, `typhoon-ocr`, `claude-opus-4-6`, and `claude-haiku-4-5` as of Feb 25, 2026.
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

| Task | Metric | Paper GPT-4o | gpt-5-mini | claude-opus-4-6 | gpt-4o | typhoon-ocr | claude-haiku-4-5 |
|---|---|---|---|---|---|---|---|
| Fine-grained text recognition | BMFL | 0.259 | 0.324 | 0.290 | 0.259 | 0.131 | 0.099 |
| Full-page OCR | BMFL | 0.637 | 0.848 | **0.949** | 0.715 | 0.881 | 0.474 |
| Handwritten content extraction | BMFL | 0.414 | 0.545 | 0.626 | 0.418 | 0.331 | 0.195 |
| Text recognition | ANLS | 0.729 | 0.765 | 0.608 | 0.732 | 0.572 | 0.476 |
| Document classification | ANLS | 0.987 | 0.967 | 0.967 | **0.987** | 0.267 | 0.855 |
| Diagram VQA | ANLS | 0.509 | 0.535 | 0.513 | 0.479 | 0.400 | 0.204 |
| Cognition VQA | ANLS | 0.717 | 0.764 | 0.674 | 0.724 | 0.500 | 0.526 |
| Infographics | ANLS | 0.692 | 0.803 | 0.678 | 0.665 | 0.533 | 0.417 |
| **Overall Average** | | **0.618** | **0.694** | **0.663** | **0.626** | **0.452** | **0.406** |

### Cost

| Model | Total Cost | Cost/Image | Input Tokens | Output Tokens |
|---|---|---|---|---|
| gpt-5-mini | $1.24 | $0.0052 | 328,134 | 199,161 |
| claude-opus-4-6 | $8.46 | $0.0352 | 302,206 | 52,352 |
| claude-haiku-4-5 | $0.44 | $0.0018 | 302,206 | 48,476 |
| typhoon-ocr | — | — | 597,330 | 123,598 |

Note: ~16 images exceeded Anthropic's 5 MB base64 limit and scored 0 for Claude models. Typhoon OCR had intermittent network errors affecting some Document classification scores.