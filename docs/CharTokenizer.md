# CharTokenizer – Implementation Tracker (v0.1)

> **Goal:** finish P1.2 by **5 May 2025**.

| Field | Value |
|-------|-------|
| **Owner**      | Miloš |
| **Repo branch**| `feature/char-tokenizer` |

---

Metadata is under `src/data_pipeline/tokenizers/char_dns/v0.1/`

## Kanban snapshot
### 🚧 DOING
- [ ] `CharTokenizer.__init__` constructs `char2id / id2char` in O(|alphabet|)  
      _ETA: 2 May_

### ⏳ TODO
- [ ] `encode` + `decode` round-trip test (10 k samples)  
      _ETA: 3 May_  
- [ ] `batch_encode_plus` with padding / truncation (64-token max)  
      _ETA: 3 May_  
- [ ] Benchmark ≥ 50 k seq/s on Ryzen 5 (**pytest-benchmark** gate)  
      _ETA: 4 May_  
- [ ] `save_pretrained / from_pretrained` writes/reads `tokenizer.json`  
      _ETA: 4 May_

### ✅ DONE
- [x] Design doc signed off (30 Apr)  
- [x] Alphabet and special-token IDs finalised

---

## Definition of Done
1. All **TODO** boxes checked.  
2. `pytest -m fast` passes (< 3 s); coverage ≥ 90 %.  
3. `benchmark_encode` median ≥ 50 k seq/s (CI log attaches table).  
4. Artefacts version-tagged `tokenizers/char_dns/v0.1`.
5. `metadata.json` under `src/data_pipeline/tokenizers/char_dns/v0.1/`:
   - `alphabet` (list of characters)  
   - `special_tokens` (list of special tokens)  

---

## CI performance
- CI performance is measured on Ryzen 5 PRO 5650U with 16GB RAM
- CI runs under Python 3.10.16 with `PYTHONHASHSEED=0`
- Benchmark input : 10 k random 30-char domains
- Fail CI if median latency worsens > 15 %

---

## Decision log
| Date | Decision | Rationale |
|------|----------|-----------|
| 30 Apr | Keep only lowercase ASCII; map Punycode as literal | Simplicity, 99.9 % coverage |
| 30 Apr | Remove `www.` prefix from URLs | Simplicity, 99.9 % coverage |
| 30 Apr | Strip trailing dots from URLs like `google.com.` | Simplicity, 99.9 % coverage |
| 30 Apr | Existence of `[UNK]` token will raise an error and training will stop mid-epoch, I will try to remove the sequence and continue from the last checkpoint | Simplicity, < 1 % coverage |
| 30 Apr | `max_length` is 256 tokens | Simplicity, 100 % coverage |
| 30 Apr | `.` Will be treated as a token | Simplicity, 100 % coverage |
| 30 Apr | `*` Will not be inside the vocabulary, while `_` and `-` will | Simplicity, 100 % coverage |
| 30 Apr | The padding is done by the Huggingface library | Simplicity, 100 % coverage |

---

### Special tokens
| Token | ID | Description |
|-------|----|-------------|
| `[PAD]` | 0 | Padding token |
| `[UNK]` | 1 | Unknown token |
| `[CLS]` | 2 | Classification token |
| `[SEP]` | 3 | Separator token |
| `[MASK]` | 4 | Mask token |

---

All the decisions are frozen and downstream decisions cannot contradict them!