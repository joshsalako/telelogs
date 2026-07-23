# Qwen3.5-4B SFT on Google Colab

[`unsloth_sft.ipynb`](unsloth_sft.ipynb) fine-tunes
`unsloth/Qwen3.5-4B` on the synthetic telecom RCA reasoning data and evaluates
the resulting adapter on the official 864-question holdout.

The notebook targets a free Colab T4. It uses 16-bit LoRA rather than 4-bit
QLoRA because Unsloth currently warns that Qwen3.5 has unusually large
4-bit quantization differences. Qwen3.5 also requires Transformers v5.

## Input files

Upload the notebook to Colab and select **Runtime → Change runtime type → T4
GPU**. Then upload:

- `synthesis/sft_train_data.jsonl` as `sft_train_data.jsonl`
- `data/sft_validation_data.jsonl` as `sft_validation_data.jsonl`

The notebook also searches `/content/drive/MyDrive/` for those filenames.

The validation JSONL is derived from `data/validation_questions.csv` and
`data/validation_target.csv`. Its 864 records have this schema:

```json
{"id":"ID_XLWWVM40IW","question":"...","response":"\\boxed{C8}"}
```

Each validation question has four suffixed source targets (`_1` through `_4`).
The prepared artifact collapses them only after checking that all four labels
are identical. It contains exactly 108 questions for each class C1–C8.

## Reasoning and validation format

Synthetic SFT responses are converted at runtime to:

```text
<think>
...validated synthetic reasoning...
</think>

\boxed{R#}
```

The randomized `R1`–`R8` identifiers are intentional: every synthetic question
defines its own randomized candidate mapping. Validation questions use the
original `C1`–`C8` mapping.

Do not synthesize thoughts for validation examples. Their answer-only
completion supplies a small completion-only validation loss during training,
while the primary evaluation asks the model to generate its own reasoning and
final label. The notebook reports:

- held-out completion loss after each epoch;
- strict generated-label accuracy;
- strict boxed-format rate;
- per-class accuracy;
- a confusion matrix; and
- complete per-question generations.

Validation examples are passed only as `eval_dataset`; they are never included
in optimizer updates.

## Outputs

The notebook restores the checkpoint with the best validation loss and writes:

- `qwen35_4b_sft_lora.zip` — the LoRA adapter, tokenizer, and GRPO handoff
  metadata;
- `validation_metrics.json`; and
- `validation_predictions.jsonl`.

Running generated evaluation across all 864 long questions can take substantial
time on a T4. `EVAL_LIMIT` may be set for a smoke test, but official metrics
must leave it as `None`.

## GRPO handoff

The referenced `Qwen3_(4B)-GRPO` example is useful conceptually but is not
adapter-compatible as written. It loads `unsloth/Qwen3-4B-Base`, enables a
different inference path, and teaches custom `<start_working_out>` tags.

The later GRPO stage must instead:

1. load `unsloth/Qwen3.5-4B`;
2. attach `qwen35_4b_sft_lora`;
3. use `fast_inference=False`;
4. retain the saved tokenizer and native Qwen chat template;
5. keep `<think>...</think>` followed by a final boxed identifier; and
6. reward the strict final `\boxed{...}` label.

Do not add the official validation questions to either SFT or GRPO training.
