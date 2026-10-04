# DART-ES: Evolution Strategies Fine-Tuning on GSM8K

This repository contains DART-ES, a multi-GPU evolution-strategies (ES)
training implementation for fine-tuning a Qwen2.5 causal language model on
GSM8K. DART-ES combines three data-adaptation modules:

1. difficulty-aware sample weighting based on an exponential moving average
   of per-sample pass rates;
2. prioritized replay of rarely solved samples; and
3. online updates of sample difficulty statistics from the ES population.

## Repository layout

```text
.
├── train_dart_es_gsm8k.py         # DART-ES training entry point
├── eval_gsm8k_vllm.py             # GSM8K evaluation with vLLM
├── scripts/run_dart_es.sh         # Reference experiment configuration
├── gsm8k/
│   ├── gsm8k_main_train.json      # Processed training split
│   ├── data/gsm8k_main_test.json  # Processed test split (1,319 examples)
│   └── reward_function.py         # Format and numerical-answer rewards
├── utils/
│   └── worker_extn.py             # vLLM worker-side DART-ES operations
└── requirements.txt
```

## Environment

The reference environment uses Python 3.10, Linux, CUDA-capable NVIDIA GPUs,
PyTorch 2.8.0, vLLM 0.11.0, Ray 2.51.1, and Transformers 4.57.0. One GPU is
allocated to each vLLM engine.

Install a PyTorch build that matches the CUDA runtime of the target machine,
then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the DART-ES reference configuration

The default launcher uses eight GPUs and loads
`Qwen/Qwen2.5-0.5B-Instruct`:

```bash
bash scripts/run_dart_es.sh
```

The model, GPU list, engine count, Python executable, and output directory can
be overridden without editing the script:

```bash
MODEL_NAME=/path/to/model \
CUDA_DEVICES=0,1,2,3 \
NUM_ENGINES=4 \
RUN_NAME=outputs/dart-es-my-run \
bash scripts/run_dart_es.sh
```

Additional command-line arguments can be appended to the launcher:

```bash
bash scripts/run_dart_es.sh --epochs 1 --chunk_size 32
```

Use the same number of comma-separated CUDA devices and vLLM engines. Training
outputs are written below `outputs/` by default.

## Evaluate on GSM8K

The evaluation script and processed test split are copied without changes from
[ESSAM](https://github.com/szs777/ESSAM/tree/1aee406adaf9c3346726ee777522d8fb034c0b1f).
They use the dependencies and worker utilities already included in this repository.

Run from the repository root on a CUDA-capable machine:

```bash
python eval_gsm8k_vllm.py \
  --model_id Qwen/Qwen2.5-0.5B-Instruct \
  --trained_model_path /path/to/pytorch_model.pth \
  --eval_data_path gsm8k/data/gsm8k_main_test.json \
  --tensor_parallel_size 1 \
  --batch_size 32 \
  --dtype float16 \
  --output_dir outputs/gsm8k-eval \
  --save_responses
```

Replace `/path/to/pytorch_model.pth` with your trained checkpoint. DART-ES writes
final checkpoints to
`<experiment_dir>/dart_es_gsm8k_<timestamp>/model_saves/final_model_epoch_<N>/pytorch_model.pth`.
Set `--model_id` to the same base model used during training; the example matches
the default training launcher.

Pass `--eval_data_path` explicitly as shown because the original evaluation
script's default points to a different directory. Keep `--tensor_parallel_size 1`
for this script's single-GPU allocation. All 1,319 test examples are evaluated
unless `--eval_samples` is supplied. Decoding is greedy by default; use
`--do_sample` to enable sampling.

Evaluation writes `summary.json` to the output directory, including answer
accuracy and reward statistics. `--save_responses` also writes
`eval_detailed_results.json` with individual predictions.
