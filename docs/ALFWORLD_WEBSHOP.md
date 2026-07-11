# ALFWorld & WebShop usage

COS-PLAY supports the two SkillRL text benchmarks in two modes:

1. **Inference-time skill control** (no training): load a frozen SkillRL
   checkpoint, add the COS-PLAY skill controller (skill selection + state
   ledger + action repair), and run paired evaluations against the plain
   checkpoint.
2. **Co-evolution training**: run the full COS-PLAY loop (rollout
   segmentation → contracts → curation + GRPO) with `webshop` / `alfworld`
   as games.

## 1. Setup

```bash
bash scripts/install_text_envs.sh        # ALFWorld data + WebShop deps
```

For the **official SkillRL WebShop protocol** (Pyserini/Lucene search
instead of the BM25 shim) you additionally need:

```bash
sudo apt-get install -y openjdk-21-jre-headless
uv pip install 'pyserini==0.17.0' faiss-cpu
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

# Build the Lucene index from the 1k-product catalog (once):
#   writes resources_1k/documents.jsonl and indexes/ inside the SkillRL
#   webshop package (see scripts/eval_webshop_skillrl_official.py header
#   for the expected paths)
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input  <SkillRL>/agent_system/environments/env_package/webshop/webshop/search_engine/resources_1k \
  --index  <SkillRL>/agent_system/environments/env_package/webshop/webshop/search_engine/indexes \
  --generator DefaultLuceneDocumentGenerator --threads 1 \
  --storePositions --storeDocvectors --storeRaw
```

Prerequisites: `SkillRL` repo checked out at `/workspace/SkillRL`, WebShop
data (`items_shuffle_1000.json`, `items_ins_v2_1000.json`,
`items_human_ins.json`) under `/workspace/WebShop/data`.

## 2. Serve the frozen SkillRL checkpoints

```bash
# Download + merge (WebShop RL actor shards → HF format)
python scripts/download_webshop_rl.py
python scripts/merge_skillrl_rl_ckpt.py --out /workspace/models/webshop-7b-rl-hf

# Serve with vLLM (one GPU each)
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/alfworld-7b-rl-hf \
  --served-model-name skillrl-alfworld --port 8010

python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/webshop-7b-rl-hf \
  --served-model-name skillrl-webshop --port 8011
```

## 3. ALFWorld evaluations

```bash
# SkillRL-native replication (their prompts/templates, our wrapper)
python scripts/eval_skillrl_native.py \
  --episodes 100 --split eval_out_of_distribution

# Paired eval with COS-PLAY skill banks vs SkillRL's original skills
python scripts/eval_with_cosplay_skills.py \
  --skill-bank labeling/output/skillrl_seed_bank/alfworld/skill_bank.jsonl \
  --episodes 100 --split eval_out_of_distribution --deterministic-order
python scripts/eval_with_cosplay_skills.py \
  --their-skills --episodes 100 --split eval_out_of_distribution --deterministic-order

# Official SkillRL ID protocol (their env, seed schedule 1000+i,
# eval_in_distribution, SkillsOnlyMemory, native `won` metric)
export PYTHONPATH=/workspace/SkillRL/agent_system/environments/env_package/alfworld:/workspace/SkillRL:$(pwd)
python scripts/eval_alfworld_skillrl_official.py --mode skillrl --episodes 64 \
  --out runs/eval_alfworld_official_skillrl_64.jsonl
python scripts/eval_alfworld_skillrl_official.py --mode cosplay --episodes 64 \
  --out runs/eval_alfworld_official_cosplay_64.jsonl
```

## 4. WebShop evaluations

```bash
# Fast COS-PLAY controller vs plain checkpoint (BM25-shim env, paired goals)
python scripts/eval_webshop_cosplay_fast.py \
  --base-url http://localhost:8011/v1 --model skillrl-webshop \
  --episodes 100 --out runs/eval_webshop_cosplay_100.jsonl

# Official SkillRL protocol (their env + Lucene index, validation goal
# schedule RandomState(1000).choice(500), 15 steps, won/task_score metrics)
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PYTHONPATH=$(pwd):/workspace/SkillRL:/workspace/SkillRL/agent_system/environments/env_package/webshop/webshop
python scripts/eval_webshop_skillrl_official.py --mode skillrl --episodes 64 \
  --out runs/eval_webshop_official_skillrl_64.jsonl
python scripts/eval_webshop_skillrl_official.py --mode cosplay --episodes 64 \
  --out runs/eval_webshop_official_cosplay_64.jsonl
python scripts/eval_webshop_skillrl_official.py --mode guard --episodes 64 \
  --out runs/eval_webshop_official_guard_64.jsonl
```

Modes of `eval_webshop_skillrl_official.py`:

| Mode | Prompt | Action layer |
|------|--------|--------------|
| `skillrl` | native SkillsOnlyMemory block | none (baseline) |
| `cosplay` | native block + COS-PLAY skill controller block | full repair/veto |
| `guard`   | native block only | minimal guardrails (loop-break, buy-now option check) |

## 5. Reference results (frozen RL checkpoints, paired tasks)

| Setting | Horizon | SkillRL | +COS-PLAY |
|---------|---------|---------|-----------|
| ALFWorld OOD (`valid_unseen`, paired-100) | 50 steps | 71% | **92%** |
| ALFWorld official ID (`valid_seen`, 64 tasks) | 50 steps | 67.2% | **70.3%** |
| WebShop weak-retrieval (BM25 shim, paired-100 synthetic) | 15 steps | 36% | **54%** |
| WebShop official Lucene (64 tasks) | 15 steps | 75.0% | 75.0% (non-degrading) |

Gains track the amount of *recoverable* failure: long-horizon /
off-distribution settings benefit most; on the saturated official WebShop
protocol the controller is neutral.

## 6. Co-evolution training

```bash
python scripts/run_coevolution.py --games alfworld \
  --alfworld-split train --alfworld-eval-split eval_out_of_distribution \
  --load-adapters-from runs/sft_textenv_v5/adapters_flat \
  --seed-bank-dir labeling/output/skillrl_seed_bank

# or both text envs sequentially:
bash scripts/run_textenv_coevo_pilot.sh
```
