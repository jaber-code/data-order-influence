# Data Order Influence

Research code studying **how the order in which classes are introduced affects incremental (class‑incremental) training** of image classifiers. A model is trained on a growing subset of classes, one increment at a time, and we compare final/again‑seen accuracy across different class orderings and different epoch‑budget schedules.

## Idea

Instead of training on all classes at once, start with `initial_num_classes` and add `step_num_classes` each increment until all classes are included. Two things are varied:

- **Class order** – the sequence in which classes are added (alphabetical, random, semantically most‑dissimilar‑first, most‑similar‑first, hybrids).
- **Epoch schedule** – how a fixed total training budget is spread over increments: `static` (equal), `dec` (more epochs early), `inc` (more epochs late). See `data_order_core/incremental_scheduler.py`.

The learning rate is re‑scaled per increment based on how many classes are currently active.

## Components

| Path | Purpose |
|------|---------|
| `classes_order_generator/` | Builds the class‑order files. Each class name is first expanded into a short definition by an **LLM** (self‑hosted Llama‑3.1‑70B behind an OpenAI‑compatible `/v1/chat/completions` endpoint), then pairwise **semantic similarity** on those definitions (SBERT / BERT / WordNet) drives a greedy "most dissimilar first" ordering. Also emits alphabetical and random orders. |
| `data_order_core/` | The training pipeline: dataset loaders (CIFAR‑10, ImageNet), model factory (small CNNs, NFNet), incremental trainer, AMP + optional DDP multi‑GPU, CSV logging of accuracy per increment. |
| `plotting/` | Reads the training CSV logs and produces comparison plots (accuracy curves, per‑class effort, cumulative performance, order‑vs‑order comparisons). |
| `*_order.txt`, `hybrid-*.txt` | Pre‑generated class orderings consumed by the trainer. |

## Running

Both entry points are configured through YAML, not CLI flags.

**Generate class orders**

```bash
python classes_order_generator/generator.py      # writes generated_files/*_order.txt
```

The LLM label‑expansion is a one‑time pre‑compute step (its output is cached in `extended_names.txt`, and the call is toggled by `_disable_extend_words_generator`).

**Incremental training**

```bash
python data_order_core/main.py                    # reads data_order_core/config/config.yaml
```

Key config fields (`data_order_core/config/config.yaml`):

- `select_dataset_type` – `cifar10` / `image_net_1k`
- `selected_model` – `custom_cnn_cifar` / `nfnet` / `custom_cnn_imgnet`
- `inc_training_type` – class order: `alph`, `random`, `most_diss_first`, …
- `inc_decay_type` – epoch schedule: `static` / `dec` / `inc`
- `initial_num_classes`, `step_num_classes`, `epochs`, `batch_size`, `learning_rate`

**Plots**

```bash
python plotting/plotting.py                       # point it at a training_data_*.csv log
```

The `.slurm` files show the multi‑GPU cluster setup (containerised PyTorch, `torchrun`‑style `WORLD_SIZE`/`LOCAL_RANK`).

## Stack

PyTorch (AMP, DDP, `torch.compile`), pytorch‑ignite (warmup LR scheduler), timm, sentence‑transformers / transformers / NLTK for the semantic ordering, pandas + matplotlib + seaborn for plots.

## Highlights

- **AI / LLM** – calls an LLM API (OpenAI‑compatible chat endpoint) with a purpose‑built prompt to enrich class labels; uses transformer sentence embeddings and cosine similarity to order data semantically.
- **Modern architecture** – config‑driven (YAML), factory / singleton / strategy patterns for models, datasets and similarity backends, `Enum`‑typed options, `ABC` interfaces so a new model, dataset or similarity method is a drop‑in class.
- **Automation & scale** – Slurm batch scripts, containerised runs, multi‑GPU distributed training, CSV experiment logging feeding an automated plotting pipeline.
- **Python** – the whole project; NumPy/pandas data handling, `dataclass`es, decorators (`@timer`), context managers.
