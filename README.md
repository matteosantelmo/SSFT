# SSFT — multi-instance LLM serving + inference (PoC)

Proof-of-concept that serves an 8B model as **2 replicas** (2 nodes × 4 GH200 GPUs,
**TP=4**, one replica per node), load-balanced via the **OpenTela mesh + serving-api
gateway**, waits for it to be ready, then sends example queries through the
**OpenAI API** and prints the outputs.

Serving is driven by the Swiss AI [`model-launch`](https://github.com/swiss-ai/model-launch)
framework (the `sml` CLI), vendored here as the `model_launch/` git submodule.

```
2 nodes × 4 GPUs ─► 2 × vllm replicas (TP=4, port 8080 each)
                         │  register on the OpenTela mesh under one model id
                         ▼
       gateway  api.swissai.svc.cscs.ch/v1  (OpenAI-compatible, load-balances)
                         ▲
               src/client.py  (openai SDK + CSCS API key)
```

Apertus-1.5 is served with **vllm** (the framework `sml` uses for this model
family). The replicas register on the OpenTela mesh under a single
`--served-model-name`; the gateway is the one endpoint that fans requests across
them — that's the "router". (The in-job `sglang` router is *not* used here: it
only ships in the sglang container, so it can't front vllm replicas.)

## Layout
- `scripts/setup.sh` — one-time: init the submodule, create `.venv`, install `sml` + `openai`.
- `scripts/run_poc.sh` — **launching half**: submits the serving job via `sml advanced`, then runs the client.
- `src/client.py` — **waiting half** (CLI): waits for readiness, then sends example prompts (temperature 0.7, top-p 0.95) and prints results.
- `src/serving.py` — helpers shared by the client: API-key resolution, Slurm job watch, gateway readiness polling.

## Run
```bash
./scripts/setup.sh   # once
sml init             # once, if not already done (slurm launcher + CSCS API key)
./scripts/run_poc.sh
```
Logs for the serving job land in `~/.sml/logs/<jobid>/` (`log.out` = orchestration,
`replica_*.out` = each vllm replica).

## Notes / gotchas
- **Container mounts.** The serving container mounts `/capstor` and `/iopsstor` but **not `/users`**. `scripts/run_poc.sh` runs `realpath` on `MODEL_PATH` so a `/users/...scratch...` symlink resolves to the real `/iopsstor` path the container can see.
- **Text-only Apertus-1.5.** The checkpoint ships an omni-modal tokenizer, so vllm tries to profile an `audio` modality and crashes with `Modality 'audio' not found`. We pass `--skip-mm-profiling` (plus `--trust-request-chat-template`) to serve it as plain text.
- **Routing.** `--router sglang` only works with `--framework sglang` (the router binary lives in the sglang container). For vllm replicas we use the default OpenTela routing and query the gateway.

## Config (env overrides for `scripts/run_poc.sh`)
| var | default | meaning |
|-----|---------|---------|
| `MODEL_PATH` | Apertus-1.5-8B thinking-token-fixed checkpoint | model to serve |
| `SERVED_MODEL_NAME` | `apertus-8b-thinking-$USER` | model id (must be unique on the mesh) |
| `PARTITION` | `normal` | Slurm partition |
| `RESERVATION` | `<none>` | Slurm reservation, e.g. `SD-69241-apertus-1-5-0` |
| `TIME_LIMIT` | `01:00:00` | job uptime |
| `REPLICAS` / `NODES_PER_REPLICA` / `TP_SIZE` | `2` / `1` / `4` | serving layout |
| `MAX_MODEL_LEN` | `8192` | vllm context length |
| `GPU_MEM_UTIL` | `0.8` | vllm KV-cache fraction per GPU |
| `TEMPERATURE` / `TOP_P` | `0.7` / `0.95` | sampling |
| `KEEP_ALIVE` | `1` | `0` runs `scancel` after the client finishes |

To scale, raise `REPLICAS` (total nodes = `REPLICAS × NODES_PER_REPLICA`); all
replicas register under the same model id and the gateway load-balances across them.

## Query a running service yourself
The serving job stays up until its time limit (or `scancel`). The client talks to
the gateway, so you only need the model id (and a CSCS API key, taken from
`$CSCS_API_KEY` or your `sml` config):
```bash
python src/client.py --served-model-name apertus-8b-thinking-$USER
# add --job-id <jobid> to also fail fast if the serving job dies
```
