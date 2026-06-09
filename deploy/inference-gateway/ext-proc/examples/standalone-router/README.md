<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Standalone KV-aware router with an HTTP frontend (no gateway, no Dynamo runtime)

This example runs the Dynamo KV-aware router as a **standalone OpenAI HTTP
frontend** — no gateway/Envoy, and no Dynamo runtime (no etcd/NATS, no
`dynamo.vllm` worker). It serves stock `vllm serve` pods discovered from
Kubernetes by a plain label selector.

```
client ─▶ standalone HTTP frontend (dynamo-ext-proc, DYN_EPP_HTTP_FRONTEND=true)
              │  1) RoutingService: tokenize → hints → KV/load-aware select
              │  2) resolve worker_id → pod ip:port (pod reflector)
              │  3) proxy the request to that pod, stream the response back
              ▼
        vanilla vLLM pod  (vllm serve :8000, KV events over ZMQ)
```

## Why this exists — feature parity with the EPP router

The routing business logic lives in **`dynamo_llm::kv_router::routing`**
(`RoutingService`: tokenize → extract hints → select prefill/decode), shared by
both front ends. The gateway EPP and this standalone frontend run the *same*
`Router`:

| Concern | Gateway EPP (ext_proc) | Standalone frontend (this) |
| --- | --- | --- |
| Routing decision | `RoutingService::route` | `RoutingService::route` (same) |
| Worker discovery | K8s pod reflector | K8s pod reflector (same) |
| KV load ingestion | per-pod ZMQ listeners | per-pod ZMQ listeners (same) |
| `worker_id → ip:port` | pod reflector | pod reflector (same) |
| Request forwarding | **Envoy/gateway** does it | **the frontend** does it |
| Transport | ext_proc gRPC from Envoy | OpenAI HTTP directly |

The only thing the frontend adds over the EPP is the forwarding step Envoy
normally performs. Everything that decides *where a request goes* is identical,
so the standalone router has feature parity with the EPP router — including
KV/load-aware routing and (optionally) disaggregated P/D.

This is distinct from `python -m dynamo.frontend`, which requires the Dynamo
distributed runtime (etcd + NATS) for worker discovery. This frontend has no
such dependency.

## Files

* `workers-agg.yaml` — two aggregated stock `vllm serve` pods labeled
  `app=epp-vanilla-vllm`, publishing KV-cache events over ZMQ. Each pod is a
  single-rank worker; scale data parallelism by adding pods (the standard vLLM
  external-LB pattern).
* `run-frontend.sh` — launches the binary in HTTP-frontend mode with
  gateway-free, label-selector discovery.

## Run

```bash
kubectl apply -f workers-agg.yaml

# In the EPP/router pod (the dynamo-ext-proc binary built at
# /work/dynamo/target/release):
bash run-frontend.sh 8000

# From a shell with network access to the frontend (e.g. kubectl port-forward
# pod/<router-pod> 8000:8000):
curl -s http://127.0.0.1:8000/v1/models
curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":16}'
```

The frontend log prints the selected worker and endpoint per request
(`Standalone frontend routed request; proxying to worker`); load spreads across
the two pods as their ZMQ-reported load changes.

## Configuration

| Env | Default | Meaning |
| --- | --- | --- |
| `DYN_EPP_HTTP_FRONTEND` | `false` | `true` runs the HTTP frontend instead of the ext_proc gRPC service. |
| `DYN_EPP_HTTP_PORT` / `DYN_EPP_HTTP_HOST` | `8000` / `0.0.0.0` | Frontend listen address. |
| `DYN_EPP_POD_SELECTOR` | (Dynamo worker convention) | Label selector for worker pods — replaces the InferencePool. |
| `DYN_EPP_TARGET_PORT` | named port `http` | Port to reach each worker on. |
| `DYN_EPP_UPSTREAM_SCHEME` | `http` | Scheme used to proxy to workers. |
| `DYN_MODEL_NAME` | `vllm` | Model id advertised on `/v1/models`. |
| `DYN_EPP_KV_EVENTS` / `DYN_EPP_KV_EVENT_PORT` | `false` / `5557` | Consume worker ZMQ KV events for precise load/prefix routing. |

## Disaggregated P/D

Disaggregation works through this frontend as well: set the role-partition envs
(`DYN_EPP_ROLE_LABEL`, `DYN_EPP_PREFILL_ROLE_VALUES`,
`DYN_EPP_DECODE_ROLE_VALUES`, `DYN_ENFORCE_DISAGG=true`) and point the selector
at prefill+decode workers (see `../vanilla-vllm-disagg`). The frontend selects a
decode worker and a prefill worker, then forwards `x-prefiller-host-port` to the
decode-side P/D routing sidecar — the same headers the gateway EPP emits.
