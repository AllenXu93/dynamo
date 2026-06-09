#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch the native router as a STANDALONE HTTP frontend — no gateway/Envoy and
# no Dynamo runtime (no etcd/NATS). It reuses the exact same routing core as the
# ext_proc EPP (RoutingService + Kubernetes pod-reflector discovery + ZMQ KV
# ingestion), then proxies each request to the worker it selects.
#
# Workers are discovered by a plain label selector (DYN_EPP_POD_SELECTOR), so no
# InferencePool / HTTPRoute is involved. Serves OpenAI on DYN_EPP_HTTP_PORT.
#
# Arg 1 = HTTP listen port (default 8000).
cd /work
export DYN_DISCOVERY_BACKEND=mem POD_NAMESPACE="${POD_NAMESPACE:-default}" DYN_EPP_EXTERNAL=true
export DYN_MODEL_NAME="${DYN_MODEL_NAME:-Qwen/Qwen3-0.6B}" DYN_KV_CACHE_BLOCK_SIZE=16

# Gateway-free discovery: select worker pods directly by label + target port
# (no InferencePool CR). These two replace DYN_EPP_INFERENCE_POOL.
export DYN_EPP_POD_SELECTOR="${DYN_EPP_POD_SELECTOR:-app=epp-vanilla-vllm}"
export DYN_EPP_TARGET_PORT="${DYN_EPP_TARGET_PORT:-8000}"

# Standalone HTTP frontend instead of the ext_proc gRPC service.
export DYN_EPP_HTTP_FRONTEND=true DYN_EPP_HTTP_PORT="${1:-8000}"

# Load-aware routing from worker ZMQ KV-cache events (no NATS).
export DYN_OVERLAP_SCORE_WEIGHT=1.0 DYN_EPP_KV_EVENTS=true DYN_EPP_KV_EVENT_PORT=5557
export RUST_LOG="${RUST_LOG:-info}"

# Disaggregated P/D is supported here too: set the role-partition envs below and
# point at prefill+decode workers (see ../vanilla-vllm-disagg). The frontend then
# forwards x-prefiller-host-port to the decode-side routing sidecar, exactly as
# the gateway EPP does.
#   export DYN_EPP_ROLE_LABEL="nvidia.com/dynamo-component-type"
#   export DYN_EPP_PREFILL_ROLE_VALUES=prefill DYN_EPP_DECODE_ROLE_VALUES=decode
#   export DYN_ENFORCE_DISAGG=true

exec -a dyn-epp-frontend /work/dynamo/target/release/dynamo-ext-proc
