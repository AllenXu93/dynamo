---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: DGDR Examples
---

Practical examples for deploying with `DynamoGraphDeploymentRequest` (DGDR).
The DGDR workflow can use native AIC estimates, optional bootstrap profiling
data, or live FPM warmup depending on the model/backend combination. For DGDR
concepts, see the [DGDR Reference](dgdr.md). For profiling concepts, see the
[Profiler Guide](../components/profiler/profiler-guide.md).

## DGDR Examples

### Minimal DGDR with AIC (Fastest)

The simplest way to generate a deployment from native AIC estimates. Uses AI
Configurator for offline profiling (20-30 seconds instead of hours):

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: sla-aic
spec:
  model: Qwen/Qwen3-32B
  backend: vllm
  image: "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.1.1"  # dynamo-frontend for Dynamo < 1.1.0
```

Deploy:
```bash
export NAMESPACE=your-namespace
# Save the manifest above as sla-aic.yaml first.
kubectl apply -f sla-aic.yaml -n $NAMESPACE
```

### Online Profiling (Real Measurements)

Standard online profiling runs real GPU measurements for more accurate results. Takes 2-4 hours:

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: sla-online
spec:
  model: meta-llama/Llama-3.3-70B-Instruct
  backend: vllm
  image: "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.1.1"  # dynamo-frontend for Dynamo < 1.1.0
  searchStrategy: thorough
```

Deploy:
```bash
# Save the manifest above as sla-online.yaml first.
kubectl apply -f sla-online.yaml -n $NAMESPACE
```

> **Note**: Starting with Dynamo 1.0.0 (DGDR API version v1beta1), DGDR fields use structured spec fields (e.g., `spec.workload`, `spec.sla`, `spec.hardware`) instead of the nested `profilingConfig.config` blob used in v1alpha1.

## Additional DGDR Patterns

### MoE Models (SGLang)

For Mixture-of-Experts models like DeepSeek-R1, use SGLang backend:

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: sla-moe
spec:
  model: deepseek-ai/DeepSeek-R1
  backend: sglang
  image: "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.1.1"  # dynamo-frontend for Dynamo < 1.1.0
```

Deploy:
```bash
# Save the manifest above as sla-moe.yaml first.
kubectl apply -f sla-moe.yaml -n $NAMESPACE
```

### Using Existing DGD Configs (Custom Setups)

Reference an existing DynamoGraphDeployment config via ConfigMap:

**Step 1: Create ConfigMap from your DGD config:**

```bash
kubectl create configmap deepseek-r1-config \
  --from-file=disagg.yaml=/path/to/your/disagg.yaml \
  --namespace $NAMESPACE \
  --dry-run=client -o yaml | kubectl apply -f -
```

**Step 2: Reference it in your DGDR:**

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: deepseek-r1
spec:
  model: deepseek-ai/DeepSeek-R1
  backend: sglang
  image: "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.1.1"  # dynamo-frontend for Dynamo < 1.1.0
```

The profiler uses the DGD config from the ConfigMap as a **base template**, then optimizes it based on your SLA targets. The controller automatically injects `spec.model` and `spec.backend` into the final configuration.

### Inline Configuration (Simple Use Cases)

For simple use cases without a custom DGD config, provide the configuration directly in the v1beta1 DGDR spec fields. The profiler auto-generates a basic DGD configuration:

```yaml
spec:
  workload:
    isl: 8000
    osl: 200

  sla:
    ttft: 200.0
    itl: 10.0

  hardware:
    gpuSku: h200_sxm

  searchStrategy: rapid
```

### Simulation with Mocker

Deploy a mocker backend that simulates GPU timing behavior without real GPUs. Useful for:
- Large-scale experiments without GPU resources
- Testing profiling behavior and infrastructure
- Validating deployment configurations

```yaml
spec:
  model: <model-name>
  backend: trtllm  # Real backend for profiling
  features:
    mocker:
      enabled: true  # Deploy mocker instead of real backend

  image: "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.1.1"  # dynamo-frontend for Dynamo < 1.1.0
```

Profiling runs against the real backend (via GPUs or AIC). The mocker deployment then uses profiling data to simulate realistic timing.

### Model Cache PVC (0.8.1+)

For large models, use a pre-populated PVC instead of downloading from HuggingFace:

See [SLA-Driven Profiling](../components/profiler/profiler-guide.md) for configuration details.

## Advanced DGDR Patterns

### Review Before Deploy (autoApply: false)

Disable auto-deployment to inspect the generated DGD:

```yaml
spec:
  autoApply: false
```

After profiling completes:

```bash
# Extract and review generated DGD
kubectl get dgdr sla-aic -n $NAMESPACE \
  -o jsonpath='{.status.profilingResults.selectedConfig}' > my-dgd.yaml

# Review and modify as needed
vi my-dgd.yaml

# Deploy manually
kubectl apply -f my-dgd.yaml -n $NAMESPACE
```

### Profiling Artifacts with PVC

Save detailed profiling artifacts (plots, logs, raw data) to a PVC:

```yaml
spec:
  workload:
    isl: 3000
    osl: 150

  sla:
    ttft: 200
    itl: 20
```

Setup:
```bash
export NAMESPACE=your-namespace
deploy/utils/setup_benchmarking_resources.sh
```

Access results:
```bash
kubectl apply -f deploy/utils/manifests/pvc-access-pod.yaml -n $NAMESPACE
kubectl wait --for=condition=Ready pod/pvc-access-pod -n $NAMESPACE --timeout=60s
kubectl cp $NAMESPACE/pvc-access-pod:/data ./profiling-results
kubectl delete pod pvc-access-pod -n $NAMESPACE
```

## Related Documentation

- [DGDR Reference](dgdr.md) -- DGDR field reference and lifecycle
- [Profiler Guide](../components/profiler/profiler-guide.md) -- Profiling workflow
