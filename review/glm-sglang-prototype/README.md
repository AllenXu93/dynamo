<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM SGLang processor prototype review

This branch preserves an isolated code prototype and its review material for the
repository owner's inspection. **It does not install or enable the prototype in
this checkout and is not ready for an implementation PR.** The code is provided
as [prototype.patch](prototype.patch), against the original development snapshot.
It does not apply cleanly to the current upstream base (`b20fe6fe94`).

## Review files

- [Code patch](prototype.patch): only the proposed GLM repair; unrelated local
  changes are excluded.
- [Reproduction inputs](repro-cases.json): public schemas, fixed model output,
  expected events and prototype observations.
- [Validation](validation.json): controlled paired output-pipeline replay results
  and their limitations.
- [Review findings](review.txt): correctness and upstream integration blockers.
- [Issue draft](issue-body.txt), [lightweight DEP draft](dep-body.txt), and
  [provisional PR description](pr-body.txt): discussion material, not filed issues.

## Validation scope

The development replay invoked Dynamo's `_generate_and_stream` and real
postprocessor with fixed routed output and a byte test tokenizer. With authored
chunks, matches against the reference increased from 205/301 to 300/301. These
counts use explicitly supplied fixture starting states. They do **not** validate
public `generator()` initialization, current upstream main, real GLM tokenization,
a complete wheel build, HTTP serving, model accuracy or performance.

The remaining empty-tools case has identical output before and after: Dynamo
keeps tool markup as text when no tools are offered, while the generic parser
reference still recognizes calls.

## Must resolve before an implementation PR

1. Align and test production inline/pool initialization without test-only state
   overrides; the current replay and production paths can select different states.
2. Pass the actual constrained argument-framing policy. An object argument whose
   JSON string contains literal `</arg_value>` can currently be truncated.
3. Port to current main while preserving its stop-token and guided-content logic,
   reconcile the existing parser dependency, and remove already-upstream changes.
4. Build the full wheel, rerun upstream tests, and review per-chunk JSON/FFI/GIL
   allocation costs.

No community issue, DEP or upstream PR has been filed by this review branch.
Internal environment paths, private registry names and raw infrastructure logs
are intentionally absent from these files.
