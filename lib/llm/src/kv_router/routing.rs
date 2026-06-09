// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared request-routing helpers used by both the gateway Endpoint Picker
//! (ext_proc) and the standalone router, so the routing business logic has a
//! single implementation rather than one copy per front end.
//!
//! This is the first piece of that shared surface: parsing the router-relevant
//! hints out of an OpenAI request body, plus the tokenizer abstraction. The
//! select/book orchestration moves here incrementally.

use std::collections::HashSet;
use std::sync::Arc;

use dynamo_kv_router::config::RouterConfigOverride;
use dynamo_kv_router::protocols::{RoutingConstraints, WorkerWithDpRank};

use crate::kv_router::prefill_router::PrefillQueryOutcome;
use crate::kv_router::{KvRouter, PrefillRouter};
use crate::preprocessor::OpenAIPreprocessor;
use crate::types::openai::chat_completions::NvCreateChatCompletionRequest;

/// Parse routing hints — `priority_jump` and expected output length (`osl`) —
/// from `nvext.agent_hints`, directly from the raw request body so they survive
/// every tokenization mode (a sidecar tokenizer returns tokens only) and both
/// request shapes (chat + completion). Returns `(0.0, None)` on parse failure.
///
/// Negative priorities clamp to `0.0` so a low-priority hint never pushes a
/// request behind FCFS arrivals (matches the standalone preprocessor lift in
/// `lib/llm/src/preprocessor.rs`); falls back to the deprecated
/// `latency_sensitivity` alias.
pub fn extract_hints(body_str: &str) -> (f64, Option<u32>) {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(body_str) else {
        return (0.0, None);
    };
    let hints = v.get("nvext").and_then(|n| n.get("agent_hints"));
    let priority_jump = hints
        .and_then(|h| {
            h.get("priority")
                .and_then(|p| p.as_i64())
                .map(|p| p.max(0) as f64)
                .or_else(|| h.get("latency_sensitivity").and_then(|l| l.as_f64()))
        })
        .unwrap_or(0.0);
    let osl = hints
        .and_then(|h| h.get("osl"))
        .and_then(|o| o.as_u64())
        .map(|o| o as u32);
    (priority_jump, osl)
}

/// Tokenize a request body into the token IDs used for routing (KV-prefix
/// matching / load projection). Implemented per front end so the routing core
/// stays tokenizer-agnostic and the gateway EPP and the standalone router share
/// it rather than each carrying a copy.
#[async_trait::async_trait]
pub trait RequestTokenizer: Send + Sync {
    async fn tokenize_for_routing(&self, body: &str) -> anyhow::Result<Vec<u32>>;
}

/// Exact in-process tokenization via the model's [`OpenAIPreprocessor`] — parity
/// with a Dynamo worker / the standalone router (applies the chat template, then
/// tokenizes).
pub struct PreprocessorTokenizer {
    preprocessor: Arc<OpenAIPreprocessor>,
}

impl PreprocessorTokenizer {
    pub fn new(preprocessor: Arc<OpenAIPreprocessor>) -> Self {
        Self { preprocessor }
    }
}

#[async_trait::async_trait]
impl RequestTokenizer for PreprocessorTokenizer {
    async fn tokenize_for_routing(&self, body: &str) -> anyhow::Result<Vec<u32>> {
        let request: NvCreateChatCompletionRequest = serde_json::from_str(body)?;
        let formatted = self
            .preprocessor
            .apply_template(&request)?
            .unwrap_or_default();
        Ok(self.preprocessor.tokenize(&formatted)?.token_ids().to_vec())
    }
}

/// Load-aware placeholder: no real tokenization; returns a token vector sized to
/// the request so the scheduler sees `isl > 0` (pair with overlap weight 0 for
/// pure load-aware routing).
pub struct LoadPlaceholderTokenizer;

#[async_trait::async_trait]
impl RequestTokenizer for LoadPlaceholderTokenizer {
    async fn tokenize_for_routing(&self, body: &str) -> anyhow::Result<Vec<u32>> {
        const AVG_CHARS_PER_TOKEN: usize = 4;
        Ok(vec![0u32; (body.len() / AVG_CHARS_PER_TOKEN).max(1)])
    }
}

/// Outcome of [`RoutingService::route`]: the chosen decode worker, the chosen
/// prefill worker when disaggregated, and the tokens/hints the caller needs for
/// load bookkeeping and (for the EPP) header emission.
pub struct RoutingDecision {
    pub tokens: Vec<u32>,
    pub priority_jump: f64,
    pub osl: Option<u32>,
    pub decode_worker: WorkerWithDpRank,
    /// `Some((worker_id, dp_rank))` when the request is routed disaggregated.
    pub prefill_worker: Option<(u64, Option<u32>)>,
    pub is_disaggregated: bool,
}

/// The shared per-request routing pipeline — tokenize, extract hints, select a
/// prefill worker (disaggregated) then a decode worker. Used by both the gateway
/// Endpoint Picker and the standalone router so the routing *decision* has one
/// implementation; load bookkeeping and front-end transport (ext_proc header
/// mutation, HTTP forwarding) stay in the respective adapters.
pub struct RoutingService {
    tokenizer: Arc<dyn RequestTokenizer>,
    decode_router: Arc<KvRouter>,
    prefill_router: Arc<PrefillRouter>,
}

impl RoutingService {
    pub fn new(
        tokenizer: Arc<dyn RequestTokenizer>,
        decode_router: Arc<KvRouter>,
        prefill_router: Arc<PrefillRouter>,
    ) -> Self {
        Self {
            tokenizer,
            decode_router,
            prefill_router,
        }
    }

    /// Select a prefill worker. Returns `(worker_id, dp_rank)`. Backpressure is
    /// surfaced as an error so the caller's enforce-disagg / aggregated-fallback
    /// logic can decide whether to fail the request or serve decode-only.
    pub async fn route_prefill(
        &self,
        tokens: &[u32],
        priority_jump: f64,
        allowed_worker_ids: Option<HashSet<u64>>,
    ) -> anyhow::Result<(u64, Option<u32>)> {
        if let Some(ref ids) = allowed_worker_ids {
            self.prefill_router.register_workers(ids);
        }

        let outcome = self
            .prefill_router
            .query_prefill_worker(
                tokens,
                None,
                false,
                None,
                priority_jump,
                allowed_worker_ids,
                RoutingConstraints::default(),
            )
            .await
            .map_err(|e| anyhow::anyhow!("Prefill query failed: {:?}", e))?;

        match outcome {
            PrefillQueryOutcome::Routed { worker_id, dp_rank } => Ok((worker_id, dp_rank)),
            PrefillQueryOutcome::Backpressure {
                reason,
                queued_isl_tokens,
                max_queued_isl_tokens,
            } => Err(anyhow::anyhow!(
                "Prefill router backpressure: {:?} (queued_isl_tokens={}, max={:?})",
                reason,
                queued_isl_tokens,
                max_queued_isl_tokens
            )),
        }
    }

    /// Select a decode worker. Returns `(WorkerWithDpRank, overlap_blocks)`.
    pub async fn route_decode(
        &self,
        tokens: &[u32],
        is_disaggregated: bool,
        priority_jump: f64,
        expected_output_tokens: Option<u32>,
        allowed_worker_ids: Option<HashSet<u64>>,
    ) -> anyhow::Result<(WorkerWithDpRank, u32)> {
        if let Some(ref ids) = allowed_worker_ids {
            self.decode_router.register_workers(ids);
        }

        let config_override = if is_disaggregated {
            Some(RouterConfigOverride {
                overlap_score_credit: Some(0.0),
                assume_kv_reuse: Some(false),
                track_prefill_tokens: Some(false),
                ..Default::default()
            })
        } else {
            None
        };

        self.decode_router
            .find_best_match(
                None,
                tokens,
                None,
                config_override.as_ref(),
                false,
                None,
                priority_jump,
                expected_output_tokens,
                allowed_worker_ids,
                RoutingConstraints::default(),
            )
            .await
            .map_err(|e| anyhow::anyhow!("Decode query failed: {:?}", e))
    }

    /// Run the full routing decision for a request body against the given
    /// prefill/decode candidate sets. Prefill is attempted only when the prefill
    /// set is non-empty; on prefill failure with `enforce_disagg`, this errors,
    /// otherwise it falls back to aggregated (decode-only).
    pub async fn route(
        &self,
        body: &str,
        prefill_ids: HashSet<u64>,
        decode_ids: HashSet<u64>,
    ) -> anyhow::Result<RoutingDecision> {
        let tokens = self.tokenizer.tokenize_for_routing(body).await?;
        let (priority_jump, osl) = extract_hints(body);

        let prefill_result = if prefill_ids.is_empty() {
            Err(anyhow::anyhow!("no prefill-role workers in candidate set"))
        } else {
            self.route_prefill(&tokens, priority_jump, Some(prefill_ids))
                .await
        };

        let is_disaggregated = match &prefill_result {
            Ok(_) => true,
            Err(e) => {
                if self.prefill_router.enforce_disagg() {
                    return Err(anyhow::anyhow!(
                        "prefill routing failed under enforce_disagg: {e}"
                    ));
                }
                tracing::debug!(error = %e, "Prefill routing failed; falling back to aggregated mode");
                false
            }
        };

        let (decode_worker, _overlap) = self
            .route_decode(
                &tokens,
                is_disaggregated,
                priority_jump,
                osl,
                Some(decode_ids),
            )
            .await?;

        Ok(RoutingDecision {
            tokens,
            priority_jump,
            osl,
            decode_worker,
            prefill_worker: prefill_result.ok(),
            is_disaggregated,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hints_parsed_from_nvext_agent_hints() {
        let (p, osl) =
            extract_hints(r#"{"prompt":"hi","nvext":{"agent_hints":{"priority":5,"osl":128}}}"#);
        assert_eq!(p, 5.0);
        assert_eq!(osl, Some(128));
    }

    #[test]
    fn negative_priority_clamped_and_absent_hints_default() {
        assert_eq!(
            extract_hints(r#"{"nvext":{"agent_hints":{"priority":-3}}}"#).0,
            0.0
        );
        assert_eq!(extract_hints(r#"{"prompt":"hi"}"#), (0.0, None));
        assert_eq!(extract_hints("not json"), (0.0, None));
    }
}
