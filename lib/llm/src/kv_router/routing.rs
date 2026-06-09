// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared request-routing helpers used by both the gateway Endpoint Picker
//! (ext_proc) and the standalone router, so the routing business logic has a
//! single implementation rather than one copy per front end.
//!
//! This is the first piece of that shared surface: parsing the router-relevant
//! hints out of an OpenAI request body. The tokenizer abstraction and the
//! select/book orchestration move here incrementally.

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hints_parsed_from_nvext_agent_hints() {
        let (p, osl) = extract_hints(
            r#"{"prompt":"hi","nvext":{"agent_hints":{"priority":5,"osl":128}}}"#,
        );
        assert_eq!(p, 5.0);
        assert_eq!(osl, Some(128));
    }

    #[test]
    fn negative_priority_clamped_and_absent_hints_default() {
        assert_eq!(extract_hints(r#"{"nvext":{"agent_hints":{"priority":-3}}}"#).0, 0.0);
        assert_eq!(extract_hints(r#"{"prompt":"hi"}"#), (0.0, None));
        assert_eq!(extract_hints("not json"), (0.0, None));
    }
}
