// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standalone HTTP frontend over the Dynamo KV-aware router — no Dynamo runtime
//! (no etcd/NATS), no gateway/Envoy.
//!
//! This reuses the *exact same* [`crate::Router`] the ext_proc EPP uses — the
//! shared routing core ([`dynamo_llm::kv_router::routing::RoutingService`]:
//! tokenize -> hints -> prefill/decode select), the Kubernetes pod-reflector
//! discovery, the ZMQ KV-event ingestion, and the `worker_id -> pod IP:port`
//! resolution. The only thing it adds is the piece Envoy normally provides in
//! the gateway deployment: forwarding the request to the selected worker.
//!
//! It demonstrates that the routing business logic runs as a standalone router
//! behind an HTTP frontend, at feature parity with the gateway EPP, without any
//! Dynamo runtime: vanilla vLLM pods are discovered from the InferencePool
//! selector and routed to over plain HTTP.
//!
//! Enabled by `DYN_EPP_HTTP_FRONTEND=true`; `DYN_EPP_HTTP_HOST` (default
//! `0.0.0.0`) / `DYN_EPP_HTTP_PORT` (default 8000) set the bind address,
//! `DYN_MODEL_NAME` the advertised model id, `DYN_EPP_UPSTREAM_SCHEME` (default
//! `http`) the scheme used to reach workers.

use std::sync::Arc;

use anyhow::Result;
use axum::{
    Router as AxumRouter,
    body::Body,
    extract::{Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use futures::StreamExt;
use serde_json::json;

use crate::Router;
use crate::picker::{EndpointPicker, RequestInfo};
use crate::server::{extract_model_from_body, inject_token_data};

/// Cap on the request body we will buffer before routing. Routing needs the
/// whole body (it tokenizes the prompt), so the body is read fully rather than
/// streamed; this bounds memory per in-flight request.
const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;

struct FrontendState {
    picker: Arc<Router>,
    client: reqwest::Client,
    model_name: String,
    upstream_scheme: String,
}

pub async fn run(picker: Arc<Router>) -> Result<()> {
    let host = std::env::var("DYN_EPP_HTTP_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port: u16 = std::env::var("DYN_EPP_HTTP_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8000);
    let model_name = std::env::var("DYN_MODEL_NAME").unwrap_or_else(|_| "vllm".to_string());
    let upstream_scheme =
        std::env::var("DYN_EPP_UPSTREAM_SCHEME").unwrap_or_else(|_| "http".to_string());

    let state = Arc::new(FrontendState {
        picker,
        client: reqwest::Client::new(),
        model_name,
        upstream_scheme,
    });

    let app = AxumRouter::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/models", get(models_handler))
        .route("/v1/chat/completions", post(proxy_handler))
        .route("/v1/completions", post(proxy_handler))
        .with_state(state);

    let addr: std::net::SocketAddr = format!("{host}:{port}").parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(
        %addr,
        "Standalone HTTP frontend listening (no Dynamo runtime, no gateway): \
         POST /v1/chat/completions, /v1/completions"
    );
    axum::serve(listener, app).await?;
    Ok(())
}

/// OpenAI-style model listing so clients (and the OpenAI SDK) can discover the
/// served model id. Single entry — the InferencePool serves one model.
async fn models_handler(State(state): State<Arc<FrontendState>>) -> impl IntoResponse {
    axum::Json(json!({
        "object": "list",
        "data": [{
            "id": state.model_name,
            "object": "model",
            "owned_by": "dynamo",
        }],
    }))
}

/// Route + proxy. Selects a worker via the shared router (same decision the
/// gateway EPP makes), then forwards the request to that worker's pod and
/// streams the response straight back to the client.
async fn proxy_handler(State(state): State<Arc<FrontendState>>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    // Upstream path mirrors the inbound path (/v1/chat/completions or
    // /v1/completions) so the same handler serves both.
    let path = parts.uri.path().to_string();

    let raw_body = match axum::body::to_bytes(body, MAX_BODY_BYTES).await {
        Ok(b) => b.to_vec(),
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!("failed to read request body: {e}"),
            );
        }
    };

    let request_id = parts
        .headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    let headers: Vec<(String, String)> = parts
        .headers
        .iter()
        .filter_map(|(k, v)| {
            v.to_str()
                .ok()
                .map(|vs| (k.as_str().to_string(), vs.to_string()))
        })
        .collect();

    let req_info = RequestInfo {
        request_id: request_id.clone(),
        headers,
        body: raw_body.clone(),
        model: extract_model_from_body(&raw_body),
        // No Envoy subset metadata in the standalone path: route over the whole
        // ready InferencePool.
        candidate_subset: vec![],
    };

    // Same routing decision the gateway EPP makes. `pick` with an empty endpoint
    // list resolves candidates from the pod reflector (see `Router::pick`).
    let result = match state.picker.pick(&req_info, &[]).await {
        Ok(r) => r,
        Err(e) => {
            let status = match e {
                crate::picker::PickError::NoEndpoints => StatusCode::SERVICE_UNAVAILABLE,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            };
            return error_response(status, format!("routing failed: {e}"));
        }
    };

    let is_disaggregated = result
        .headers
        .iter()
        .any(|(k, v)| k == "x-dynamo-routing-mode" && v == "disaggregated");

    // Parity with the ext_proc path: inject nvext.token_data only when the
    // picker produced token IDs (dynamo-worker preprocessor mode). In external
    // vanilla-vLLM mode `token_ids` is None and the body is forwarded as-is, so
    // the worker tokenizes (no double-tokenization machinery here).
    let forwarded_body = match &result.token_ids {
        Some(ids) => inject_token_data(&raw_body, ids).unwrap_or(raw_body),
        None => raw_body,
    };

    let url = format!("{}://{}{}", state.upstream_scheme, result.endpoint, path);

    // Forward the picker's routing headers (worker/dp ids, routing mode, and —
    // for disagg — x-prefiller-host-port, which a decode-side P/D sidecar reads
    // to run vLLM's prefill->decode handshake).
    let mut upstream = state
        .client
        .post(&url)
        .header("content-type", "application/json")
        .body(forwarded_body);
    for (k, v) in &result.headers {
        upstream = upstream.header(k.as_str(), v.as_str());
    }

    tracing::info!(
        request_id = %request_id,
        endpoint = %result.endpoint,
        is_disaggregated,
        "Standalone frontend routed request; proxying to worker"
    );

    let resp = match upstream.send().await {
        Ok(r) => r,
        Err(e) => {
            // The pick is booked; release it so load accounting stays correct.
            state.picker.on_request_complete(&request_id).await;
            return error_response(
                StatusCode::BAD_GATEWAY,
                format!("upstream request to {} failed: {e}", result.endpoint),
            );
        }
    };

    // First bytes from the decode worker mean prefill is done (disagg): release
    // the prefill booking. Mirrors the ext_proc server's PostResponse hook.
    if is_disaggregated {
        state.picker.on_prefill_complete(&request_id).await;
    }

    let status = resp.status();
    let mut builder = Response::builder().status(status);
    // Preserve upstream response headers (content-type, SSE framing, etc.).
    for (k, v) in resp.headers() {
        builder = builder.header(k, v);
    }

    // Free the router booking when the response stream ends or the client
    // disconnects (whichever drops the stream first). `on_request_complete` is
    // async, so the guard hands it to a detached task on drop.
    let guard = CompletionGuard {
        picker: state.picker.clone(),
        request_id,
    };
    let stream = resp.bytes_stream().map(move |chunk| {
        // Keep the guard alive for the lifetime of the response stream.
        let _keep = &guard;
        chunk
    });

    builder
        .body(Body::from_stream(stream))
        .unwrap_or_else(|e| error_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

/// Releases a request's router booking exactly once, when the response stream
/// is fully consumed or dropped. `on_request_complete` is async, so the drop
/// detaches it onto the runtime.
struct CompletionGuard {
    picker: Arc<Router>,
    request_id: String,
}

impl Drop for CompletionGuard {
    fn drop(&mut self) {
        let picker = self.picker.clone();
        let request_id = std::mem::take(&mut self.request_id);
        tokio::spawn(async move {
            picker.on_request_complete(&request_id).await;
        });
    }
}

fn error_response(status: StatusCode, message: String) -> Response {
    tracing::warn!(%status, %message, "Standalone frontend returning error");
    (
        status,
        axum::Json(json!({ "error": { "message": message } })),
    )
        .into_response()
}
