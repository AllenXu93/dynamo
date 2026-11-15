// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers_v2::Tool;
use dynamo_parsers_v2::unified::{
    InvalidGuidedPayloadPolicy, UnifiedParser, UnifiedParserEvent, UnifiedParserExt,
    UnifiedParserInit, UnifiedParserStartingState, UnifiedToolOutputMode,
    create_unified_parser_for_family,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{Value, json};

/// One request owns one parser for reasoning, visible text and tool arguments.
#[pyclass(unsendable)]
pub struct UnifiedChatParser {
    parser: Box<dyn UnifiedParser>,
    finished: bool,
}

#[pymethods]
impl UnifiedChatParser {
    #[new]
    #[pyo3(signature = (tools_json, starting_state="None", guided=false, named_tool=None, constrained=false))]
    fn new(
        tools_json: &str,
        starting_state: &str,
        guided: bool,
        named_tool: Option<String>,
        constrained: bool,
    ) -> PyResult<Self> {
        let tools: Vec<Tool> =
            serde_json::from_str(tools_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let starting_state = match starting_state {
            "None" => UnifiedParserStartingState::None,
            "Reasoning" => UnifiedParserStartingState::Reasoning,
            "Response" => UnifiedParserStartingState::Response,
            _ => return Err(PyValueError::new_err("invalid parser starting state")),
        };
        let mut parser = create_unified_parser_for_family("glm47", &tools)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        parser
            .initialize_request(UnifiedParserInit {
                starting_state,
                tool_output_mode: if guided {
                    UnifiedToolOutputMode::GuidedJson { named_tool }
                } else {
                    UnifiedToolOutputMode::Native
                },
                invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                ..Default::default()
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if constrained {
            return Err(PyValueError::new_err(
                "constrained native GLM argument framing is not supported by this bridge",
            ));
        }
        Ok(Self {
            parser,
            finished: false,
        })
    }

    /// Return ordered deltas; arguments remain raw JSON strings across the FFI.
    fn advance(&mut self, text: &str, finish: bool) -> PyResult<String> {
        if self.finished {
            if text.is_empty() {
                return Ok("[]".into());
            }
            return Err(PyValueError::new_err("parser already finished"));
        }
        let mut events = self
            .parser
            .push(text)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if finish {
            events.extend(
                self.parser
                    .finish()
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
                    .events,
            );
            self.finished = true;
        }
        let events: Vec<Value> = events
            .into_iter()
            .map(|event| {
                Ok(match event {
                    UnifiedParserEvent::Text(text) => json!({"kind": "text", "text": text}),
                    UnifiedParserEvent::Reasoning(text) => {
                        json!({"kind": "reasoning", "text": text})
                    }
                    UnifiedParserEvent::ToolCall(call) => json!({
                        "kind": "tool_call", "index": call.tool_index,
                        "name": call.name, "arguments": call.arguments, "complete": call.complete,
                    }),
                    UnifiedParserEvent::MessageStart(_) | UnifiedParserEvent::MessageEnd => {
                        return Err(PyValueError::new_err("unexpected GLM message-phase event"));
                    }
                })
            })
            .collect::<PyResult<_>>()?;
        serde_json::to_string(&events).map_err(|e| PyValueError::new_err(e.to_string()))
    }
}
