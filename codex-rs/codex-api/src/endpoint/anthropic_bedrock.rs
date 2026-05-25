use crate::common::ResponseEvent;
use crate::common::ResponseStream;
use crate::common::ResponsesApiRequest;
use crate::endpoint::session::EndpointSession;
use crate::error::ApiError;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use codex_client::HttpTransport;
use codex_protocol::config_types::ReasoningSummary;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::ReasoningItemReasoningSummary;
use codex_protocol::models::ResponseItem;
use codex_protocol::models::WebSearchAction;
use codex_protocol::openai_models::ReasoningEffort;
use codex_protocol::protocol::TokenUsage;
use futures::StreamExt;
use http::HeaderMap;
use http::HeaderValue;
use http::Method;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use serde_json::json;
use std::collections::HashMap;
use tokio::sync::mpsc;

const ANTHROPIC_VERSION: &str = "bedrock-2023-05-31";
const MAX_OUTPUT_TOKENS: u32 = 128_000;
const ANTHROPIC_THINKING_METADATA_PREFIX: &str = "anthropic-bedrock-thinking:";

pub(crate) async fn stream_request<T: HttpTransport>(
    session: &EndpointSession<T>,
    request: ResponsesApiRequest,
    mut headers: HeaderMap,
) -> Result<ResponseStream, ApiError> {
    headers.insert(
        http::header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    headers.insert(
        http::header::ACCEPT,
        HeaderValue::from_static("application/vnd.amazon.eventstream"),
    );

    let body = anthropic_request_body(&request);
    let path = format!("model/{}/invoke-with-response-stream", request.model);
    let response = session
        .stream_with(Method::POST, &path, headers, Some(body), |_| {})
        .await?;

    Ok(response_stream_from_eventstream(
        response.bytes,
        request.model,
    ))
}

fn anthropic_request_body(request: &ResponsesApiRequest) -> Value {
    let mut body = json!({
        "anthropic_version": ANTHROPIC_VERSION,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "messages": anthropic_messages(&request.input),
        "tools": anthropic_tools(&request.tools),
        "tool_choice": {"type": "auto"},
    });
    if let Some(system) = anthropic_system(&request.instructions) {
        body["system"] = system;
    }
    if let Some(thinking) = anthropic_thinking(request) {
        body["thinking"] = thinking;
    }
    if let Some(effort) = anthropic_effort(request) {
        body["output_config"] = json!({ "effort": effort });
    }
    body
}

fn anthropic_system(instructions: &str) -> Option<Value> {
    if instructions.is_empty() {
        return None;
    }
    Some(json!([{
        "type": "text",
        "text": instructions,
        "cache_control": {"type": "ephemeral"},
    }]))
}

fn anthropic_thinking(request: &ResponsesApiRequest) -> Option<Value> {
    anthropic_effort(request)?;
    let mut thinking = json!({ "type": "adaptive" });
    if request
        .reasoning
        .as_ref()
        .and_then(|reasoning| reasoning.summary)
        .is_some_and(|summary| summary != ReasoningSummary::None)
    {
        thinking["display"] = json!("summarized");
    }
    Some(thinking)
}

fn anthropic_effort(request: &ResponsesApiRequest) -> Option<&'static str> {
    match request.reasoning.as_ref()?.effort? {
        ReasoningEffort::Low => Some("low"),
        ReasoningEffort::Medium => Some("medium"),
        ReasoningEffort::High => Some("high"),
        ReasoningEffort::None | ReasoningEffort::Minimal | ReasoningEffort::XHigh => None,
    }
}

fn anthropic_model_matches_requested(requested_model: &str, response_model: &str) -> bool {
    match (
        claude_model_family(requested_model),
        claude_model_family(response_model),
    ) {
        (Some(requested), Some(response)) => requested == response,
        (Some(_), None) => false,
        _ => true,
    }
}

fn claude_model_family(model: &str) -> Option<&'static str> {
    if model.contains("claude-opus-4-6") {
        Some("claude-opus-4-6")
    } else {
        None
    }
}

fn anthropic_messages(items: &[ResponseItem]) -> Vec<Value> {
    let mut builder = AnthropicMessagesBuilder::default();
    for item in items {
        match item {
            ResponseItem::Message { role, content, .. } => {
                let role = if role == "assistant" {
                    "assistant"
                } else {
                    "user"
                };
                builder.push_message(role, anthropic_content(content));
            }
            ResponseItem::FunctionCall {
                call_id,
                name,
                arguments,
                ..
            } => {
                let input =
                    serde_json::from_str(arguments).unwrap_or(Value::String(arguments.clone()));
                builder.push_tool_use(
                    call_id,
                    json!({"type": "tool_use", "id": call_id, "name": name, "input": input}),
                );
            }
            ResponseItem::FunctionCallOutput { call_id, output } => {
                builder.push_tool_result(call_id, tool_result_block(call_id, output));
            }
            ResponseItem::CustomToolCall {
                call_id,
                name,
                input,
                ..
            } => {
                let input = serde_json::from_str(input).unwrap_or(Value::String(input.clone()));
                builder.push_tool_use(
                    call_id,
                    json!({"type": "tool_use", "id": call_id, "name": name, "input": input}),
                );
            }
            ResponseItem::CustomToolCallOutput {
                call_id, output, ..
            } => {
                builder.push_tool_result(call_id, tool_result_block(call_id, output));
            }
            ResponseItem::Reasoning {
                encrypted_content, ..
            } => {
                if let Some(blocks) = anthropic_reasoning_blocks(encrypted_content.as_deref()) {
                    for block in blocks {
                        builder.push_assistant_content(block);
                    }
                }
            }
            _ => {}
        }
    }
    let mut messages = builder.finish();
    mark_last_message_for_prompt_cache(&mut messages);
    messages
}

#[derive(Default)]
struct AnthropicMessagesBuilder {
    messages: Vec<Value>,
    pending_tool_uses: Vec<PendingToolUse>,
    pending_tool_results: HashMap<String, Value>,
    deferred_messages: Vec<DeferredMessage>,
}

struct PendingToolUse {
    id: String,
    block: Value,
}

struct DeferredMessage {
    role: &'static str,
    content: Vec<Value>,
}

impl AnthropicMessagesBuilder {
    fn push_assistant_content(&mut self, block: Value) {
        self.push_message("assistant", vec![block]);
    }

    fn push_message(&mut self, role: &'static str, content: Vec<Value>) {
        if self.pending_tool_uses.is_empty() {
            push_message(&mut self.messages, role, content);
        } else {
            self.deferred_messages
                .push(DeferredMessage { role, content });
        }
    }

    fn push_tool_use(&mut self, call_id: &str, block: Value) {
        self.pending_tool_uses.push(PendingToolUse {
            id: call_id.to_string(),
            block,
        });
    }

    fn push_tool_result(&mut self, call_id: &str, block: Value) {
        if self
            .pending_tool_uses
            .iter()
            .any(|tool_use| tool_use.id == call_id)
        {
            self.pending_tool_results.insert(call_id.to_string(), block);
            if self
                .pending_tool_uses
                .iter()
                .all(|tool_use| self.pending_tool_results.contains_key(tool_use.id.as_str()))
            {
                self.flush_tool_exchange();
            }
        }
    }

    fn finish(mut self) -> Vec<Value> {
        if !self.pending_tool_uses.is_empty() {
            let missing_tool_results = self
                .pending_tool_uses
                .iter()
                .filter(|tool_use| !self.pending_tool_results.contains_key(tool_use.id.as_str()))
                .map(|tool_use| tool_use.id.clone())
                .collect::<Vec<_>>();
            for tool_use_id in missing_tool_results {
                self.pending_tool_results.insert(
                    tool_use_id.clone(),
                    interrupted_tool_result_block(&tool_use_id),
                );
            }
            self.flush_tool_exchange();
        }
        self.messages
    }

    fn flush_tool_exchange(&mut self) {
        if self.pending_tool_uses.is_empty() {
            return;
        }
        let pending_tool_uses = std::mem::take(&mut self.pending_tool_uses);
        let mut tool_use_blocks = Vec::new();
        let mut tool_result_blocks = Vec::new();
        for tool_use in pending_tool_uses {
            tool_result_blocks.push(
                self.pending_tool_results
                    .remove(tool_use.id.as_str())
                    .unwrap_or_else(|| interrupted_tool_result_block(&tool_use.id)),
            );
            tool_use_blocks.push(tool_use.block);
        }
        push_message(&mut self.messages, "assistant", tool_use_blocks);
        push_message(&mut self.messages, "user", tool_result_blocks);

        for DeferredMessage { role, content } in std::mem::take(&mut self.deferred_messages) {
            push_message(&mut self.messages, role, content);
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct AnthropicThinkingMetadata {
    blocks: Vec<AnthropicThinkingMetadataBlock>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicThinkingMetadataBlock {
    Thinking { thinking: String, signature: String },
    RedactedThinking { data: String },
}

fn anthropic_reasoning_blocks(encrypted_content: Option<&str>) -> Option<Vec<Value>> {
    let metadata = encrypted_content?
        .strip_prefix(ANTHROPIC_THINKING_METADATA_PREFIX)
        .and_then(|metadata| serde_json::from_str::<AnthropicThinkingMetadata>(metadata).ok())?;
    Some(
        metadata
            .blocks
            .into_iter()
            .map(|block| match block {
                AnthropicThinkingMetadataBlock::Thinking {
                    thinking,
                    signature,
                } => json!({
                    "type": "thinking",
                    "thinking": thinking,
                    "signature": signature,
                }),
                AnthropicThinkingMetadataBlock::RedactedThinking { data } => json!({
                    "type": "redacted_thinking",
                    "data": data,
                }),
            })
            .collect(),
    )
}

fn push_message(messages: &mut Vec<Value>, role: &str, content: Vec<Value>) {
    if content.is_empty() {
        return;
    }
    if let Some(last) = messages.last_mut()
        && last.get("role").and_then(Value::as_str) == Some(role)
        && let Some(existing_content) = last.get_mut("content").and_then(Value::as_array_mut)
    {
        existing_content.extend(content);
        return;
    }
    messages.push(json!({"role": role, "content": content}));
}

fn mark_last_message_for_prompt_cache(messages: &mut [Value]) {
    let Some(content) = messages
        .last_mut()
        .and_then(|message| message.get_mut("content"))
        .and_then(Value::as_array_mut)
    else {
        return;
    };
    let Some(last_block) = content.last_mut().and_then(Value::as_object_mut) else {
        return;
    };
    last_block.insert("cache_control".to_string(), json!({"type": "ephemeral"}));
}

fn anthropic_content(content: &[ContentItem]) -> Vec<Value> {
    let mut blocks = Vec::new();
    for item in content {
        match item {
            ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                blocks.push(json!({"type": "text", "text": text}));
            }
            ContentItem::InputImage { image_url, .. } => {
                if let Some((media_type, data)) = image_url
                    .strip_prefix("data:")
                    .and_then(|rest| rest.split_once(";base64,"))
                {
                    blocks.push(json!({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        }
                    }));
                }
            }
        }
    }
    if blocks.is_empty() {
        blocks.push(json!({"type": "text", "text": ""}));
    }
    blocks
}

fn tool_result_block(call_id: &str, output: &FunctionCallOutputPayload) -> Value {
    json!({
        "type": "tool_result",
        "tool_use_id": call_id,
        "content": function_output_text(&output.body),
        "is_error": output.success == Some(false),
    })
}

fn interrupted_tool_result_block(call_id: &str) -> Value {
    json!({
        "type": "tool_result",
        "tool_use_id": call_id,
        "content": "Tool execution was interrupted before a result was recorded.",
        "is_error": true,
    })
}

fn function_output_text(body: &FunctionCallOutputBody) -> String {
    body.to_text().unwrap_or_default()
}

fn anthropic_tools(tools: &[Value]) -> Vec<Value> {
    let mut converted: Vec<Value> = tools
        .iter()
        .flat_map(|tool| anthropic_tool(tool).unwrap_or_default())
        .collect();
    if let Some(last) = converted.last_mut()
        && let Some(last) = last.as_object_mut()
    {
        last.insert("cache_control".to_string(), json!({"type": "ephemeral"}));
    }
    converted
}

fn anthropic_tool(tool: &Value) -> Option<Vec<Value>> {
    match tool.get("type").and_then(Value::as_str) {
        Some("function") => anthropic_function_tool(tool).map(|tool| vec![tool]),
        Some("namespace") => {
            let namespace = tool.get("name").and_then(Value::as_str).unwrap_or("");
            let namespace_description = tool
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("");
            let tools = tool.get("tools")?.as_array()?;
            Some(
                tools
                    .iter()
                    .filter_map(|tool| {
                        let mut converted = anthropic_function_tool(tool)?;
                        let object = converted.as_object_mut()?;
                        if let Some(name) = object.get("name").and_then(Value::as_str) {
                            object.insert(
                                "name".to_string(),
                                Value::String(format!("{namespace}{name}")),
                            );
                        }
                        if let Some(description) = object.get("description").and_then(Value::as_str)
                        {
                            object.insert(
                                "description".to_string(),
                                Value::String(format!("{namespace_description}\n\n{description}")),
                            );
                        }
                        Some(converted)
                    })
                    .collect(),
            )
        }
        _ => None,
    }
}

fn anthropic_function_tool(tool: &Value) -> Option<Value> {
    Some(json!({
        "name": tool.get("name")?.clone(),
        "description": tool.get("description").cloned().unwrap_or(Value::String(String::new())),
        "input_schema": tool.get("parameters").cloned().unwrap_or_else(|| json!({"type": "object"})),
    }))
}

fn response_stream_from_eventstream(
    mut bytes: codex_client::ByteStream,
    requested_model: String,
) -> ResponseStream {
    let (tx_event, rx_event) = mpsc::channel::<Result<ResponseEvent, ApiError>>(1600);
    tokio::spawn(async move {
        let _ = tx_event.send(Ok(ResponseEvent::Created)).await;

        let mut buffer = Vec::new();
        let mut state = AnthropicStreamState {
            requested_model: Some(requested_model),
            ..Default::default()
        };
        while let Some(chunk) = bytes.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(err) => {
                    let _ = tx_event.send(Err(ApiError::Transport(err))).await;
                    return;
                }
            };
            buffer.extend_from_slice(&chunk);
            let messages = match take_eventstream_messages(&mut buffer) {
                Ok(messages) => messages,
                Err(err) => {
                    let _ = tx_event.send(Err(err)).await;
                    return;
                }
            };
            for message in messages {
                if let Some(error) = eventstream_error(&message) {
                    let _ = tx_event.send(Err(error)).await;
                    return;
                }
                let payload = decode_bedrock_payload(message.payload);
                let event: Value = match serde_json::from_slice(&payload) {
                    Ok(event) => event,
                    Err(err) => {
                        let _ = tx_event
                            .send(Err(ApiError::Stream(format!(
                                "failed to decode Bedrock Claude stream event: {err}"
                            ))))
                            .await;
                        return;
                    }
                };
                if process_anthropic_stream_event(&tx_event, &mut state, event)
                    .await
                    .is_err()
                {
                    return;
                }
                if state.completed {
                    return;
                }
            }
        }

        if !state.completed {
            let _ = tx_event
                .send(Err(ApiError::Stream(
                    "Bedrock Claude stream closed before message_stop".to_string(),
                )))
                .await;
        }
    });
    ResponseStream {
        rx_event,
        upstream_request_id: None,
    }
}

#[derive(Default)]
struct AnthropicStreamState {
    requested_model: Option<String>,
    response_id: Option<String>,
    usage: Option<TokenUsage>,
    text_open: bool,
    text_block_index: u32,
    output_text: String,
    tool_blocks: HashMap<i64, AnthropicToolBlock>,
    thinking_blocks: HashMap<i64, AnthropicThinkingBlock>,
    completed: bool,
}

struct AnthropicToolBlock {
    id: String,
    name: String,
    input_json: String,
    kind: AnthropicToolBlockKind,
}

enum AnthropicToolBlockKind {
    Function,
    WebSearch,
}

struct AnthropicThinkingBlock {
    kind: AnthropicThinkingBlockKind,
}

enum AnthropicThinkingBlockKind {
    Thinking { thinking: String, signature: String },
    RedactedThinking { data: String },
}

async fn process_anthropic_stream_event(
    tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>,
    state: &mut AnthropicStreamState,
    event: Value,
) -> Result<(), ()> {
    match event.get("type").and_then(Value::as_str) {
        Some("message_start") => {
            if let Some(message) = event.get("message") {
                state.response_id = message
                    .get("id")
                    .and_then(Value::as_str)
                    .map(ToString::to_string);
                if let Some(response_model) = message.get("model").and_then(Value::as_str)
                    && let Some(requested_model) = state.requested_model.as_deref()
                    && !anthropic_model_matches_requested(requested_model, response_model)
                {
                    tx_event
                        .send(Err(ApiError::Stream(format!(
                            "Bedrock Claude returned model `{response_model}` for requested model `{requested_model}`"
                        ))))
                        .await
                        .map_err(|_| ())?;
                    state.completed = true;
                    return Err(());
                }
                state.usage = message
                    .get("usage")
                    .and_then(|usage| serde_json::from_value::<AnthropicUsage>(usage.clone()).ok())
                    .map(Into::into);
            }
        }
        Some("content_block_start") => {
            let index = event.get("index").and_then(Value::as_i64).unwrap_or(0);
            let Some(block) = event.get("content_block") else {
                return Ok(());
            };
            let block_type = block.get("type").and_then(Value::as_str);
            if block_type == Some("thinking") {
                flush_streaming_output_text(tx_event, state).await?;
                state.thinking_blocks.insert(
                    index,
                    AnthropicThinkingBlock {
                        kind: AnthropicThinkingBlockKind::Thinking {
                            thinking: block
                                .get("thinking")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string(),
                            signature: block
                                .get("signature")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string(),
                        },
                    },
                );
            } else if block_type == Some("redacted_thinking") {
                flush_streaming_output_text(tx_event, state).await?;
                state.thinking_blocks.insert(
                    index,
                    AnthropicThinkingBlock {
                        kind: AnthropicThinkingBlockKind::RedactedThinking {
                            data: block
                                .get("data")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string(),
                        },
                    },
                );
            } else if block_type == Some("tool_use") || block_type == Some("server_tool_use") {
                flush_streaming_output_text(tx_event, state).await?;
                let name = block
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("tool")
                    .to_string();
                let kind = if block_type == Some("server_tool_use") && name == "web_search" {
                    AnthropicToolBlockKind::WebSearch
                } else {
                    AnthropicToolBlockKind::Function
                };
                state.tool_blocks.insert(
                    index,
                    AnthropicToolBlock {
                        id: block
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or("toolu_bedrock")
                            .to_string(),
                        name,
                        input_json: block
                            .get("input")
                            .map(Value::to_string)
                            .unwrap_or_else(|| "{}".to_string()),
                        kind,
                    },
                );
            }
        }
        Some("content_block_delta") => {
            let delta = event.get("delta").unwrap_or(&Value::Null);
            match delta.get("type").and_then(Value::as_str) {
                Some("text_delta") => {
                    if let Some(text) = delta.get("text").and_then(Value::as_str) {
                        send_streaming_output_text_delta(tx_event, state, text).await?;
                    }
                }
                Some("input_json_delta") => {
                    let index = event.get("index").and_then(Value::as_i64).unwrap_or(0);
                    if let Some(partial) = delta.get("partial_json").and_then(Value::as_str)
                        && let Some(block) = state.tool_blocks.get_mut(&index)
                    {
                        if block.input_json == "{}" {
                            block.input_json.clear();
                        }
                        block.input_json.push_str(partial);
                    }
                }
                Some("thinking_delta") => {
                    let index = event.get("index").and_then(Value::as_i64).unwrap_or(0);
                    if let Some(thinking) = delta.get("thinking").and_then(Value::as_str)
                        && let Some(block) = state.thinking_blocks.get_mut(&index)
                        && let AnthropicThinkingBlockKind::Thinking { thinking: text, .. } =
                            &mut block.kind
                    {
                        text.push_str(thinking);
                    }
                }
                Some("signature_delta") => {
                    let index = event.get("index").and_then(Value::as_i64).unwrap_or(0);
                    if let Some(signature_delta) = delta.get("signature").and_then(Value::as_str)
                        && let Some(block) = state.thinking_blocks.get_mut(&index)
                        && let AnthropicThinkingBlockKind::Thinking { signature, .. } =
                            &mut block.kind
                    {
                        signature.push_str(signature_delta);
                    }
                }
                _ => {}
            }
        }
        Some("content_block_stop") => {
            let index = event.get("index").and_then(Value::as_i64).unwrap_or(0);
            if let Some(block) = state.thinking_blocks.remove(&index) {
                tx_event
                    .send(Ok(ResponseEvent::OutputItemDone(anthropic_reasoning_item(
                        block,
                    ))))
                    .await
                    .map_err(|_| ())?;
            } else if let Some(block) = state.tool_blocks.remove(&index) {
                let arguments: Value = serde_json::from_str(&block.input_json)
                    .unwrap_or(Value::Object(Default::default()));
                let item = match block.kind {
                    AnthropicToolBlockKind::Function => ResponseItem::FunctionCall {
                        id: None,
                        name: block.name,
                        namespace: None,
                        arguments: arguments.to_string(),
                        call_id: block.id,
                    },
                    AnthropicToolBlockKind::WebSearch => ResponseItem::WebSearchCall {
                        id: Some(block.id),
                        status: Some("completed".to_string()),
                        action: Some(WebSearchAction::Search {
                            query: arguments
                                .get("query")
                                .and_then(Value::as_str)
                                .map(ToString::to_string),
                            queries: None,
                        }),
                    },
                };
                tx_event
                    .send(Ok(ResponseEvent::OutputItemDone(item)))
                    .await
                    .map_err(|_| ())?;
            }
        }
        Some("message_delta") => {
            if let Some(usage) = event
                .get("usage")
                .and_then(|usage| serde_json::from_value::<AnthropicUsageDelta>(usage.clone()).ok())
            {
                let existing = state.usage.get_or_insert_with(TokenUsage::default);
                if let Some(output_tokens) = usage.output_tokens {
                    existing.output_tokens = output_tokens;
                    existing.total_tokens = existing.input_tokens + existing.output_tokens;
                }
            }
        }
        Some("message_stop") => {
            flush_streaming_output_text(tx_event, state).await?;
            let response_id = state
                .response_id
                .clone()
                .unwrap_or_else(|| "bedrock-claude-response".to_string());
            tx_event
                .send(Ok(ResponseEvent::Completed {
                    response_id,
                    token_usage: state.usage.clone(),
                    end_turn: None,
                }))
                .await
                .map_err(|_| ())?;
            state.completed = true;
        }
        Some("ping") => {}
        Some(kind) if kind.ends_with("_error") => {
            let message = event
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("Bedrock Claude stream error");
            tx_event
                .send(Err(ApiError::Stream(message.to_string())))
                .await
                .map_err(|_| ())?;
            state.completed = true;
        }
        _ => {}
    }
    Ok(())
}

fn anthropic_reasoning_item(block: AnthropicThinkingBlock) -> ResponseItem {
    let metadata_block = match block.kind {
        AnthropicThinkingBlockKind::Thinking {
            thinking,
            signature,
        } => AnthropicThinkingMetadataBlock::Thinking {
            thinking,
            signature,
        },
        AnthropicThinkingBlockKind::RedactedThinking { data } => {
            AnthropicThinkingMetadataBlock::RedactedThinking { data }
        }
    };
    let summary = match &metadata_block {
        AnthropicThinkingMetadataBlock::Thinking { thinking, .. } if !thinking.is_empty() => {
            vec![ReasoningItemReasoningSummary::SummaryText {
                text: thinking.clone(),
            }]
        }
        _ => Vec::new(),
    };
    let metadata = AnthropicThinkingMetadata {
        blocks: vec![metadata_block],
    };
    let encrypted_content = serde_json::to_string(&metadata)
        .ok()
        .map(|metadata| format!("{ANTHROPIC_THINKING_METADATA_PREFIX}{metadata}"));
    ResponseItem::Reasoning {
        id: "bedrock-claude-thinking".to_string(),
        summary,
        content: None,
        encrypted_content,
    }
}

async fn send_streaming_output_text_delta(
    tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>,
    state: &mut AnthropicStreamState,
    text: &str,
) -> Result<(), ()> {
    if !state.text_open {
        state.text_open = true;
        let item = ResponseItem::Message {
            id: Some(format!("bedrock-claude-message-{}", state.text_block_index)),
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText {
                text: String::new(),
            }],
            phase: None,
        };
        tx_event
            .send(Ok(ResponseEvent::OutputItemAdded(item)))
            .await
            .map_err(|_| ())?;
    }
    state.output_text.push_str(text);
    tx_event
        .send(Ok(ResponseEvent::OutputTextDelta(text.to_string())))
        .await
        .map_err(|_| ())
}

async fn flush_streaming_output_text(
    tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>,
    state: &mut AnthropicStreamState,
) -> Result<(), ()> {
    if !state.text_open {
        return Ok(());
    }
    let item = ResponseItem::Message {
        id: Some(format!("bedrock-claude-message-{}", state.text_block_index)),
        role: "assistant".to_string(),
        content: vec![ContentItem::OutputText {
            text: std::mem::take(&mut state.output_text),
        }],
        phase: None,
    };
    state.text_open = false;
    state.text_block_index += 1;
    tx_event
        .send(Ok(ResponseEvent::OutputItemDone(item)))
        .await
        .map_err(|_| ())
}

struct AwsEventStreamMessage {
    headers: HashMap<String, String>,
    payload: Vec<u8>,
}

fn take_eventstream_messages(buffer: &mut Vec<u8>) -> Result<Vec<AwsEventStreamMessage>, ApiError> {
    let mut messages = Vec::new();
    loop {
        if buffer.len() < 12 {
            return Ok(messages);
        }
        let total_len = u32::from_be_bytes(buffer[0..4].try_into().unwrap()) as usize;
        let headers_len = u32::from_be_bytes(buffer[4..8].try_into().unwrap()) as usize;
        if total_len < 16 || headers_len > total_len.saturating_sub(16) {
            return Err(ApiError::Stream(
                "invalid AWS event stream frame length".to_string(),
            ));
        }
        if buffer.len() < total_len {
            return Ok(messages);
        }
        let headers_start = 12;
        let headers_end = headers_start + headers_len;
        let payload_end = total_len - 4;
        let headers = parse_eventstream_headers(&buffer[headers_start..headers_end])?;
        let payload = buffer[headers_end..payload_end].to_vec();
        buffer.drain(..total_len);
        messages.push(AwsEventStreamMessage { headers, payload });
    }
}

fn parse_eventstream_headers(bytes: &[u8]) -> Result<HashMap<String, String>, ApiError> {
    let mut headers = HashMap::new();
    let mut offset = 0;
    while offset < bytes.len() {
        let name_len = *bytes
            .get(offset)
            .ok_or_else(|| ApiError::Stream("truncated AWS event stream header name".to_string()))?
            as usize;
        offset += 1;
        let name_end = offset + name_len;
        let name = std::str::from_utf8(bytes.get(offset..name_end).ok_or_else(|| {
            ApiError::Stream("truncated AWS event stream header name".to_string())
        })?)
        .map_err(|err| ApiError::Stream(format!("invalid AWS event stream header name: {err}")))?
        .to_string();
        offset = name_end;
        let value_type = *bytes.get(offset).ok_or_else(|| {
            ApiError::Stream("truncated AWS event stream header type".to_string())
        })?;
        offset += 1;

        match value_type {
            0 | 1 => {
                headers.insert(name, (value_type == 0).to_string());
            }
            2 => offset += 1,
            3 => offset += 2,
            4 => offset += 4,
            5 | 8 => offset += 8,
            6 | 7 => {
                let len_end = offset + 2;
                let len = u16::from_be_bytes(
                    bytes
                        .get(offset..len_end)
                        .ok_or_else(|| {
                            ApiError::Stream(
                                "truncated AWS event stream header value length".to_string(),
                            )
                        })?
                        .try_into()
                        .unwrap(),
                ) as usize;
                offset = len_end;
                let value_end = offset + len;
                if value_type == 7 {
                    let value =
                        std::str::from_utf8(bytes.get(offset..value_end).ok_or_else(|| {
                            ApiError::Stream(
                                "truncated AWS event stream string header value".to_string(),
                            )
                        })?)
                        .map_err(|err| {
                            ApiError::Stream(format!(
                                "invalid AWS event stream string header value: {err}"
                            ))
                        })?
                        .to_string();
                    headers.insert(name, value);
                }
                offset = value_end;
            }
            9 => offset += 16,
            other => {
                return Err(ApiError::Stream(format!(
                    "unsupported AWS event stream header type {other}"
                )));
            }
        }
        if offset > bytes.len() {
            return Err(ApiError::Stream(
                "truncated AWS event stream header value".to_string(),
            ));
        }
    }
    Ok(headers)
}

fn eventstream_error(message: &AwsEventStreamMessage) -> Option<ApiError> {
    let message_type = message.headers.get(":message-type")?;
    if message_type != "exception" && message_type != "error" {
        return None;
    }
    let payload: Value = serde_json::from_slice(&message.payload).unwrap_or(Value::Null);
    let message = payload
        .get("message")
        .or_else(|| payload.get("Message"))
        .and_then(Value::as_str)
        .unwrap_or("Bedrock stream error");
    Some(ApiError::Stream(message.to_string()))
}

fn decode_bedrock_payload(payload: Vec<u8>) -> Vec<u8> {
    let Ok(value) = serde_json::from_slice::<Value>(&payload) else {
        return payload;
    };
    let Some(encoded) = value.get("bytes").and_then(Value::as_str) else {
        return payload;
    };
    BASE64_STANDARD.decode(encoded).unwrap_or(payload)
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    #[serde(default)]
    input_tokens: i64,
    #[serde(default)]
    output_tokens: i64,
    #[serde(
        default,
        alias = "cacheReadInputTokens",
        alias = "CacheReadInputTokens"
    )]
    cache_read_input_tokens: i64,
    #[serde(
        default,
        alias = "cacheCreationInputTokens",
        alias = "CacheCreationInputTokens",
        alias = "cache_write_input_tokens",
        alias = "cacheWriteInputTokens",
        alias = "CacheWriteInputTokens"
    )]
    cache_creation_input_tokens: i64,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsageDelta {
    output_tokens: Option<i64>,
}

impl From<AnthropicUsage> for TokenUsage {
    fn from(value: AnthropicUsage) -> Self {
        let input_tokens =
            value.input_tokens + value.cache_read_input_tokens + value.cache_creation_input_tokens;
        TokenUsage {
            input_tokens,
            cached_input_tokens: value.cache_read_input_tokens,
            output_tokens: value.output_tokens,
            reasoning_output_tokens: 0,
            total_tokens: input_tokens + value.output_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request_with_instructions_and_tools(
        instructions: impl Into<String>,
        tools: Vec<Value>,
    ) -> ResponsesApiRequest {
        ResponsesApiRequest {
            model: "global.anthropic.claude-opus-4-6-v1".to_string(),
            instructions: instructions.into(),
            input: vec![ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "hello".to_string(),
                }],
                phase: None,
            }],
            tools,
            tool_choice: "auto".to_string(),
            parallel_tool_calls: true,
            reasoning: None,
            store: false,
            stream: false,
            include: Vec::new(),
            service_tier: None,
            prompt_cache_key: None,
            text: None,
            client_metadata: None,
        }
    }

    #[test]
    fn anthropic_request_body_marks_system_prompt_for_cache() {
        let request = request_with_instructions_and_tools("You are Codex.", Vec::new());

        let body = anthropic_request_body(&request);

        assert_eq!(body["system"][0]["type"], "text");
        assert_eq!(body["system"][0]["text"], "You are Codex.");
        assert_eq!(
            body["system"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );
    }

    #[test]
    fn anthropic_request_body_marks_last_tool_for_cache() {
        let request = request_with_instructions_and_tools(
            "",
            vec![
                json!({
                    "type": "function",
                    "name": "first_tool",
                    "description": "First tool",
                    "parameters": {"type": "object"},
                }),
                json!({
                    "type": "function",
                    "name": "second_tool",
                    "description": "Second tool",
                    "parameters": {"type": "object"},
                }),
            ],
        );

        let body = anthropic_request_body(&request);

        let tools = body["tools"].as_array().expect("tools array");
        assert_eq!(tools.len(), 2);
        assert!(tools[0].get("cache_control").is_none());
        assert_eq!(tools[1]["cache_control"], json!({"type": "ephemeral"}));
    }

    #[test]
    fn anthropic_request_body_marks_final_message_block_for_cache() {
        let request = ResponsesApiRequest {
            model: "global.anthropic.claude-opus-4-6-v1".to_string(),
            instructions: String::new(),
            input: vec![
                ResponseItem::Message {
                    id: None,
                    role: "user".to_string(),
                    content: vec![ContentItem::InputText {
                        text: "large stable prompt".to_string(),
                    }],
                    phase: None,
                },
                ResponseItem::Message {
                    id: None,
                    role: "assistant".to_string(),
                    content: vec![ContentItem::OutputText {
                        text: "previous answer".to_string(),
                    }],
                    phase: None,
                },
                ResponseItem::Message {
                    id: None,
                    role: "user".to_string(),
                    content: vec![ContentItem::InputText {
                        text: "follow up".to_string(),
                    }],
                    phase: None,
                },
            ],
            tools: Vec::new(),
            tool_choice: "auto".to_string(),
            parallel_tool_calls: true,
            reasoning: None,
            store: false,
            stream: false,
            include: Vec::new(),
            service_tier: None,
            prompt_cache_key: None,
            text: None,
            client_metadata: None,
        };

        let body = anthropic_request_body(&request);

        assert_eq!(
            body["messages"][2]["content"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );
        assert!(
            body["messages"][0]["content"][0]
                .get("cache_control")
                .is_none()
        );
        assert!(
            body["messages"][1]["content"][0]
                .get("cache_control")
                .is_none()
        );
    }

    #[test]
    fn anthropic_namespace_tools_use_canonical_display_names() {
        let tools = anthropic_tools(&[json!({
            "type": "namespace",
            "name": "mcp__delegento__",
            "description": "Delegento tools",
            "tools": [{
                "type": "function",
                "name": "google_ads_request",
                "description": "Call Google Ads",
                "parameters": {"type": "object"},
            }],
        })]);

        assert_eq!(tools[0]["name"], "mcp__delegento__google_ads_request");
    }

    #[test]
    fn anthropic_messages_keeps_tool_result_immediately_after_tool_use() {
        let messages = anthropic_messages(&[
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "make a file".to_string(),
                }],
                phase: None,
            },
            ResponseItem::Message {
                id: None,
                role: "assistant".to_string(),
                content: vec![ContentItem::OutputText {
                    text: "I'll do that.".to_string(),
                }],
                phase: None,
            },
            ResponseItem::FunctionCall {
                id: None,
                name: "shell".to_string(),
                namespace: None,
                arguments: r#"{"cmd":"mkdir -p smoke-project"}"#.to_string(),
                call_id: "toolu_test".to_string(),
            },
            ResponseItem::FunctionCallOutput {
                call_id: "toolu_test".to_string(),
                output: FunctionCallOutputPayload::from_text("ok".to_string()),
            },
        ]);

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["content"][0]["type"], "text");
        assert_eq!(messages[1]["content"][1]["type"], "tool_use");
        assert_eq!(messages[2]["role"], "user");
        assert_eq!(messages[2]["content"][0]["type"], "tool_result");
        assert_eq!(messages[2]["content"][0]["tool_use_id"], "toolu_test");
    }

    #[test]
    fn anthropic_messages_defers_regular_messages_until_after_tool_result() {
        let messages = anthropic_messages(&[
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "make a file".to_string(),
                }],
                phase: None,
            },
            ResponseItem::FunctionCall {
                id: None,
                name: "shell".to_string(),
                namespace: None,
                arguments: r#"{"cmd":"mkdir -p smoke-project"}"#.to_string(),
                call_id: "toolu_deferred".to_string(),
            },
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "background context that arrived before the output".to_string(),
                }],
                phase: None,
            },
            ResponseItem::FunctionCallOutput {
                call_id: "toolu_deferred".to_string(),
                output: FunctionCallOutputPayload::from_text("ok".to_string()),
            },
        ]);

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["content"][0]["type"], "tool_use");
        assert_eq!(messages[2]["role"], "user");
        assert_eq!(messages[2]["content"][0]["type"], "tool_result");
        assert_eq!(messages[2]["content"][0]["tool_use_id"], "toolu_deferred");
        assert_eq!(messages[2]["content"][1]["type"], "text");
    }

    #[test]
    fn anthropic_messages_synthesizes_result_for_dangling_tool_use() {
        let messages = anthropic_messages(&[
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "run something".to_string(),
                }],
                phase: None,
            },
            ResponseItem::FunctionCall {
                id: None,
                name: "shell".to_string(),
                namespace: None,
                arguments: r#"{"cmd":"sleep 10"}"#.to_string(),
                call_id: "toolu_interrupted".to_string(),
            },
        ]);

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["content"][0]["type"], "tool_use");
        assert_eq!(messages[2]["role"], "user");
        assert_eq!(messages[2]["content"][0]["type"], "tool_result");
        assert_eq!(
            messages[2]["content"][0]["tool_use_id"],
            "toolu_interrupted"
        );
        assert_eq!(messages[2]["content"][0]["is_error"], true);
    }

    #[test]
    fn anthropic_messages_orders_parallel_tool_results_by_tool_use() {
        let messages = anthropic_messages(&[
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "run two things".to_string(),
                }],
                phase: None,
            },
            ResponseItem::FunctionCall {
                id: None,
                name: "shell".to_string(),
                namespace: None,
                arguments: r#"{"cmd":"echo one"}"#.to_string(),
                call_id: "toolu_one".to_string(),
            },
            ResponseItem::FunctionCall {
                id: None,
                name: "shell".to_string(),
                namespace: None,
                arguments: r#"{"cmd":"echo two"}"#.to_string(),
                call_id: "toolu_two".to_string(),
            },
            ResponseItem::FunctionCallOutput {
                call_id: "toolu_two".to_string(),
                output: FunctionCallOutputPayload::from_text("two".to_string()),
            },
            ResponseItem::FunctionCallOutput {
                call_id: "toolu_one".to_string(),
                output: FunctionCallOutputPayload::from_text("one".to_string()),
            },
        ]);

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1]["content"][0]["id"], "toolu_one");
        assert_eq!(messages[1]["content"][1]["id"], "toolu_two");
        assert_eq!(messages[2]["content"][0]["tool_use_id"], "toolu_one");
        assert_eq!(messages[2]["content"][1]["tool_use_id"], "toolu_two");
    }

    #[test]
    fn anthropic_messages_round_trips_thinking_blocks_before_tool_use() {
        let reasoning_item = anthropic_reasoning_item(AnthropicThinkingBlock {
            kind: AnthropicThinkingBlockKind::Thinking {
                thinking: "Checking the plan.".to_string(),
                signature: "sig_test".to_string(),
            },
        });

        let messages = anthropic_messages(&[
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "make a plan".to_string(),
                }],
                phase: None,
            },
            reasoning_item,
            ResponseItem::FunctionCall {
                id: None,
                name: "shell".to_string(),
                namespace: None,
                arguments: r#"{"cmd":"pwd"}"#.to_string(),
                call_id: "toolu_thinking".to_string(),
            },
            ResponseItem::FunctionCallOutput {
                call_id: "toolu_thinking".to_string(),
                output: FunctionCallOutputPayload::from_text("ok".to_string()),
            },
        ]);

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["content"][0]["type"], "thinking");
        assert_eq!(messages[1]["content"][0]["thinking"], "Checking the plan.");
        assert_eq!(messages[1]["content"][0]["signature"], "sig_test");
        assert_eq!(messages[1]["content"][1]["type"], "tool_use");
        assert_eq!(messages[2]["content"][0]["type"], "tool_result");
    }

    #[test]
    fn anthropic_request_body_maps_reasoning_effort_to_adaptive_thinking() {
        let request = ResponsesApiRequest {
            model: "global.anthropic.claude-opus-4-6-v1".to_string(),
            instructions: String::new(),
            input: vec![ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "think".to_string(),
                }],
                phase: None,
            }],
            tools: Vec::new(),
            tool_choice: "auto".to_string(),
            parallel_tool_calls: true,
            reasoning: Some(crate::common::Reasoning {
                effort: Some(ReasoningEffort::High),
                summary: None,
            }),
            store: false,
            stream: false,
            include: Vec::new(),
            service_tier: None,
            prompt_cache_key: None,
            text: None,
            client_metadata: None,
        };

        let body = anthropic_request_body(&request);

        assert_eq!(body["thinking"], json!({ "type": "adaptive" }));
        assert_eq!(body["output_config"], json!({ "effort": "high" }));
    }

    #[test]
    fn anthropic_model_match_rejects_unrelated_claude_substitution() {
        assert!(anthropic_model_matches_requested(
            "global.anthropic.claude-opus-4-6-v1",
            "anthropic.claude-opus-4-6-v1"
        ));
        assert!(!anthropic_model_matches_requested(
            "global.anthropic.claude-opus-4-6-v1",
            "anthropic.claude-sonnet-4-5-v1"
        ));
    }

    #[test]
    fn anthropic_usage_maps_prompt_cache_tokens() {
        let usage: TokenUsage = serde_json::from_value::<AnthropicUsage>(json!({
            "input_tokens": 50,
            "cache_read_input_tokens": 1000,
            "cache_creation_input_tokens": 2000,
            "output_tokens": 25,
        }))
        .unwrap()
        .into();

        assert_eq!(usage.input_tokens, 3050);
        assert_eq!(usage.cached_input_tokens, 1000);
        assert_eq!(usage.output_tokens, 25);
        assert_eq!(usage.total_tokens, 3075);
    }

    #[tokio::test]
    async fn anthropic_stream_errors_when_response_model_differs() {
        let (tx, mut rx) = mpsc::channel::<Result<ResponseEvent, ApiError>>(8);
        let mut state = AnthropicStreamState {
            requested_model: Some("global.anthropic.claude-opus-4-6-v1".to_string()),
            ..Default::default()
        };

        let result = process_anthropic_stream_event(
            &tx,
            &mut state,
            json!({
                "type": "message_start",
                "message": {
                    "id": "msg_bedrock",
                    "model": "anthropic.claude-sonnet-4-5-v1",
                    "usage": {"input_tokens": 1, "output_tokens": 0}
                }
            }),
        )
        .await;

        assert!(result.is_err());
        assert!(state.completed);
        let event = rx.recv().await.expect("event");
        let Err(ApiError::Stream(message)) = event else {
            panic!("expected stream error");
        };
        assert!(message.contains("claude-sonnet-4-5-v1"));
        assert!(message.contains("claude-opus-4-6-v1"));
    }

    #[tokio::test]
    async fn anthropic_stream_events_preserve_thinking_signature() {
        let (tx, mut rx) = mpsc::channel::<Result<ResponseEvent, ApiError>>(8);
        let mut state = AnthropicStreamState::default();

        process_anthropic_stream_event(
            &tx,
            &mut state,
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""}
            }),
        )
        .await
        .unwrap();
        process_anthropic_stream_event(
            &tx,
            &mut state,
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Checking docs."}
            }),
        )
        .await
        .unwrap();
        process_anthropic_stream_event(
            &tx,
            &mut state,
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "sig_123"}
            }),
        )
        .await
        .unwrap();
        process_anthropic_stream_event(
            &tx,
            &mut state,
            json!({"type": "content_block_stop", "index": 0}),
        )
        .await
        .unwrap();

        let event = rx.recv().await.expect("event").expect("ok");
        let ResponseEvent::OutputItemDone(ResponseItem::Reasoning {
            summary,
            encrypted_content,
            ..
        }) = event
        else {
            panic!("expected reasoning item");
        };
        assert_eq!(
            summary,
            vec![ReasoningItemReasoningSummary::SummaryText {
                text: "Checking docs.".to_string()
            }]
        );
        let blocks =
            anthropic_reasoning_blocks(encrypted_content.as_deref()).expect("thinking blocks");
        assert_eq!(blocks[0]["type"], "thinking");
        assert_eq!(blocks[0]["thinking"], "Checking docs.");
        assert_eq!(blocks[0]["signature"], "sig_123");
    }

    #[tokio::test]
    async fn anthropic_stream_events_emit_text_deltas_before_completion() {
        let (tx, mut rx) = mpsc::channel::<Result<ResponseEvent, ApiError>>(8);
        let mut state = AnthropicStreamState::default();

        process_anthropic_stream_event(
            &tx,
            &mut state,
            json!({
                "type": "message_start",
                "message": {
                    "id": "msg_bedrock",
                    "usage": {"input_tokens": 11, "output_tokens": 0}
                }
            }),
        )
        .await
        .unwrap();
        process_anthropic_stream_event(
            &tx,
            &mut state,
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello "}
            }),
        )
        .await
        .unwrap();
        process_anthropic_stream_event(
            &tx,
            &mut state,
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "world"}
            }),
        )
        .await
        .unwrap();
        process_anthropic_stream_event(
            &tx,
            &mut state,
            json!({"type": "message_delta", "usage": {"output_tokens": 2}}),
        )
        .await
        .unwrap();
        process_anthropic_stream_event(&tx, &mut state, json!({"type": "message_stop"}))
            .await
            .unwrap();

        match rx.recv().await.unwrap().unwrap() {
            ResponseEvent::OutputItemAdded(ResponseItem::Message { id, .. }) => {
                assert_eq!(id.as_deref(), Some("bedrock-claude-message-0"));
            }
            event => panic!("expected OutputItemAdded, got {event:?}"),
        }
        assert!(matches!(
            rx.recv().await.unwrap().unwrap(),
            ResponseEvent::OutputTextDelta(text) if text == "hello "
        ));
        assert!(matches!(
            rx.recv().await.unwrap().unwrap(),
            ResponseEvent::OutputTextDelta(text) if text == "world"
        ));
        match rx.recv().await.unwrap().unwrap() {
            ResponseEvent::OutputItemDone(ResponseItem::Message { content, .. }) => {
                assert_eq!(
                    content,
                    vec![ContentItem::OutputText {
                        text: "hello world".to_string()
                    }]
                );
            }
            event => panic!("expected OutputItemDone message, got {event:?}"),
        }
        match rx.recv().await.unwrap().unwrap() {
            ResponseEvent::Completed {
                response_id,
                token_usage,
                ..
            } => {
                assert_eq!(response_id, "msg_bedrock");
                let token_usage = token_usage.unwrap();
                assert_eq!(token_usage.input_tokens, 11);
                assert_eq!(token_usage.output_tokens, 2);
                assert_eq!(token_usage.total_tokens, 13);
            }
            event => panic!("expected Completed, got {event:?}"),
        }
    }

    #[tokio::test]
    async fn anthropic_stream_events_accumulate_tool_arguments() {
        let (tx, mut rx) = mpsc::channel::<Result<ResponseEvent, ApiError>>(8);
        let mut state = AnthropicStreamState::default();

        process_anthropic_stream_event(
            &tx,
            &mut state,
            json!({
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "shell",
                    "input": {}
                }
            }),
        )
        .await
        .unwrap();
        process_anthropic_stream_event(
            &tx,
            &mut state,
            json!({
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": "{\"cmd\":\""}
            }),
        )
        .await
        .unwrap();
        process_anthropic_stream_event(
            &tx,
            &mut state,
            json!({
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": "echo ok\"}"}
            }),
        )
        .await
        .unwrap();
        process_anthropic_stream_event(
            &tx,
            &mut state,
            json!({"type": "content_block_stop", "index": 1}),
        )
        .await
        .unwrap();

        match rx.recv().await.unwrap().unwrap() {
            ResponseEvent::OutputItemDone(ResponseItem::FunctionCall {
                name,
                arguments,
                call_id,
                ..
            }) => {
                assert_eq!(name, "shell");
                assert_eq!(call_id, "toolu_1");
                assert_eq!(arguments, json!({"cmd": "echo ok"}).to_string());
            }
            event => panic!("expected FunctionCall, got {event:?}"),
        }
    }

    #[tokio::test]
    async fn anthropic_stream_events_map_server_web_search() {
        let (tx, mut rx) = mpsc::channel::<Result<ResponseEvent, ApiError>>(8);
        let mut state = AnthropicStreamState::default();

        process_anthropic_stream_event(
            &tx,
            &mut state,
            json!({
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "server_tool_use",
                    "id": "srvtoolu_1",
                    "name": "web_search",
                    "input": {}
                }
            }),
        )
        .await
        .unwrap();
        process_anthropic_stream_event(
            &tx,
            &mut state,
            json!({
                "type": "content_block_delta",
                "index": 1,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": "{\"query\":\"latest codex cli release\"}"
                }
            }),
        )
        .await
        .unwrap();
        process_anthropic_stream_event(
            &tx,
            &mut state,
            json!({"type": "content_block_stop", "index": 1}),
        )
        .await
        .unwrap();

        match rx.recv().await.unwrap().unwrap() {
            ResponseEvent::OutputItemDone(ResponseItem::WebSearchCall { id, status, action }) => {
                assert_eq!(id.as_deref(), Some("srvtoolu_1"));
                assert_eq!(status.as_deref(), Some("completed"));
                assert_eq!(
                    action,
                    Some(WebSearchAction::Search {
                        query: Some("latest codex cli release".to_string()),
                        queries: None,
                    })
                );
            }
            event => panic!("expected WebSearchCall, got {event:?}"),
        }
    }

    #[test]
    fn aws_eventstream_parser_waits_for_complete_frames() {
        let mut frame = aws_eventstream_frame(
            &[(":message-type", "event"), (":event-type", "chunk")],
            br#"{"type":"ping"}"#,
        );
        let rest = frame.split_off(10);
        let mut buffer = frame;

        assert!(take_eventstream_messages(&mut buffer).unwrap().is_empty());
        buffer.extend(rest);
        let messages = take_eventstream_messages(&mut buffer).unwrap();

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].headers[":message-type"], "event");
        assert_eq!(messages[0].headers[":event-type"], "chunk");
        assert_eq!(messages[0].payload, br#"{"type":"ping"}"#);
        assert!(buffer.is_empty());
    }

    fn aws_eventstream_frame(headers: &[(&str, &str)], payload: &[u8]) -> Vec<u8> {
        let mut header_bytes = Vec::new();
        for (name, value) in headers {
            header_bytes.push(name.len() as u8);
            header_bytes.extend_from_slice(name.as_bytes());
            header_bytes.push(7);
            header_bytes.extend_from_slice(&(value.len() as u16).to_be_bytes());
            header_bytes.extend_from_slice(value.as_bytes());
        }
        let total_len = 16 + header_bytes.len() + payload.len();
        let mut frame = Vec::new();
        frame.extend_from_slice(&(total_len as u32).to_be_bytes());
        frame.extend_from_slice(&(header_bytes.len() as u32).to_be_bytes());
        frame.extend_from_slice(&0_u32.to_be_bytes());
        frame.extend_from_slice(&header_bytes);
        frame.extend_from_slice(payload);
        frame.extend_from_slice(&0_u32.to_be_bytes());
        frame
    }
}
