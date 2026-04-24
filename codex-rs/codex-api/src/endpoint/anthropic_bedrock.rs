use crate::common::ResponseEvent;
use crate::common::ResponseStream;
use crate::common::ResponsesApiRequest;
use crate::endpoint::session::EndpointSession;
use crate::error::ApiError;
use codex_client::HttpTransport;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ReasoningEffort;
use codex_protocol::protocol::TokenUsage;
use http::HeaderMap;
use http::HeaderValue;
use http::Method;
use serde::Deserialize;
use serde_json::Value;
use serde_json::json;
use tokio::sync::mpsc;

const ANTHROPIC_VERSION: &str = "bedrock-2023-05-31";
const MAX_OUTPUT_TOKENS: u32 = 128_000;

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
        HeaderValue::from_static("application/json"),
    );

    let body = anthropic_request_body(&request);
    let path = format!("model/{}/invoke", request.model);
    let response = session
        .execute(Method::POST, &path, headers, Some(body))
        .await?;
    let message: AnthropicMessage = serde_json::from_slice(&response.body)
        .map_err(|e| ApiError::Stream(format!("failed to decode Bedrock Claude response: {e}")))?;

    Ok(response_stream_from_message(message))
}

fn anthropic_request_body(request: &ResponsesApiRequest) -> Value {
    let mut body = json!({
        "anthropic_version": ANTHROPIC_VERSION,
        "system": request.instructions,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "messages": anthropic_messages(&request.input),
        "tools": anthropic_tools(&request.tools),
        "tool_choice": {"type": "auto"},
    });
    if let Some(thinking) = anthropic_thinking(request) {
        body["thinking"] = thinking;
    }
    body
}

fn anthropic_thinking(request: &ResponsesApiRequest) -> Option<Value> {
    let effort = request.reasoning.as_ref()?.effort?;
    let effort = match effort {
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::None | ReasoningEffort::Minimal | ReasoningEffort::XHigh => return None,
    };
    Some(json!({
        "type": "adaptive",
        "effort": effort,
        "display": "summarized",
    }))
}

fn anthropic_messages(items: &[ResponseItem]) -> Vec<Value> {
    let mut messages = Vec::new();
    let mut pending_tool_uses = Vec::new();
    let mut pending_tool_results = Vec::new();
    for item in items {
        match item {
            ResponseItem::Message { role, content, .. } => {
                flush_pending_tool_exchange(
                    &mut messages,
                    &mut pending_tool_uses,
                    &mut pending_tool_results,
                );
                let role = if role == "assistant" {
                    "assistant"
                } else {
                    "user"
                };
                push_message(&mut messages, role, anthropic_content(content));
            }
            ResponseItem::FunctionCall {
                call_id,
                name,
                arguments,
                ..
            } => {
                let input =
                    serde_json::from_str(arguments).unwrap_or(Value::String(arguments.clone()));
                pending_tool_uses
                    .push(json!({"type": "tool_use", "id": call_id, "name": name, "input": input}));
            }
            ResponseItem::FunctionCallOutput { call_id, output } => {
                pending_tool_results.push(tool_result_block(call_id, output));
            }
            ResponseItem::CustomToolCall {
                call_id,
                name,
                input,
                ..
            } => {
                let input = serde_json::from_str(input).unwrap_or(Value::String(input.clone()));
                pending_tool_uses
                    .push(json!({"type": "tool_use", "id": call_id, "name": name, "input": input}));
            }
            ResponseItem::CustomToolCallOutput {
                call_id, output, ..
            } => {
                pending_tool_results.push(tool_result_block(call_id, output));
            }
            _ => {}
        }
    }
    flush_pending_tool_exchange(
        &mut messages,
        &mut pending_tool_uses,
        &mut pending_tool_results,
    );
    messages
}

fn flush_pending_tool_exchange(
    messages: &mut Vec<Value>,
    pending_tool_uses: &mut Vec<Value>,
    pending_tool_results: &mut Vec<Value>,
) {
    if pending_tool_uses.is_empty() && pending_tool_results.is_empty() {
        return;
    }

    if !pending_tool_uses.is_empty() {
        push_message(messages, "assistant", std::mem::take(pending_tool_uses));
    }
    if !pending_tool_results.is_empty() {
        push_message(messages, "user", std::mem::take(pending_tool_results));
    }
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

fn function_output_text(body: &FunctionCallOutputBody) -> String {
    body.to_text().unwrap_or_default()
}

fn anthropic_tools(tools: &[Value]) -> Vec<Value> {
    tools
        .iter()
        .flat_map(|tool| anthropic_tool(tool).unwrap_or_default())
        .collect()
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
                                Value::String(format!("{namespace}__{name}")),
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

fn response_stream_from_message(message: AnthropicMessage) -> ResponseStream {
    let (tx_event, rx_event) = mpsc::channel::<Result<ResponseEvent, ApiError>>(32);
    tokio::spawn(async move {
        let _ = tx_event.send(Ok(ResponseEvent::Created)).await;
        let mut output_text = String::new();
        for block in message.content {
            match block {
                AnthropicContentBlock::Text { text } => {
                    output_text.push_str(&text);
                }
                AnthropicContentBlock::ToolUse { id, name, input } => {
                    flush_output_text(&tx_event, &mut output_text).await;
                    let item = ResponseItem::FunctionCall {
                        id: None,
                        name,
                        namespace: None,
                        arguments: input.to_string(),
                        call_id: id,
                    };
                    let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
                }
                AnthropicContentBlock::Other => {}
            }
        }
        flush_output_text(&tx_event, &mut output_text).await;
        let _ = tx_event
            .send(Ok(ResponseEvent::Completed {
                response_id: message.id,
                token_usage: message.usage.map(Into::into),
            }))
            .await;
    });
    ResponseStream { rx_event }
}

async fn flush_output_text(
    tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>,
    output_text: &mut String,
) {
    if output_text.is_empty() {
        return;
    }
    let item = ResponseItem::Message {
        id: None,
        role: "assistant".to_string(),
        content: vec![ContentItem::OutputText {
            text: std::mem::take(output_text),
        }],
        end_turn: None,
        phase: None,
    };
    let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
}

#[derive(Debug, Deserialize)]
struct AnthropicMessage {
    id: String,
    content: Vec<AnthropicContentBlock>,
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: i64,
    output_tokens: i64,
}

impl From<AnthropicUsage> for TokenUsage {
    fn from(value: AnthropicUsage) -> Self {
        TokenUsage {
            input_tokens: value.input_tokens,
            cached_input_tokens: 0,
            output_tokens: value.output_tokens,
            reasoning_output_tokens: 0,
            total_tokens: value.input_tokens + value.output_tokens,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(other)]
    Other,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_messages_keeps_tool_result_immediately_after_tool_use() {
        let messages = anthropic_messages(&[
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "make a file".to_string(),
                }],
                end_turn: None,
                phase: None,
            },
            ResponseItem::Message {
                id: None,
                role: "assistant".to_string(),
                content: vec![ContentItem::OutputText {
                    text: "I'll do that.".to_string(),
                }],
                end_turn: None,
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
                end_turn: None,
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

        assert_eq!(
            body["thinking"],
            json!({
                "type": "adaptive",
                "effort": "high",
                "display": "summarized",
            })
        );
    }
}
