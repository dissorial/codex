use crate::common::Reasoning;
use crate::common::ResponseEvent;
use crate::common::ResponseStream;
use crate::common::ResponsesApiRequest;
use crate::endpoint::session::EndpointSession;
use crate::error::ApiError;
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
use eventsource_stream::Eventsource;
use futures::StreamExt;
use http::HeaderMap;
use http::HeaderValue;
use http::Method;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Map;
use serde_json::Value;
use serde_json::json;
use std::collections::HashMap;
use tokio::sync::mpsc;

const MAX_OUTPUT_TOKENS: u32 = 65_536;
const SIGNATURE_METADATA_PREFIX: &str = "google_gemini_function_call_signatures:";

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
        HeaderValue::from_static("text/event-stream"),
    );

    let body = gemini_request_body(&request);
    let path = format!("models/{}:streamGenerateContent?alt=sse", request.model);
    let response = session
        .stream_with(Method::POST, &path, headers, Some(body), |_| {})
        .await?;

    Ok(response_stream_from_sse(response.bytes))
}

fn gemini_request_body(request: &ResponsesApiRequest) -> Value {
    let uses_web_search = request.tools.iter().any(is_web_search_tool);
    let mut generation_config = Map::new();
    generation_config.insert("maxOutputTokens".to_string(), json!(MAX_OUTPUT_TOKENS));
    if let Some(thinking_config) = gemini_thinking_config(request.reasoning.as_ref()) {
        generation_config.insert("thinkingConfig".to_string(), thinking_config);
    }
    json!({
        "systemInstruction": {
            "parts": gemini_text_parts(&request.instructions),
        },
        "contents": gemini_contents(&request.input),
        "tools": gemini_tools(&request.tools),
        "toolConfig": {
            "includeServerSideToolInvocations": uses_web_search,
            "functionCallingConfig": {
                "mode": if uses_web_search { "VALIDATED" } else { "AUTO" },
            }
        },
        "generationConfig": Value::Object(generation_config),
    })
}

fn gemini_text_parts(text: &str) -> Vec<Value> {
    if text.is_empty() {
        Vec::new()
    } else {
        vec![json!({"text": text})]
    }
}

fn gemini_thinking_config(reasoning: Option<&Reasoning>) -> Option<Value> {
    let reasoning = reasoning?;
    let mut thinking_config = Map::new();
    if let Some(thinking_level) = gemini_thinking_level(reasoning.effort) {
        thinking_config.insert(
            "thinkingLevel".to_string(),
            Value::String(thinking_level.to_string()),
        );
    }
    if reasoning
        .summary
        .is_some_and(|summary| summary != ReasoningSummary::None)
    {
        thinking_config.insert("includeThoughts".to_string(), Value::Bool(true));
    }
    if thinking_config.is_empty() {
        None
    } else {
        Some(Value::Object(thinking_config))
    }
}

fn gemini_thinking_level(effort: Option<ReasoningEffort>) -> Option<&'static str> {
    match effort {
        Some(ReasoningEffort::Low) => Some("low"),
        Some(ReasoningEffort::Medium) => Some("medium"),
        Some(ReasoningEffort::High | ReasoningEffort::XHigh) => Some("high"),
        Some(ReasoningEffort::None | ReasoningEffort::Minimal) | None => None,
    }
}

fn gemini_contents(items: &[ResponseItem]) -> Vec<Value> {
    let mut signature_metadata = gemini_signature_metadata(items);
    let call_names = gemini_function_call_names(items);
    let mut contents = Vec::new();
    let mut pending_model_parts = Vec::new();
    let mut pending_function_response_parts = Vec::new();

    for item in items {
        match item {
            ResponseItem::Message { role, content, .. } => {
                flush_tool_exchange(
                    &mut contents,
                    &mut pending_model_parts,
                    &mut pending_function_response_parts,
                );
                let role = if role == "assistant" { "model" } else { "user" };
                let mut parts = gemini_message_parts(
                    content,
                    role == "model",
                    &mut signature_metadata.text_part_signatures,
                );
                if role == "model" {
                    prepend_server_tool_parts_for_message(
                        content,
                        &mut signature_metadata,
                        &mut parts,
                    );
                }
                push_content(&mut contents, role, parts);
            }
            ResponseItem::FunctionCall {
                call_id,
                name,
                arguments,
                ..
            } => {
                let args = serde_json::from_str(arguments).unwrap_or(Value::Object(Map::new()));
                append_server_tool_parts_for_call(
                    &mut pending_model_parts,
                    &mut signature_metadata,
                    call_id,
                );
                pending_model_parts.push(function_call_part(
                    call_id,
                    name,
                    args,
                    signature_metadata.function_call_signatures.get(call_id),
                ));
            }
            ResponseItem::CustomToolCall {
                call_id,
                name,
                input,
                ..
            } => {
                let args = serde_json::from_str(input).unwrap_or(Value::String(input.clone()));
                append_server_tool_parts_for_call(
                    &mut pending_model_parts,
                    &mut signature_metadata,
                    call_id,
                );
                pending_model_parts.push(function_call_part(
                    call_id,
                    name,
                    args,
                    signature_metadata.function_call_signatures.get(call_id),
                ));
            }
            ResponseItem::FunctionCallOutput { call_id, output } => {
                pending_function_response_parts.push(function_response_part(
                    call_id,
                    call_names.get(call_id).map(String::as_str),
                    output,
                ));
            }
            ResponseItem::CustomToolCallOutput {
                call_id,
                name,
                output,
            } => {
                pending_function_response_parts.push(function_response_part(
                    call_id,
                    name.as_deref(),
                    output,
                ));
            }
            ResponseItem::Reasoning { .. } => {}
            _ => {}
        }
    }

    flush_tool_exchange(
        &mut contents,
        &mut pending_model_parts,
        &mut pending_function_response_parts,
    );
    contents
}

fn flush_tool_exchange(
    contents: &mut Vec<Value>,
    pending_model_parts: &mut Vec<Value>,
    pending_function_response_parts: &mut Vec<Value>,
) {
    if !pending_model_parts.is_empty() {
        push_content(contents, "model", std::mem::take(pending_model_parts));
    }
    if !pending_function_response_parts.is_empty() {
        push_content(
            contents,
            "user",
            std::mem::take(pending_function_response_parts),
        );
    }
}

fn push_content(contents: &mut Vec<Value>, role: &str, parts: Vec<Value>) {
    if parts.is_empty() {
        return;
    }
    if let Some(last) = contents.last_mut()
        && last.get("role").and_then(Value::as_str) == Some(role)
        && let Some(existing_parts) = last.get_mut("parts").and_then(Value::as_array_mut)
    {
        existing_parts.extend(parts);
        return;
    }
    contents.push(json!({"role": role, "parts": parts}));
}

fn prepend_server_tool_parts_for_message(
    content: &[ContentItem],
    metadata: &mut GeminiSignatureMetadata,
    message_parts: &mut Vec<Value>,
) {
    let Some(text) = gemini_output_text_key(content) else {
        if !metadata.orphan_server_tool_parts.is_empty() {
            let mut server_parts = std::mem::take(&mut metadata.orphan_server_tool_parts);
            server_parts.append(message_parts);
            *message_parts = server_parts;
        }
        return;
    };
    let Some(position) = metadata
        .server_tool_parts_by_text
        .iter()
        .position(|entry| entry.text == text)
    else {
        if !metadata.orphan_server_tool_parts.is_empty() {
            let mut server_parts = std::mem::take(&mut metadata.orphan_server_tool_parts);
            server_parts.append(message_parts);
            *message_parts = server_parts;
        }
        return;
    };
    let mut server_parts = metadata.server_tool_parts_by_text.remove(position).parts;
    server_parts.append(message_parts);
    *message_parts = server_parts;
}

fn append_server_tool_parts_for_call(
    pending_model_parts: &mut Vec<Value>,
    metadata: &mut GeminiSignatureMetadata,
    call_id: &str,
) {
    if let Some(mut server_parts) = metadata.server_tool_parts_by_call_id.remove(call_id) {
        pending_model_parts.append(&mut server_parts);
    } else if !metadata.orphan_server_tool_parts.is_empty() {
        pending_model_parts.append(&mut metadata.orphan_server_tool_parts);
    }
}

fn gemini_output_text_key(content: &[ContentItem]) -> Option<String> {
    let text = content
        .iter()
        .filter_map(|item| match item {
            ContentItem::OutputText { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<String>();
    if text.is_empty() { None } else { Some(text) }
}

fn gemini_message_parts(
    content: &[ContentItem],
    preserve_text_signatures: bool,
    text_signatures: &mut Vec<GeminiTextPartSignature>,
) -> Vec<Value> {
    let mut parts = Vec::new();
    for item in content {
        match item {
            ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                let mut part = json!({"text": text});
                if preserve_text_signatures
                    && let ContentItem::OutputText { .. } = item
                    && let Some(position) = text_signatures
                        .iter()
                        .position(|signature| signature.text == *text)
                {
                    let signature = text_signatures.remove(position);
                    if let Some(object) = part.as_object_mut() {
                        object.insert(
                            "thoughtSignature".to_string(),
                            Value::String(signature.thought_signature),
                        );
                    }
                }
                parts.push(part);
            }
            ContentItem::InputImage { image_url, .. } => {
                if let Some((mime_type, data)) = image_url
                    .strip_prefix("data:")
                    .and_then(|rest| rest.split_once(";base64,"))
                {
                    parts.push(json!({
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": data,
                        }
                    }));
                }
            }
        }
    }
    parts
}

fn function_call_part(
    call_id: &str,
    name: &str,
    args: Value,
    thought_signature: Option<&String>,
) -> Value {
    let mut part = json!({
        "functionCall": {
            "id": call_id,
            "name": name,
            "args": args,
        }
    });
    if let Some(signature) = thought_signature
        && let Some(object) = part.as_object_mut()
    {
        object.insert(
            "thoughtSignature".to_string(),
            Value::String(signature.clone()),
        );
    }
    part
}

fn function_response_part(
    call_id: &str,
    name: Option<&str>,
    output: &FunctionCallOutputPayload,
) -> Value {
    let response = match &output.body {
        FunctionCallOutputBody::Text(text) => json!({"output": text}),
        FunctionCallOutputBody::ContentItems(_) => {
            json!({"output": output.body.to_text().unwrap_or_default()})
        }
    };

    let function_response = json!({
        "id": call_id,
        "name": name.unwrap_or(call_id),
        "response": response,
    });
    json!({ "functionResponse": function_response })
}

fn gemini_function_call_names(items: &[ResponseItem]) -> HashMap<String, String> {
    let mut names = HashMap::new();
    for item in items {
        match item {
            ResponseItem::FunctionCall { call_id, name, .. }
            | ResponseItem::CustomToolCall { call_id, name, .. } => {
                names.insert(call_id.clone(), name.clone());
            }
            _ => {}
        }
    }
    names
}

fn gemini_signature_metadata(items: &[ResponseItem]) -> GeminiSignatureMetadata {
    let mut metadata = GeminiSignatureMetadata::default();
    for item in items {
        let ResponseItem::Reasoning {
            encrypted_content: Some(encrypted_content),
            ..
        } = item
        else {
            continue;
        };
        let Some(json) = encrypted_content.strip_prefix(SIGNATURE_METADATA_PREFIX) else {
            continue;
        };
        if let Ok(function_call_signatures) = serde_json::from_str::<HashMap<String, String>>(json)
        {
            metadata
                .function_call_signatures
                .extend(function_call_signatures);
            continue;
        }
        if let Ok(parsed) = serde_json::from_str::<GeminiSignatureMetadata>(json) {
            metadata.merge(parsed);
        }
    }
    metadata
}

fn gemini_tools(tools: &[Value]) -> Vec<Value> {
    let declarations: Vec<Value> = tools
        .iter()
        .flat_map(|tool| gemini_tool_declarations(tool).unwrap_or_default())
        .collect();
    let mut gemini_tools = Vec::new();
    if !declarations.is_empty() {
        gemini_tools.push(json!({ "functionDeclarations": declarations }));
    }
    if tools.iter().any(is_web_search_tool) {
        gemini_tools.push(json!({ "google_search": {} }));
    }
    gemini_tools
}

fn is_web_search_tool(tool: &Value) -> bool {
    tool.get("type").and_then(Value::as_str) == Some("web_search")
}

fn gemini_tool_declarations(tool: &Value) -> Option<Vec<Value>> {
    match tool.get("type").and_then(Value::as_str) {
        Some("function") => gemini_function_declaration(tool).map(|tool| vec![tool]),
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
                        let mut converted = gemini_function_declaration(tool)?;
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

fn gemini_function_declaration(tool: &Value) -> Option<Value> {
    let parameters = tool
        .get("parameters")
        .cloned()
        .map(sanitize_gemini_schema)
        .unwrap_or_else(|| json!({"type": "object"}));
    Some(json!({
        "name": tool.get("name")?.clone(),
        "description": tool.get("description").cloned().unwrap_or(Value::String(String::new())),
        "parameters": parameters,
    }))
}

fn sanitize_gemini_schema(value: Value) -> Value {
    match value {
        Value::Object(mut object) => {
            object.remove("additionalProperties");
            object.remove("$schema");
            object.remove("$id");
            object.remove("title");
            object.remove("default");
            if let Some(value) = object.remove("any_of") {
                object.insert("anyOf".to_string(), value);
            }
            if let Some(value) = object.remove("one_of") {
                object.insert("oneOf".to_string(), value);
            }
            if let Some(value) = object.remove("all_of") {
                object.insert("allOf".to_string(), value);
            }
            for value in object.values_mut() {
                let taken = std::mem::take(value);
                *value = sanitize_gemini_schema(taken);
            }
            sanitize_required_properties(&mut object);
            sanitize_enum_values(&mut object);
            Value::Object(object)
        }
        Value::Array(values) => {
            Value::Array(values.into_iter().map(sanitize_gemini_schema).collect())
        }
        other => other,
    }
}

fn sanitize_required_properties(object: &mut Map<String, Value>) {
    let property_names = object
        .get("properties")
        .and_then(Value::as_object)
        .map(|properties| {
            properties
                .keys()
                .cloned()
                .collect::<std::collections::HashSet<_>>()
        });
    let Some(required) = object.get_mut("required").and_then(Value::as_array_mut) else {
        return;
    };
    let Some(property_names) = property_names else {
        object.remove("required");
        return;
    };
    required.retain(|value| {
        value
            .as_str()
            .is_some_and(|property| property_names.contains(property))
    });
    if required.is_empty() {
        object.remove("required");
    }
}

fn sanitize_enum_values(object: &mut Map<String, Value>) {
    let Some(values) = object.get_mut("enum").and_then(Value::as_array_mut) else {
        return;
    };
    values.retain(|value| value.as_str() != Some(""));
    if values.is_empty() {
        object.remove("enum");
    }
}

fn response_stream_from_sse(bytes: codex_client::ByteStream) -> ResponseStream {
    let (tx_event, rx_event) = mpsc::channel::<Result<ResponseEvent, ApiError>>(1600);
    tokio::spawn(async move {
        let _ = tx_event.send(Ok(ResponseEvent::Created)).await;
        let mut stream = bytes.eventsource();
        let mut state = GeminiStreamState::default();

        while let Some(next) = stream.next().await {
            let sse = match next {
                Ok(sse) => sse,
                Err(err) => {
                    let _ = tx_event.send(Err(ApiError::Stream(err.to_string()))).await;
                    return;
                }
            };
            if sse.data.trim() == "[DONE]" {
                break;
            }
            let response: GeminiResponse = match serde_json::from_str(&sse.data) {
                Ok(response) => response,
                Err(err) => {
                    let _ = tx_event
                        .send(Err(ApiError::Stream(format!(
                            "failed to decode Gemini stream event: {err}"
                        ))))
                        .await;
                    return;
                }
            };
            if process_gemini_stream_response(&tx_event, &mut state, response)
                .await
                .is_err()
            {
                return;
            }
        }

        if flush_gemini_streaming_output_text(&tx_event, &mut state)
            .await
            .is_err()
        {
            return;
        }
        if !state.pending_server_tool_parts.is_empty() {
            state
                .signature_metadata
                .orphan_server_tool_parts
                .append(&mut state.pending_server_tool_parts);
        }
        if !state.reasoning_summary_text.is_empty() {
            let item =
                gemini_reasoning_summary_item(std::mem::take(&mut state.reasoning_summary_text));
            let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
        }
        if !state.signature_metadata.is_empty() {
            let item =
                gemini_signature_reasoning_item(std::mem::take(&mut state.signature_metadata));
            let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
        }
        let _ = tx_event
            .send(Ok(ResponseEvent::Completed {
                response_id: state
                    .response_id
                    .unwrap_or_else(|| "gemini-response".to_string()),
                token_usage: state.usage_metadata.map(Into::into),
            }))
            .await;
    });
    ResponseStream { rx_event }
}

#[derive(Default)]
struct GeminiStreamState {
    response_id: Option<String>,
    usage_metadata: Option<GeminiUsageMetadata>,
    signature_metadata: GeminiSignatureMetadata,
    pending_server_tool_parts: Vec<Value>,
    pending_text_thought_signature: Option<String>,
    reasoning_summary_text: String,
    text_open: bool,
    text_block_index: u32,
    output_text: String,
}

#[derive(Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct GeminiSignatureMetadata {
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    function_call_signatures: HashMap<String, String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    text_part_signatures: Vec<GeminiTextPartSignature>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    server_tool_parts_by_call_id: HashMap<String, Vec<Value>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    server_tool_parts_by_text: Vec<GeminiServerToolPartsForText>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    orphan_server_tool_parts: Vec<Value>,
}

impl GeminiSignatureMetadata {
    fn is_empty(&self) -> bool {
        self.function_call_signatures.is_empty()
            && self.text_part_signatures.is_empty()
            && self.server_tool_parts_by_call_id.is_empty()
            && self.server_tool_parts_by_text.is_empty()
            && self.orphan_server_tool_parts.is_empty()
    }

    fn merge(&mut self, other: GeminiSignatureMetadata) {
        self.function_call_signatures
            .extend(other.function_call_signatures);
        self.text_part_signatures.extend(other.text_part_signatures);
        self.server_tool_parts_by_call_id
            .extend(other.server_tool_parts_by_call_id);
        self.server_tool_parts_by_text
            .extend(other.server_tool_parts_by_text);
        self.orphan_server_tool_parts
            .extend(other.orphan_server_tool_parts);
    }
}

#[derive(Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct GeminiTextPartSignature {
    text: String,
    thought_signature: String,
}

#[derive(Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct GeminiServerToolPartsForText {
    text: String,
    parts: Vec<Value>,
}

async fn process_gemini_stream_response(
    tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>,
    state: &mut GeminiStreamState,
    response: GeminiResponse,
) -> Result<(), ()> {
    if let Some(response_id) = response.response_id {
        state.response_id = Some(response_id);
    }
    if response.usage_metadata.is_some() {
        state.usage_metadata = response.usage_metadata;
    }

    let Some(candidate) = response.candidates.into_iter().next() else {
        return Ok(());
    };
    if let Some(grounding_metadata) = candidate.grounding_metadata {
        send_gemini_web_search_item(tx_event, grounding_metadata).await?;
    }
    let parts = candidate
        .content
        .map(|content| content.parts)
        .unwrap_or_default();
    for part in parts {
        if let Some(server_tool_part) = part.server_tool_part_value() {
            state.pending_server_tool_parts.push(server_tool_part);
        }
        if let Some(text) = part.text.as_deref()
            && !text.is_empty()
        {
            if part.thought == Some(true) {
                state.reasoning_summary_text.push_str(text);
            } else {
                if let Some(signature) = &part.thought_signature {
                    state.pending_text_thought_signature = Some(signature.clone());
                }
                send_gemini_streaming_output_text_delta(tx_event, state, text).await?;
            }
        }
        if let Some(function_call) = part.function_call {
            flush_gemini_streaming_output_text(tx_event, state).await?;
            let call_id = if function_call.id.is_empty() {
                function_call.name.clone()
            } else {
                function_call.id
            };
            if !state.pending_server_tool_parts.is_empty() {
                state
                    .signature_metadata
                    .server_tool_parts_by_call_id
                    .entry(call_id.clone())
                    .or_default()
                    .append(&mut state.pending_server_tool_parts);
            }
            if let Some(signature) = part.thought_signature {
                state
                    .signature_metadata
                    .function_call_signatures
                    .insert(call_id.clone(), signature);
            }
            let item = ResponseItem::FunctionCall {
                id: None,
                name: function_call.name,
                namespace: None,
                arguments: function_call.args.to_string(),
                call_id,
            };
            tx_event
                .send(Ok(ResponseEvent::OutputItemDone(item)))
                .await
                .map_err(|_| ())?;
        }
    }
    Ok(())
}

async fn send_gemini_web_search_item(
    tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>,
    grounding_metadata: GeminiGroundingMetadata,
) -> Result<(), ()> {
    if grounding_metadata.web_search_queries.is_empty() {
        return Ok(());
    }
    let query = grounding_metadata.web_search_queries.first().cloned();
    let queries = Some(grounding_metadata.web_search_queries);
    tx_event
        .send(Ok(ResponseEvent::OutputItemDone(
            ResponseItem::WebSearchCall {
                id: Some("gemini-google-search".to_string()),
                status: Some("completed".to_string()),
                action: Some(WebSearchAction::Search { query, queries }),
            },
        )))
        .await
        .map_err(|_| ())
}

async fn send_gemini_streaming_output_text_delta(
    tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>,
    state: &mut GeminiStreamState,
    text: &str,
) -> Result<(), ()> {
    if !state.text_open {
        state.text_open = true;
        let item = ResponseItem::Message {
            id: Some(format!("gemini-message-{}", state.text_block_index)),
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText {
                text: String::new(),
            }],
            end_turn: None,
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

async fn flush_gemini_streaming_output_text(
    tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>,
    state: &mut GeminiStreamState,
) -> Result<(), ()> {
    if !state.text_open {
        return Ok(());
    }
    let text = std::mem::take(&mut state.output_text);
    if let Some(signature) = state.pending_text_thought_signature.take()
        && !text.is_empty()
    {
        state
            .signature_metadata
            .text_part_signatures
            .push(GeminiTextPartSignature {
                text: text.clone(),
                thought_signature: signature,
            });
    }
    if !state.pending_server_tool_parts.is_empty() && !text.is_empty() {
        state
            .signature_metadata
            .server_tool_parts_by_text
            .push(GeminiServerToolPartsForText {
                text: text.clone(),
                parts: std::mem::take(&mut state.pending_server_tool_parts),
            });
    }
    let item = ResponseItem::Message {
        id: Some(format!("gemini-message-{}", state.text_block_index)),
        role: "assistant".to_string(),
        content: vec![ContentItem::OutputText { text }],
        end_turn: None,
        phase: None,
    };
    state.text_open = false;
    state.text_block_index += 1;
    tx_event
        .send(Ok(ResponseEvent::OutputItemDone(item)))
        .await
        .map_err(|_| ())
}

fn gemini_reasoning_summary_item(text: String) -> ResponseItem {
    ResponseItem::Reasoning {
        id: "gemini-reasoning-summary".to_string(),
        summary: vec![ReasoningItemReasoningSummary::SummaryText { text }],
        content: None,
        encrypted_content: None,
    }
}

fn gemini_signature_reasoning_item(metadata: GeminiSignatureMetadata) -> ResponseItem {
    let encrypted_content = serde_json::to_string(&metadata).unwrap_or_else(|_| "{}".to_string());
    ResponseItem::Reasoning {
        id: String::new(),
        summary: Vec::<ReasoningItemReasoningSummary>::new(),
        content: None,
        encrypted_content: Some(format!("{SIGNATURE_METADATA_PREFIX}{encrypted_content}")),
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponse {
    #[serde(default)]
    candidates: Vec<GeminiCandidate>,
    usage_metadata: Option<GeminiUsageMetadata>,
    response_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiCandidate {
    content: Option<GeminiContent>,
    grounding_metadata: Option<GeminiGroundingMetadata>,
}

#[derive(Debug, Deserialize)]
struct GeminiContent {
    #[serde(default)]
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiPart {
    text: Option<String>,
    function_call: Option<GeminiFunctionCall>,
    tool_call: Option<Value>,
    tool_response: Option<Value>,
    executable_code: Option<Value>,
    code_execution_result: Option<Value>,
    thought_signature: Option<String>,
    thought: Option<bool>,
}

impl GeminiPart {
    fn server_tool_part_value(&self) -> Option<Value> {
        if self.tool_call.is_none()
            && self.tool_response.is_none()
            && self.executable_code.is_none()
            && self.code_execution_result.is_none()
        {
            return None;
        }
        let mut object = Map::new();
        if let Some(signature) = &self.thought_signature {
            object.insert(
                "thoughtSignature".to_string(),
                Value::String(signature.clone()),
            );
        }
        if let Some(tool_call) = &self.tool_call {
            object.insert("toolCall".to_string(), tool_call.clone());
        }
        if let Some(tool_response) = &self.tool_response {
            object.insert("toolResponse".to_string(), tool_response.clone());
        }
        if let Some(executable_code) = &self.executable_code {
            object.insert("executableCode".to_string(), executable_code.clone());
        }
        if let Some(code_execution_result) = &self.code_execution_result {
            object.insert(
                "codeExecutionResult".to_string(),
                code_execution_result.clone(),
            );
        }
        Some(Value::Object(object))
    }
}

#[derive(Debug, Deserialize)]
struct GeminiFunctionCall {
    #[serde(default)]
    id: String,
    name: String,
    #[serde(default)]
    args: Value,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiGroundingMetadata {
    #[serde(default)]
    web_search_queries: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiUsageMetadata {
    prompt_token_count: Option<i64>,
    cached_content_token_count: Option<i64>,
    candidates_token_count: Option<i64>,
    thoughts_token_count: Option<i64>,
    total_token_count: Option<i64>,
}

impl From<GeminiUsageMetadata> for TokenUsage {
    fn from(value: GeminiUsageMetadata) -> Self {
        let input_tokens = value.prompt_token_count.unwrap_or(0);
        let cached_input_tokens = value.cached_content_token_count.unwrap_or(0);
        let output_tokens = value.candidates_token_count.unwrap_or(0);
        let reasoning_output_tokens = value.thoughts_token_count.unwrap_or(0);
        TokenUsage {
            input_tokens,
            cached_input_tokens,
            output_tokens,
            reasoning_output_tokens,
            total_tokens: value
                .total_token_count
                .unwrap_or(input_tokens + output_tokens + reasoning_output_tokens),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gemini_text_part(text: &str) -> GeminiPart {
        GeminiPart {
            text: Some(text.to_string()),
            function_call: None,
            tool_call: None,
            tool_response: None,
            executable_code: None,
            code_execution_result: None,
            thought_signature: None,
            thought: None,
        }
    }

    fn gemini_function_call_part(
        id: &str,
        name: &str,
        args: Value,
        thought_signature: Option<&str>,
    ) -> GeminiPart {
        GeminiPart {
            text: None,
            function_call: Some(GeminiFunctionCall {
                id: id.to_string(),
                name: name.to_string(),
                args,
            }),
            tool_call: None,
            tool_response: None,
            executable_code: None,
            code_execution_result: None,
            thought_signature: thought_signature.map(str::to_string),
            thought: None,
        }
    }

    fn gemini_candidate(parts: Vec<GeminiPart>) -> GeminiCandidate {
        GeminiCandidate {
            grounding_metadata: None,
            content: Some(GeminiContent { parts }),
        }
    }

    fn gemini_usage(
        prompt_token_count: i64,
        cached_content_token_count: i64,
        candidates_token_count: i64,
        thoughts_token_count: i64,
        total_token_count: i64,
    ) -> GeminiUsageMetadata {
        GeminiUsageMetadata {
            prompt_token_count: Some(prompt_token_count),
            cached_content_token_count: Some(cached_content_token_count),
            candidates_token_count: Some(candidates_token_count),
            thoughts_token_count: Some(thoughts_token_count),
            total_token_count: Some(total_token_count),
        }
    }

    #[test]
    fn gemini_namespace_tools_use_canonical_display_names() {
        let tools = gemini_tools(&[json!({
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

        assert_eq!(
            tools[0]["functionDeclarations"][0]["name"],
            "mcp__delegento__google_ads_request"
        );
    }

    #[test]
    fn gemini_contents_preserves_function_call_signature_metadata() {
        let contents = gemini_contents(&[
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "make a file".to_string(),
                }],
                end_turn: None,
                phase: None,
            },
            ResponseItem::Reasoning {
                id: String::new(),
                summary: Vec::new(),
                content: None,
                encrypted_content: Some(format!(
                    "{SIGNATURE_METADATA_PREFIX}{}",
                    serde_json::to_string(&HashMap::from([(
                        "call_1".to_string(),
                        "signature_1".to_string(),
                    )]))
                    .unwrap()
                )),
            },
            ResponseItem::FunctionCall {
                id: None,
                name: "shell".to_string(),
                namespace: None,
                arguments: r#"{"cmd":"echo hello"}"#.to_string(),
                call_id: "call_1".to_string(),
            },
            ResponseItem::FunctionCallOutput {
                call_id: "call_1".to_string(),
                output: FunctionCallOutputPayload::from_text("ok".to_string()),
            },
        ]);

        assert_eq!(contents.len(), 3);
        assert_eq!(contents[1]["role"], "model");
        assert_eq!(contents[1]["parts"][0]["thoughtSignature"], "signature_1");
        assert_eq!(contents[2]["role"], "user");
        assert_eq!(contents[2]["parts"][0]["functionResponse"]["id"], "call_1");
    }

    #[test]
    fn sanitize_gemini_schema_removes_unsupported_json_schema_fields() {
        let schema = sanitize_gemini_schema(json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["items", "missing"],
            "properties": {
                "items": {
                    "type": "array",
                    "any_of": [{"type": "string"}],
                    "items": {
                        "type": "object",
                        "additionalProperties": false,
                        "default": {},
                        "required": ["missing"],
                        "enum": ["", "ok"],
                    }
                }
            }
        }));

        assert!(schema.pointer("/additionalProperties").is_none());
        assert_eq!(schema.pointer("/required/0"), Some(&json!("items")));
        assert!(schema.pointer("/required/1").is_none());
        assert!(schema.pointer("/properties/items/any_of").is_none());
        assert!(schema.pointer("/properties/items/anyOf").is_some());
        assert!(
            schema
                .pointer("/properties/items/items/additionalProperties")
                .is_none()
        );
        assert!(schema.pointer("/properties/items/items/default").is_none());
        assert!(schema.pointer("/properties/items/items/required").is_none());
        assert_eq!(
            schema.pointer("/properties/items/items/enum/0"),
            Some(&json!("ok"))
        );
    }

    #[test]
    fn gemini_tools_maps_web_search_to_google_search() {
        let tools = gemini_tools(&[
            json!({"type": "web_search", "external_web_access": true}),
            json!({
                "type": "function",
                "name": "shell",
                "description": "Run a shell command",
                "parameters": {"type": "object"}
            }),
        ]);

        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0]["functionDeclarations"][0]["name"], "shell");
        assert_eq!(tools[1], json!({"google_search": {}}));
    }

    #[test]
    fn gemini_request_body_enables_server_side_tool_invocations_for_web_search() {
        let request = ResponsesApiRequest {
            model: "gemini-3.1-pro-preview-customtools".to_string(),
            instructions: String::new(),
            input: Vec::new(),
            tools: vec![json!({"type": "web_search", "external_web_access": true})],
            tool_choice: "auto".to_string(),
            parallel_tool_calls: true,
            reasoning: None,
            store: false,
            stream: true,
            include: Vec::new(),
            service_tier: None,
            prompt_cache_key: None,
            text: None,
            client_metadata: None,
        };

        let body = gemini_request_body(&request);

        assert_eq!(body["toolConfig"]["includeServerSideToolInvocations"], true);
        assert_eq!(
            body["toolConfig"]["functionCallingConfig"]["mode"],
            "VALIDATED"
        );
    }

    #[test]
    fn gemini_request_body_maps_reasoning_effort_to_thinking_level() {
        let request = ResponsesApiRequest {
            model: "gemini-3.1-pro-preview-customtools".to_string(),
            instructions: String::new(),
            input: Vec::new(),
            tools: Vec::new(),
            tool_choice: "auto".to_string(),
            parallel_tool_calls: true,
            reasoning: Some(Reasoning {
                effort: Some(ReasoningEffort::Medium),
                summary: Some(ReasoningSummary::Concise),
            }),
            store: false,
            stream: true,
            include: Vec::new(),
            service_tier: None,
            prompt_cache_key: None,
            text: None,
            client_metadata: None,
        };

        let body = gemini_request_body(&request);

        assert_eq!(
            body["generationConfig"]["thinkingConfig"]["thinkingLevel"],
            "medium"
        );
        assert_eq!(
            body["generationConfig"]["thinkingConfig"]["includeThoughts"],
            true
        );
    }

    #[test]
    fn gemini_request_body_omits_unsupported_minimal_thinking_level() {
        let request = ResponsesApiRequest {
            model: "gemini-3.1-pro-preview-customtools".to_string(),
            instructions: String::new(),
            input: Vec::new(),
            tools: Vec::new(),
            tool_choice: "auto".to_string(),
            parallel_tool_calls: true,
            reasoning: Some(Reasoning {
                effort: Some(ReasoningEffort::Minimal),
                summary: None,
            }),
            store: false,
            stream: true,
            include: Vec::new(),
            service_tier: None,
            prompt_cache_key: None,
            text: None,
            client_metadata: None,
        };

        let body = gemini_request_body(&request);

        assert!(body["generationConfig"].get("thinkingConfig").is_none());
    }

    #[tokio::test]
    async fn gemini_stream_response_emits_text_deltas_before_completion() {
        let (tx, mut rx) = mpsc::channel::<Result<ResponseEvent, ApiError>>(8);
        let mut state = GeminiStreamState::default();

        process_gemini_stream_response(
            &tx,
            &mut state,
            GeminiResponse {
                response_id: Some("gemini_1".to_string()),
                usage_metadata: None,
                candidates: vec![gemini_candidate(vec![gemini_text_part("hello ")])],
            },
        )
        .await
        .unwrap();
        process_gemini_stream_response(
            &tx,
            &mut state,
            GeminiResponse {
                response_id: None,
                usage_metadata: Some(gemini_usage(7, 3, 2, 1, 10)),
                candidates: vec![gemini_candidate(vec![gemini_text_part("world")])],
            },
        )
        .await
        .unwrap();
        flush_gemini_streaming_output_text(&tx, &mut state)
            .await
            .unwrap();

        match rx.recv().await.unwrap().unwrap() {
            ResponseEvent::OutputItemAdded(ResponseItem::Message { id, .. }) => {
                assert_eq!(id.as_deref(), Some("gemini-message-0"));
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

        assert_eq!(state.response_id.as_deref(), Some("gemini_1"));
        let usage: TokenUsage = state.usage_metadata.unwrap().into();
        assert_eq!(usage.input_tokens, 7);
        assert_eq!(usage.cached_input_tokens, 3);
        assert_eq!(usage.output_tokens, 2);
        assert_eq!(usage.reasoning_output_tokens, 1);
        assert_eq!(usage.total_tokens, 10);
    }

    #[tokio::test]
    async fn gemini_stream_response_flushes_text_before_function_call() {
        let (tx, mut rx) = mpsc::channel::<Result<ResponseEvent, ApiError>>(8);
        let mut state = GeminiStreamState::default();

        process_gemini_stream_response(
            &tx,
            &mut state,
            GeminiResponse {
                response_id: Some("gemini_2".to_string()),
                usage_metadata: None,
                candidates: vec![gemini_candidate(vec![
                    gemini_text_part("I'll run it."),
                    gemini_function_call_part(
                        "call_1",
                        "shell",
                        json!({"cmd": "echo ok"}),
                        Some("sig_1"),
                    ),
                ])],
            },
        )
        .await
        .unwrap();

        assert!(matches!(
            rx.recv().await.unwrap().unwrap(),
            ResponseEvent::OutputItemAdded(ResponseItem::Message { .. })
        ));
        assert!(matches!(
            rx.recv().await.unwrap().unwrap(),
            ResponseEvent::OutputTextDelta(text) if text == "I'll run it."
        ));
        match rx.recv().await.unwrap().unwrap() {
            ResponseEvent::OutputItemDone(ResponseItem::Message { content, .. }) => {
                assert_eq!(
                    content,
                    vec![ContentItem::OutputText {
                        text: "I'll run it.".to_string()
                    }]
                );
            }
            event => panic!("expected OutputItemDone message, got {event:?}"),
        }
        match rx.recv().await.unwrap().unwrap() {
            ResponseEvent::OutputItemDone(ResponseItem::FunctionCall {
                name,
                arguments,
                call_id,
                ..
            }) => {
                assert_eq!(name, "shell");
                assert_eq!(call_id, "call_1");
                assert_eq!(arguments, json!({"cmd": "echo ok"}).to_string());
            }
            event => panic!("expected FunctionCall, got {event:?}"),
        }
        assert_eq!(
            state.signature_metadata.function_call_signatures,
            HashMap::from([("call_1".to_string(), "sig_1".to_string())])
        );
    }

    #[tokio::test]
    async fn gemini_stream_response_tolerates_metadata_only_chunks() {
        let (tx, _rx) = mpsc::channel::<Result<ResponseEvent, ApiError>>(8);
        let mut state = GeminiStreamState::default();

        process_gemini_stream_response(
            &tx,
            &mut state,
            serde_json::from_value(json!({
                "usageMetadata": {
                    "promptTokenCount": 11,
                    "cachedContentTokenCount": 5,
                    "totalTokenCount": 11
                }
            }))
            .unwrap(),
        )
        .await
        .unwrap();
        process_gemini_stream_response(
            &tx,
            &mut state,
            serde_json::from_value(json!({
                "candidates": [{"finishReason": "SAFETY"}]
            }))
            .unwrap(),
        )
        .await
        .unwrap();

        let usage: TokenUsage = state.usage_metadata.unwrap().into();
        assert_eq!(usage.input_tokens, 11);
        assert_eq!(usage.cached_input_tokens, 5);
    }

    #[tokio::test]
    async fn gemini_stream_response_collects_thought_summary_parts() {
        let (tx, _rx) = mpsc::channel::<Result<ResponseEvent, ApiError>>(8);
        let mut state = GeminiStreamState::default();

        process_gemini_stream_response(
            &tx,
            &mut state,
            GeminiResponse {
                response_id: Some("gemini_reasoning".to_string()),
                usage_metadata: None,
                candidates: vec![gemini_candidate(vec![GeminiPart {
                    text: Some("Checking the recent docs.".to_string()),
                    function_call: None,
                    tool_call: None,
                    tool_response: None,
                    executable_code: None,
                    code_execution_result: None,
                    thought_signature: None,
                    thought: Some(true),
                }])],
            },
        )
        .await
        .unwrap();

        assert_eq!(state.reasoning_summary_text, "Checking the recent docs.");
    }

    #[tokio::test]
    async fn gemini_preserves_server_tool_parts_before_function_call() {
        let (tx, mut rx) = mpsc::channel::<Result<ResponseEvent, ApiError>>(8);
        let mut state = GeminiStreamState::default();

        process_gemini_stream_response(
            &tx,
            &mut state,
            GeminiResponse {
                response_id: Some("gemini_tools".to_string()),
                usage_metadata: None,
                candidates: vec![gemini_candidate(vec![
                    GeminiPart {
                        text: None,
                        function_call: None,
                        tool_call: Some(json!({
                            "toolType": "GOOGLE_SEARCH_WEB",
                            "args": {"queries": ["current codex cli release"]},
                            "id": "search_1"
                        })),
                        tool_response: None,
                        executable_code: None,
                        code_execution_result: None,
                        thought_signature: Some("tool_sig_1".to_string()),
                        thought: None,
                    },
                    gemini_function_call_part(
                        "call_1",
                        "shell",
                        json!({"cmd": "pwd"}),
                        Some("call_sig_1"),
                    ),
                ])],
            },
        )
        .await
        .unwrap();

        match rx.recv().await.unwrap().unwrap() {
            ResponseEvent::OutputItemDone(ResponseItem::FunctionCall { call_id, .. }) => {
                assert_eq!(call_id, "call_1");
            }
            event => panic!("expected FunctionCall, got {event:?}"),
        }

        let contents = gemini_contents(&[
            gemini_signature_reasoning_item(std::mem::take(&mut state.signature_metadata)),
            ResponseItem::FunctionCall {
                id: None,
                name: "shell".to_string(),
                namespace: None,
                arguments: r#"{"cmd":"pwd"}"#.to_string(),
                call_id: "call_1".to_string(),
            },
        ]);

        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["parts"][0]["toolCall"]["id"], "search_1");
        assert_eq!(contents[0]["parts"][0]["thoughtSignature"], "tool_sig_1");
        assert_eq!(contents[0]["parts"][1]["functionCall"]["id"], "call_1");
        assert_eq!(contents[0]["parts"][1]["thoughtSignature"], "call_sig_1");
    }

    #[tokio::test]
    async fn gemini_stream_response_maps_grounding_metadata_to_web_search() {
        let (tx, mut rx) = mpsc::channel::<Result<ResponseEvent, ApiError>>(8);
        let mut state = GeminiStreamState::default();

        process_gemini_stream_response(
            &tx,
            &mut state,
            serde_json::from_value(json!({
                "responseId": "gemini_3",
                "candidates": [{
                    "content": {"parts": []},
                    "groundingMetadata": {
                        "webSearchQueries": [
                            "latest codex cli release".to_string(),
                            "openai codex cli 0.124".to_string(),
                        ]
                    }
                }]
            }))
            .unwrap(),
        )
        .await
        .unwrap();

        match rx.recv().await.unwrap().unwrap() {
            ResponseEvent::OutputItemDone(ResponseItem::WebSearchCall { id, status, action }) => {
                assert_eq!(id.as_deref(), Some("gemini-google-search"));
                assert_eq!(status.as_deref(), Some("completed"));
                assert_eq!(
                    action,
                    Some(WebSearchAction::Search {
                        query: Some("latest codex cli release".to_string()),
                        queries: Some(vec![
                            "latest codex cli release".to_string(),
                            "openai codex cli 0.124".to_string(),
                        ]),
                    })
                );
            }
            event => panic!("expected WebSearchCall, got {event:?}"),
        }
    }
}
