use crate::common::ResponseEvent;
use crate::common::ResponseStream;
use crate::common::ResponsesApiRequest;
use crate::endpoint::session::EndpointSession;
use crate::error::ApiError;
use codex_client::HttpTransport;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::ReasoningItemReasoningSummary;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::TokenUsage;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use http::HeaderMap;
use http::HeaderValue;
use http::Method;
use serde::Deserialize;
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
    json!({
        "systemInstruction": {
            "parts": gemini_text_parts(&request.instructions),
        },
        "contents": gemini_contents(&request.input),
        "tools": gemini_tools(&request.tools),
        "toolConfig": {
            "functionCallingConfig": {
                "mode": "AUTO",
            }
        },
        "generationConfig": {
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
        },
    })
}

fn gemini_text_parts(text: &str) -> Vec<Value> {
    if text.is_empty() {
        Vec::new()
    } else {
        vec![json!({"text": text})]
    }
}

fn gemini_contents(items: &[ResponseItem]) -> Vec<Value> {
    let signatures = gemini_function_call_signatures(items);
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
                push_content(&mut contents, role, gemini_message_parts(content));
            }
            ResponseItem::FunctionCall {
                call_id,
                name,
                arguments,
                ..
            } => {
                let args = serde_json::from_str(arguments).unwrap_or(Value::Object(Map::new()));
                pending_model_parts.push(function_call_part(
                    call_id,
                    name,
                    args,
                    signatures.get(call_id),
                ));
            }
            ResponseItem::CustomToolCall {
                call_id,
                name,
                input,
                ..
            } => {
                let args = serde_json::from_str(input).unwrap_or(Value::String(input.clone()));
                pending_model_parts.push(function_call_part(
                    call_id,
                    name,
                    args,
                    signatures.get(call_id),
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

fn gemini_message_parts(content: &[ContentItem]) -> Vec<Value> {
    let mut parts = Vec::new();
    for item in content {
        match item {
            ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                parts.push(json!({"text": text}));
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

fn gemini_function_call_signatures(items: &[ResponseItem]) -> HashMap<String, String> {
    let mut signatures = HashMap::new();
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
        let Ok(parsed) = serde_json::from_str::<HashMap<String, String>>(json) else {
            continue;
        };
        signatures.extend(parsed);
    }
    signatures
}

fn gemini_tools(tools: &[Value]) -> Vec<Value> {
    let declarations: Vec<Value> = tools
        .iter()
        .flat_map(|tool| gemini_tool_declarations(tool).unwrap_or_default())
        .collect();
    if declarations.is_empty() {
        Vec::new()
    } else {
        vec![json!({ "functionDeclarations": declarations })]
    }
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
    signature_metadata: HashMap<String, String>,
    text_open: bool,
    text_block_index: u32,
    output_text: String,
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
    for part in candidate.content.parts {
        if let Some(text) = part.text
            && !text.is_empty()
        {
            send_gemini_streaming_output_text_delta(tx_event, state, &text).await?;
        }
        if let Some(function_call) = part.function_call {
            flush_gemini_streaming_output_text(tx_event, state).await?;
            let call_id = if function_call.id.is_empty() {
                function_call.name.clone()
            } else {
                function_call.id
            };
            if let Some(signature) = part.thought_signature {
                state.signature_metadata.insert(call_id.clone(), signature);
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
    let item = ResponseItem::Message {
        id: Some(format!("gemini-message-{}", state.text_block_index)),
        role: "assistant".to_string(),
        content: vec![ContentItem::OutputText {
            text: std::mem::take(&mut state.output_text),
        }],
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

fn gemini_signature_reasoning_item(signatures: HashMap<String, String>) -> ResponseItem {
    let encrypted_content = serde_json::to_string(&signatures).unwrap_or_else(|_| "{}".to_string());
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
    candidates: Vec<GeminiCandidate>,
    usage_metadata: Option<GeminiUsageMetadata>,
    response_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: GeminiContent,
}

#[derive(Debug, Deserialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiPart {
    text: Option<String>,
    function_call: Option<GeminiFunctionCall>,
    thought_signature: Option<String>,
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
struct GeminiUsageMetadata {
    prompt_token_count: Option<i64>,
    candidates_token_count: Option<i64>,
    thoughts_token_count: Option<i64>,
    total_token_count: Option<i64>,
}

impl From<GeminiUsageMetadata> for TokenUsage {
    fn from(value: GeminiUsageMetadata) -> Self {
        let input_tokens = value.prompt_token_count.unwrap_or(0);
        let output_tokens = value.candidates_token_count.unwrap_or(0);
        let reasoning_output_tokens = value.thoughts_token_count.unwrap_or(0);
        TokenUsage {
            input_tokens,
            cached_input_tokens: 0,
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
                candidates: vec![GeminiCandidate {
                    content: GeminiContent {
                        parts: vec![GeminiPart {
                            text: Some("hello ".to_string()),
                            function_call: None,
                            thought_signature: None,
                        }],
                    },
                }],
            },
        )
        .await
        .unwrap();
        process_gemini_stream_response(
            &tx,
            &mut state,
            GeminiResponse {
                response_id: None,
                usage_metadata: Some(GeminiUsageMetadata {
                    prompt_token_count: Some(7),
                    candidates_token_count: Some(2),
                    thoughts_token_count: Some(1),
                    total_token_count: Some(10),
                }),
                candidates: vec![GeminiCandidate {
                    content: GeminiContent {
                        parts: vec![GeminiPart {
                            text: Some("world".to_string()),
                            function_call: None,
                            thought_signature: None,
                        }],
                    },
                }],
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
                candidates: vec![GeminiCandidate {
                    content: GeminiContent {
                        parts: vec![
                            GeminiPart {
                                text: Some("I'll run it.".to_string()),
                                function_call: None,
                                thought_signature: None,
                            },
                            GeminiPart {
                                text: None,
                                function_call: Some(GeminiFunctionCall {
                                    id: "call_1".to_string(),
                                    name: "shell".to_string(),
                                    args: json!({"cmd": "echo ok"}),
                                }),
                                thought_signature: Some("sig_1".to_string()),
                            },
                        ],
                    },
                }],
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
            state.signature_metadata,
            HashMap::from([("call_1".to_string(), "sig_1".to_string())])
        );
    }
}
