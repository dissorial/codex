use super::*;
use crate::ModelsManagerConfig;
use pretty_assertions::assert_eq;

#[test]
fn bedrock_claude_opus_46_has_explicit_model_metadata() {
    let model = model_info_from_slug("global.anthropic.claude-opus-4-6-v1");

    assert_eq!(model.slug, "global.anthropic.claude-opus-4-6-v1");
    assert_eq!(model.display_name, "Claude Opus 4.6 (Bedrock)");
    assert_eq!(model.context_window, Some(1_000_000));
    assert_eq!(model.default_reasoning_level, Some(ReasoningEffort::High));
    assert_eq!(model.supported_reasoning_levels.len(), 3);
    assert!(model.supports_reasoning_summaries);
    assert!(!model.used_fallback_model_metadata);
    assert!(model.model_messages.is_none());
}

#[test]
fn gemini_31_pro_customtools_has_explicit_model_metadata() {
    let model = model_info_from_slug("gemini-3.1-pro-preview-customtools");

    assert_eq!(model.slug, "gemini-3.1-pro-preview-customtools");
    assert_eq!(model.display_name, "Gemini 3.1 Pro Preview Custom Tools");
    assert_eq!(model.context_window, Some(1_048_576));
    assert_eq!(model.max_context_window, Some(1_048_576));
    assert!(model.supports_parallel_tool_calls);
    assert!(!model.used_fallback_model_metadata);
    assert!(model.model_messages.is_none());
}

#[test]
fn reasoning_summaries_override_true_enables_support() {
    let model = model_info_from_slug("unknown-model");
    let config = ModelsManagerConfig {
        model_supports_reasoning_summaries: Some(true),
        ..Default::default()
    };

    let updated = with_config_overrides(model.clone(), &config);
    let mut expected = model;
    expected.supports_reasoning_summaries = true;

    assert_eq!(updated, expected);
}

#[test]
fn reasoning_summaries_override_false_does_not_disable_support() {
    let mut model = model_info_from_slug("unknown-model");
    model.supports_reasoning_summaries = true;
    let config = ModelsManagerConfig {
        model_supports_reasoning_summaries: Some(false),
        ..Default::default()
    };

    let updated = with_config_overrides(model.clone(), &config);

    assert_eq!(updated, model);
}

#[test]
fn reasoning_summaries_override_false_is_noop_when_model_is_false() {
    let model = model_info_from_slug("unknown-model");
    let config = ModelsManagerConfig {
        model_supports_reasoning_summaries: Some(false),
        ..Default::default()
    };

    let updated = with_config_overrides(model.clone(), &config);

    assert_eq!(updated, model);
}

#[test]
fn model_context_window_override_clamps_to_max_context_window() {
    let mut model = model_info_from_slug("unknown-model");
    model.context_window = Some(273_000);
    model.max_context_window = Some(400_000);
    let config = ModelsManagerConfig {
        model_context_window: Some(500_000),
        ..Default::default()
    };

    let updated = with_config_overrides(model.clone(), &config);
    let mut expected = model;
    expected.context_window = Some(400_000);

    assert_eq!(updated, expected);
}

#[test]
fn model_context_window_uses_model_value_without_override() {
    let mut model = model_info_from_slug("unknown-model");
    model.context_window = Some(273_000);
    model.max_context_window = Some(400_000);
    let config = ModelsManagerConfig::default();

    let updated = with_config_overrides(model.clone(), &config);

    assert_eq!(updated, model);
}
