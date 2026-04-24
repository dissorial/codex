pub(super) fn base_url(region: &str) -> String {
    format!("https://bedrock-runtime.{region}.amazonaws.com")
}
