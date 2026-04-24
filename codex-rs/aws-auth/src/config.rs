use aws_config::BehaviorVersion;
use aws_config::SdkConfig;
use aws_credential_types::Credentials;
use aws_credential_types::provider::SharedCredentialsProvider;
use aws_types::region::Region;

use crate::AwsAuthConfig;
use crate::AwsAuthError;

pub(crate) async fn load_sdk_config(config: &AwsAuthConfig) -> Result<SdkConfig, AwsAuthError> {
    if config.service.trim().is_empty() {
        return Err(AwsAuthError::EmptyService);
    }

    let mut loader = aws_config::defaults(BehaviorVersion::latest());
    if let Some(profile) = config.profile.as_ref() {
        loader = loader.profile_name(profile);
    }
    if let Some(credentials_provider) = static_credentials_provider(config) {
        loader = loader.credentials_provider(credentials_provider);
    }
    if let Some(region) = config.region.as_ref() {
        loader = loader.region(Region::new(region.clone()));
    }

    Ok(loader.load().await)
}

fn static_credentials_provider(config: &AwsAuthConfig) -> Option<SharedCredentialsProvider> {
    let access_key_id = config
        .access_key_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())?;
    let secret_access_key = config
        .secret_access_key
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())?;
    let session_token = config
        .session_token
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);

    Some(SharedCredentialsProvider::new(Credentials::new(
        access_key_id,
        secret_access_key,
        session_token,
        /*expires_after*/ None,
        "codex-provider-config",
    )))
}

pub(crate) fn credentials_provider(
    sdk_config: &SdkConfig,
) -> Result<SharedCredentialsProvider, AwsAuthError> {
    sdk_config
        .credentials_provider()
        .ok_or(AwsAuthError::MissingCredentialsProvider)
}

pub(crate) fn resolved_region(sdk_config: &SdkConfig) -> Result<String, AwsAuthError> {
    sdk_config
        .region()
        .map(ToString::to_string)
        .ok_or(AwsAuthError::MissingRegion)
}
