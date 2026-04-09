import { useEffect, useMemo, useState } from "react";
import { getModels, getProviders } from "@mariozechner/pi-ai";
import { api, type LlmSettings } from "../lib/api";
import "./Settings.css";

type ProviderAuthMode = "api_key" | "oauth";

type ProviderEntry = {
  label: string;
  authMode: ProviderAuthMode;
  models: { id: string; name: string }[];
};

function providerAuthMode(provider: string): ProviderAuthMode {
  return provider === "openai-codex" ? "oauth" : "api_key";
}

// Build the provider/model list from pi-ai's registry at module load.
function buildProviderMap() {
  const map: Record<string, ProviderEntry> = {};

  for (const provider of getProviders()) {
    const models = getModels(provider);
    if (models.length === 0) continue;

    map[provider] = {
      label: formatProviderLabel(provider),
      authMode: providerAuthMode(provider),
      models: models.map((m) => ({ id: m.id, name: m.name || m.id })),
    };
  }

  return map;
}

function formatProviderLabel(provider: string): string {
  const labels: Record<string, string> = {
    anthropic: "Anthropic",
    openai: "OpenAI",
    google: "Google",
    "google-vertex": "Google Vertex",
    "google-gemini-cli": "Gemini CLI",
    mistral: "Mistral",
    groq: "Groq",
    cerebras: "Cerebras",
    xai: "xAI",
    openrouter: "OpenRouter",
    "amazon-bedrock": "Amazon Bedrock",
    "azure-openai-responses": "Azure OpenAI",
    "openai-codex": "ChatGPT Plus/Pro (Codex)",
    "github-copilot": "GitHub Copilot",
    "vercel-ai-gateway": "Vercel AI Gateway",
    minimax: "MiniMax",
  };
  return labels[provider] ?? provider;
}

const PROVIDERS = buildProviderMap();

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export default function Settings() {
  const [provider, setProvider] = useState<string | null>(null);
  const [model, setModel] = useState<string | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [oauthConnected, setOauthConnected] = useState(false);
  const [oauthExpiresAt, setOauthExpiresAt] = useState<number | null>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [oauthBusy, setOauthBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  const applyLoadedSettings = (settings: LlmSettings) => {
    setProvider(settings.provider);
    setModel(settings.model);
    setApiKey(settings.provider === "openai-codex" ? "" : (settings.api_key ?? ""));
    setOauthConnected(Boolean(settings.oauth_connected));
    setOauthExpiresAt(settings.oauth_expires_at ?? null);
  };

  useEffect(() => {
    api
      .getLlmSettings()
      .then((settings) => {
        applyLoadedSettings(settings);
        setLoadError(null);
      })
      .catch((error) => {
        setLoadError(error instanceof Error ? error.message : "Could not load settings.");
      })
      .finally(() => setLoading(false));
  }, []);

  const handleProviderChange = (nextProvider: string) => {
    setProvider(nextProvider);
    const models = PROVIDERS[nextProvider]?.models ?? [];
    setModel(models[0]?.id ?? null);
    if (providerAuthMode(nextProvider) === "api_key") {
      setApiKey("");
    }
    setSaved(false);
    setSaveError(null);
  };

  const handleModelChange = (nextModel: string) => {
    setModel(nextModel);
    setSaved(false);
    setSaveError(null);
  };

  const handleSave = async () => {
    setSaving(true);
    setSaved(false);
    setSaveError(null);
    try {
      const next = await api.updateLlmSettings({
        provider,
        model,
        api_key: provider && providerAuthMode(provider) === "api_key" ? apiKey || null : null,
      });
      applyLoadedSettings(next);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : "Could not save settings.");
    } finally {
      setSaving(false);
    }
  };

  const handleOpenAICodexLogin = async () => {
    if (!provider || !model) return;

    setOauthBusy(true);
    setSaved(false);
    setSaveError(null);

    try {
      await api.updateLlmSettings({
        provider,
        model,
        api_key: null,
      });

      const { authorization_url } = await api.startOpenAICodexOAuth();
      const popup = window.open(authorization_url, "_blank", "noopener,noreferrer");
      if (!popup) {
        throw new Error("Popup blocked. Allow popups for docbert and try again.");
      }

      const deadline = Date.now() + 120_000;
      while (Date.now() < deadline) {
        await sleep(1000);
        const next = await api.getLlmSettings();
        if (next.oauth_connected) {
          applyLoadedSettings(next);
          setSaved(true);
          setTimeout(() => setSaved(false), 2000);
          return;
        }
      }

      throw new Error(
        "Sign-in is still pending. Finish the ChatGPT browser flow, then reload Settings.",
      );
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : "Could not start ChatGPT sign-in.");
    } finally {
      setOauthBusy(false);
    }
  };

  const handleOpenAICodexLogout = async () => {
    setOauthBusy(true);
    setSaved(false);
    setSaveError(null);
    try {
      await api.logoutOpenAICodexOAuth();
      setOauthConnected(false);
      setOauthExpiresAt(null);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : "Could not disconnect ChatGPT.");
    } finally {
      setOauthBusy(false);
    }
  };

  const selectedProvider = useMemo(
    () => (provider ? (PROVIDERS[provider] ?? null) : null),
    [provider],
  );
  const models = useMemo(
    () => (selectedProvider ? selectedProvider.models : []),
    [selectedProvider],
  );
  const providerEntries = useMemo(() => Object.entries(PROVIDERS), []);
  const usesOAuth = selectedProvider?.authMode === "oauth";
  const oauthExpiryText =
    oauthConnected && oauthExpiresAt
      ? new Date(oauthExpiresAt).toLocaleString(undefined, {
          dateStyle: "medium",
          timeStyle: "short",
        })
      : null;

  if (loading) {
    return (
      <div className="settings-page">
        <header className="settings-header">
          <div className="settings-header-inner">
            <p className="settings-eyebrow">Configuration</p>
            <div className="settings-header-copy">
              <h2>Settings</h2>
              <p className="settings-header-description">
                Manage the model provider used by docbert chat and agent workflows.
              </p>
            </div>
          </div>
        </header>
        <div className="settings-body">
          <p className="settings-loading">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="settings-page">
      <header className="settings-header">
        <div className="settings-header-inner">
          <p className="settings-eyebrow">Configuration</p>
          <div className="settings-header-copy">
            <h2>Settings</h2>
            <p className="settings-header-description">
              Manage the model provider used by docbert chat and agent workflows.
            </p>
          </div>
        </div>
      </header>
      <div className="settings-body">
        <section className="settings-section">
          <h3>LLM Provider</h3>
          <p className="settings-description">
            Choose the provider and model used for chat responses.
          </p>
          {loadError && (
            <p className="settings-error" role="alert">
              {loadError}
            </p>
          )}

          <div className="settings-field">
            <label className="settings-label" htmlFor="provider-select">
              Provider
            </label>
            <select
              id="provider-select"
              className="settings-select"
              value={provider ?? ""}
              onChange={(e) =>
                e.target.value ? handleProviderChange(e.target.value) : setProvider(null)
              }
            >
              <option value="">Select a provider...</option>
              {providerEntries.map(([key, { label }]) => (
                <option key={key} value={key}>
                  {label}
                </option>
              ))}
            </select>
          </div>

          {provider && models.length > 0 && (
            <div className="settings-field">
              <label className="settings-label" htmlFor="model-select">
                Model
              </label>
              <select
                id="model-select"
                className="settings-select"
                value={model ?? ""}
                onChange={(e) => handleModelChange(e.target.value)}
              >
                {models.map((entry) => (
                  <option key={entry.id} value={entry.id}>
                    {entry.name}
                  </option>
                ))}
              </select>
            </div>
          )}

          {provider && usesOAuth && (
            <div className="settings-field">
              <label className="settings-label">ChatGPT subscription</label>
              <div className="settings-oauth-card">
                <p className="settings-oauth-copy">
                  Use your ChatGPT Plus or Pro subscription through OAuth instead of an API key.
                </p>
                <div className="settings-oauth-status-row">
                  <span className={oauthConnected ? "settings-saved" : "settings-oauth-pending"}>
                    {oauthConnected ? "Connected" : "Not connected"}
                  </span>
                  {oauthExpiryText && (
                    <span className="settings-hint">
                      Session refresh deadline: {oauthExpiryText}
                    </span>
                  )}
                </div>
                <div className="settings-actions settings-oauth-actions">
                  <button
                    className="settings-save"
                    onClick={handleOpenAICodexLogin}
                    disabled={oauthBusy || saving || !provider || !model}
                  >
                    {oauthBusy ? "Waiting for ChatGPT..." : "Sign in with ChatGPT"}
                  </button>
                  {oauthConnected && (
                    <button
                      className="settings-secondary"
                      onClick={handleOpenAICodexLogout}
                      disabled={oauthBusy}
                    >
                      Disconnect
                    </button>
                  )}
                </div>
              </div>
            </div>
          )}

          {provider && !usesOAuth && (
            <div className="settings-field">
              <label className="settings-label" htmlFor="api-key">
                API Key
              </label>
              <input
                id="api-key"
                type="password"
                className="settings-input"
                placeholder={`Enter ${selectedProvider?.label ?? provider} API key`}
                value={apiKey}
                onChange={(e) => {
                  setApiKey(e.target.value);
                  setSaved(false);
                  setSaveError(null);
                }}
              />
              <p className="settings-hint">Falls back to environment variable if not set.</p>
            </div>
          )}

          <div className="settings-actions">
            <button
              className="settings-save"
              onClick={handleSave}
              disabled={saving || oauthBusy || !provider || !model}
            >
              {saving ? "Saving..." : "Save"}
            </button>
            {saved && <span className="settings-saved">Saved</span>}
            {saveError && (
              <span className="settings-error" role="alert">
                {saveError}
              </span>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
