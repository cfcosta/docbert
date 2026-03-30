import { useEffect, useMemo, useState } from "react";
import { getProviders, getModels } from "@mariozechner/pi-ai";
import { api } from "../lib/api";
import "./Settings.css";

// Build the provider/model list from pi-ai's registry at module load.
function buildProviderMap() {
  const map: Record<string, { label: string; models: { id: string; name: string }[] }> = {};

  for (const provider of getProviders()) {
    const models = getModels(provider);
    if (models.length === 0) continue;

    map[provider] = {
      label: formatProviderLabel(provider),
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
    "openai-codex": "OpenAI Codex",
    "github-copilot": "GitHub Copilot",
    "vercel-ai-gateway": "Vercel AI Gateway",
    minimax: "MiniMax",
  };
  return labels[provider] ?? provider;
}

const PROVIDERS = buildProviderMap();

export default function Settings() {
  const [provider, setProvider] = useState<string | null>(null);
  const [model, setModel] = useState<string | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getLlmSettings()
      .then((s) => {
        setProvider(s.provider);
        setModel(s.model);
        setApiKey(s.api_key ?? "");
        setLoadError(null);
      })
      .catch((error) => {
        setLoadError(error instanceof Error ? error.message : "Could not load settings.");
      })
      .finally(() => setLoading(false));
  }, []);

  const handleProviderChange = (p: string) => {
    setProvider(p);
    const models = PROVIDERS[p]?.models ?? [];
    setModel(models[0]?.id ?? null);
    setSaved(false);
    setSaveError(null);
  };

  const handleModelChange = (m: string) => {
    setModel(m);
    setSaved(false);
    setSaveError(null);
  };

  const handleSave = async () => {
    setSaving(true);
    setSaved(false);
    setSaveError(null);
    try {
      await api.updateLlmSettings({
        provider,
        model,
        api_key: apiKey || null,
      });
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : "Could not save settings.");
    } finally {
      setSaving(false);
    }
  };

  const models = useMemo(() => (provider ? (PROVIDERS[provider]?.models ?? []) : []), [provider]);

  const providerEntries = useMemo(() => Object.entries(PROVIDERS), []);

  if (loading) {
    return (
      <div className="settings-page">
        <header className="settings-header">
          <h2>Settings</h2>
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
        <h2>Settings</h2>
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
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
            </div>
          )}

          {provider && (
            <div className="settings-field">
              <label className="settings-label" htmlFor="api-key">
                API Key
              </label>
              <input
                id="api-key"
                type="password"
                className="settings-input"
                placeholder={`Enter ${PROVIDERS[provider]?.label ?? provider} API key`}
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
              disabled={saving || !provider || !model}
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
