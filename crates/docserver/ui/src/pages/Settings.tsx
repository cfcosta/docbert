import { useEffect, useState } from "react";
import { api } from "../lib/api";
import "./Settings.css";

const PROVIDERS: Record<string, { label: string; models: string[] }> = {
  anthropic: {
    label: "Anthropic",
    models: ["claude-sonnet-4-5-20250514", "claude-haiku-4-5-20251001", "claude-opus-4-20250514"],
  },
  openai: {
    label: "OpenAI",
    models: [
      "gpt-4o",
      "gpt-4o-mini",
      "gpt-4.1",
      "gpt-4.1-mini",
      "gpt-4.1-nano",
      "o4-mini",
      "o3",
      "o3-mini",
    ],
  },
};

export default function Settings() {
  const [provider, setProvider] = useState<string | null>(null);
  const [model, setModel] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .getLlmSettings()
      .then((s) => {
        setProvider(s.provider);
        setModel(s.model);
      })
      .catch(() => {
        // Settings not configured yet — show the form anyway.
      })
      .finally(() => setLoading(false));
  }, []);

  const handleProviderChange = (p: string) => {
    setProvider(p);
    // Reset model to first option for this provider.
    const models = PROVIDERS[p]?.models ?? [];
    setModel(models[0] ?? null);
    setSaved(false);
  };

  const handleModelChange = (m: string) => {
    setModel(m);
    setSaved(false);
  };

  const handleSave = async () => {
    setSaving(true);
    setSaved(false);
    try {
      await api.updateLlmSettings({ provider, model });
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch {
      /* ignore */
    } finally {
      setSaving(false);
    }
  };

  const models = provider ? (PROVIDERS[provider]?.models ?? []) : [];

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

          <div className="settings-field">
            <label className="settings-label">Provider</label>
            <div className="settings-radio-group">
              {Object.entries(PROVIDERS).map(([key, { label }]) => (
                <button
                  key={key}
                  className={`settings-radio${provider === key ? " active" : ""}`}
                  onClick={() => handleProviderChange(key)}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {provider && models.length > 0 && (
            <div className="settings-field">
              <label className="settings-label">Model</label>
              <select
                className="settings-select"
                value={model ?? ""}
                onChange={(e) => handleModelChange(e.target.value)}
              >
                {models.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
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
          </div>
        </section>
      </div>
    </div>
  );
}
