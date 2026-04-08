import "../test/setup";

import { afterEach, describe, expect, mock, test } from "bun:test";
import { render, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { api, type LlmSettings } from "../lib/api";

mock.module("@mariozechner/pi-ai", () => ({
  getProviders: () => ["openai", "openai-codex"],
  getModels: (provider: string) => {
    if (provider === "openai") {
      return [{ id: "gpt-4.1", name: "GPT-4.1" }];
    }
    if (provider === "openai-codex") {
      return [{ id: "gpt-5.1-codex-mini", name: "GPT-5.1 Codex Mini" }];
    }
    return [];
  },
}));

const { default: Settings } = await import("./Settings");

const originalApi = { ...api };
const originalWindowOpen = window.open;

function renderSettings() {
  return render(<Settings />);
}

async function waitForCondition(condition: () => boolean, message: () => string, timeoutMs = 1000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (condition()) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 10));
  }

  throw new Error(message());
}

function installSettingsApiStubs({
  settings,
  loadError,
  saveError,
  startOAuth,
  logoutOAuth,
}: {
  settings?: LlmSettings;
  loadError?: Error;
  saveError?: Error;
  startOAuth?: () => Promise<{ authorization_url: string }>;
  logoutOAuth?: () => Promise<void>;
}) {
  let currentSettings =
    settings ??
    ({
      provider: "openai",
      model: "gpt-4.1",
      api_key: null,
      oauth_connected: false,
      oauth_expires_at: null,
    } satisfies LlmSettings);

  api.getLlmSettings = async () => {
    if (loadError) {
      throw loadError;
    }
    return currentSettings;
  };

  api.updateLlmSettings = async (nextSettings) => {
    if (saveError) {
      throw saveError;
    }
    currentSettings = {
      ...currentSettings,
      provider: nextSettings.provider,
      model: nextSettings.model,
      api_key: nextSettings.api_key,
    };
    return currentSettings;
  };

  api.startOpenAICodexOAuth =
    startOAuth ??
    (async () => ({
      authorization_url: "https://auth.openai.com/oauth/authorize?test=1",
    }));

  api.logoutOpenAICodexOAuth =
    logoutOAuth ??
    (async () => {
      currentSettings = {
        ...currentSettings,
        api_key: null,
        oauth_connected: false,
        oauth_expires_at: null,
      };
    });

  return {
    getCurrentSettings: () => currentSettings,
    setCurrentSettings: (next: LlmSettings) => {
      currentSettings = next;
    },
  };
}

afterEach(() => {
  Object.assign(api, originalApi);
  window.open = originalWindowOpen;
});

describe("Settings page", () => {
  test("load_failure_renders_inline_error_text", async () => {
    installSettingsApiStubs({ loadError: new Error("Load failed") });

    const view = renderSettings();

    await waitForCondition(
      () => view.container.textContent?.includes("Load failed") ?? false,
      () => `load error never rendered: ${JSON.stringify(view.container.textContent)}`,
    );

    expect(within(view.container).getByRole("alert").textContent).toContain("Load failed");
  });

  test("save_failure_renders_inline_error_text", async () => {
    installSettingsApiStubs({ saveError: new Error("Save failed") });

    const user = userEvent.setup({ pointerEventsCheck: 0 });
    const view = renderSettings();

    await waitForCondition(
      () =>
        within(view.container)
          .queryByRole("button", { name: /^save$/i })
          ?.hasAttribute("disabled") === false,
      () => `save button never enabled: ${JSON.stringify(view.container.textContent)}`,
    );

    await user.click(within(view.container).getByRole("button", { name: /^save$/i }));

    await waitForCondition(
      () => view.container.textContent?.includes("Save failed") ?? false,
      () => `save error never rendered: ${JSON.stringify(view.container.textContent)}`,
    );

    expect(within(view.container).getByRole("alert").textContent).toContain("Save failed");
  });

  test("successful_save_still_renders_saved", async () => {
    installSettingsApiStubs({});

    const user = userEvent.setup({ pointerEventsCheck: 0 });
    const view = renderSettings();

    await waitForCondition(
      () =>
        within(view.container)
          .queryByRole("button", { name: /^save$/i })
          ?.hasAttribute("disabled") === false,
      () => `save button never enabled: ${JSON.stringify(view.container.textContent)}`,
    );

    await user.click(within(view.container).getByRole("button", { name: /^save$/i }));

    await waitForCondition(
      () => view.container.textContent?.includes("Saved") ?? false,
      () => `saved state never rendered: ${JSON.stringify(view.container.textContent)}`,
    );

    expect(view.container.textContent).toContain("Saved");
  });

  test("openai_codex_provider_shows_oauth_sign_in_controls", async () => {
    installSettingsApiStubs({
      settings: {
        provider: "openai-codex",
        model: "gpt-5.1-codex-mini",
        api_key: null,
        oauth_connected: false,
        oauth_expires_at: null,
      },
    });

    const view = renderSettings();

    await waitForCondition(
      () => view.container.textContent?.includes("ChatGPT subscription") ?? false,
      () => `oauth controls never rendered: ${JSON.stringify(view.container.textContent)}`,
    );

    expect(
      within(view.container).getByRole("button", { name: /sign in with chatgpt/i }),
    ).toBeTruthy();
    expect(within(view.container).queryByLabelText(/api key/i)).toBeNull();
  });

  test("openai_codex_sign_in_starts_oauth_flow_and_refreshes_connected_state", async () => {
    const trackers = installSettingsApiStubs({
      settings: {
        provider: "openai-codex",
        model: "gpt-5.1-codex-mini",
        api_key: null,
        oauth_connected: false,
        oauth_expires_at: null,
      },
    });

    let startCount = 0;
    api.startOpenAICodexOAuth = async () => {
      startCount += 1;
      setTimeout(() => {
        trackers.setCurrentSettings({
          provider: "openai-codex",
          model: "gpt-5.1-codex-mini",
          api_key: "fresh-token",
          oauth_connected: true,
          oauth_expires_at: Date.now() + 60_000,
        });
      }, 20);
      return { authorization_url: "https://auth.openai.com/oauth/authorize?test=1" };
    };

    const openCalls: string[] = [];
    window.open = ((url?: string | URL) => {
      openCalls.push(String(url));
      return { closed: false } as Window;
    }) as typeof window.open;

    const user = userEvent.setup({ pointerEventsCheck: 0 });
    const view = renderSettings();

    await waitForCondition(
      () =>
        Boolean(within(view.container).queryByRole("button", { name: /sign in with chatgpt/i })),
      () => `sign-in button never rendered: ${JSON.stringify(view.container.textContent)}`,
    );

    await user.click(within(view.container).getByRole("button", { name: /sign in with chatgpt/i }));

    await waitForCondition(
      () => view.container.textContent?.includes("Connected") ?? false,
      () => `oauth never reached connected state: ${JSON.stringify(view.container.textContent)}`,
      2000,
    );

    expect(startCount).toBe(1);
    expect(openCalls[0]).toContain("auth.openai.com/oauth/authorize");
  });
});
