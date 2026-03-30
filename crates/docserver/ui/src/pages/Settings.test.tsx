import "../test/setup";

import { afterEach, describe, expect, mock, test } from "bun:test";
import { render } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { api, type LlmSettings } from "../lib/api";

mock.module("@mariozechner/pi-ai", () => ({
  getProviders: () => ["openai"],
  getModels: (provider: string) =>
    provider === "openai" ? [{ id: "gpt-4.1", name: "GPT-4.1" }] : [],
}));

const { default: Settings } = await import("./Settings");

const originalApi = { ...api };

function renderSettings() {
  return render(<Settings />);
}

async function waitForCondition(
  condition: () => boolean,
  message: () => string,
  timeoutMs = 1000,
) {
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
}: {
  settings?: LlmSettings;
  loadError?: Error;
  saveError?: Error;
}) {
  api.getLlmSettings = async () => {
    if (loadError) {
      throw loadError;
    }
    return settings ?? { provider: "openai", model: "gpt-4.1", api_key: null };
  };

  api.updateLlmSettings = async (nextSettings) => {
    if (saveError) {
      throw saveError;
    }
    return nextSettings;
  };
}

afterEach(() => {
  Object.assign(api, originalApi);
});

describe("Settings page", () => {
  test("load_failure_renders_inline_error_text", async () => {
    installSettingsApiStubs({ loadError: new Error("Load failed") });

    const view = renderSettings();

    await waitForCondition(
      () => view.container.textContent?.includes("Load failed") ?? false,
      () => `load error never rendered: ${JSON.stringify(view.container.textContent)}`,
    );

    expect(view.getByRole("alert").textContent).toContain("Load failed");
  });

  test("save_failure_renders_inline_error_text", async () => {
    installSettingsApiStubs({ saveError: new Error("Save failed") });

    const user = userEvent.setup({ pointerEventsCheck: 0 });
    const view = renderSettings();

    await waitForCondition(
      () => view.queryByRole("button", { name: "Save" })?.hasAttribute("disabled") === false,
      () => `save button never enabled: ${JSON.stringify(view.container.textContent)}`,
    );

    await user.click(view.getByRole("button", { name: "Save" }));

    await waitForCondition(
      () => view.container.textContent?.includes("Save failed") ?? false,
      () => `save error never rendered: ${JSON.stringify(view.container.textContent)}`,
    );

    expect(view.getByRole("alert").textContent).toContain("Save failed");
  });

  test("successful_save_still_renders_saved", async () => {
    installSettingsApiStubs({});

    const user = userEvent.setup({ pointerEventsCheck: 0 });
    const view = renderSettings();

    await waitForCondition(
      () => view.queryByRole("button", { name: "Save" })?.hasAttribute("disabled") === false,
      () => `save button never enabled: ${JSON.stringify(view.container.textContent)}`,
    );

    await user.click(view.getByRole("button", { name: "Save" }));

    await waitForCondition(
      () => view.container.textContent?.includes("Saved") ?? false,
      () => `saved state never rendered: ${JSON.stringify(view.container.textContent)}`,
    );

    expect(view.container.textContent).toContain("Saved");
  });
});
