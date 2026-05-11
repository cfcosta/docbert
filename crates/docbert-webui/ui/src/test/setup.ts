import { afterEach } from "bun:test";
import { cleanup } from "@testing-library/react";
import { GlobalRegistrator } from "@happy-dom/global-registrator";

GlobalRegistrator.register({
  url: "http://localhost/",
  width: 1280,
  height: 720,
});

Object.defineProperty(globalThis, "IS_REACT_ACT_ENVIRONMENT", {
  value: true,
  configurable: true,
  writable: true,
});

Object.defineProperty(window.HTMLElement.prototype, "scrollIntoView", {
  value: () => {},
  configurable: true,
  writable: true,
});

Object.defineProperty(window, "matchMedia", {
  value: () => ({
    matches: false,
    media: "",
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  }),
  configurable: true,
  writable: true,
});

afterEach(() => {
  cleanup();
  document.body.innerHTML = "";
});
