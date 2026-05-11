import { useEffect, useId, useRef, useState } from "react";

import { usePrefersDarkMode } from "../lib/use-prefers-dark";
import "./MermaidDiagram.css";

const MOCHA_THEME_VARIABLES = {
  darkMode: true,
  background: "#1e1e2e",
  primaryColor: "#313244",
  primaryTextColor: "#cdd6f4",
  primaryBorderColor: "#89b4fa",
  secondaryColor: "#45475a",
  secondaryTextColor: "#cdd6f4",
  secondaryBorderColor: "#89b4fa",
  tertiaryColor: "#585b70",
  tertiaryTextColor: "#cdd6f4",
  tertiaryBorderColor: "#74c7ec",
  lineColor: "#7f849c",
  textColor: "#cdd6f4",
  mainBkg: "#313244",
  secondBkg: "#45475a",
  noteBkgColor: "#45475a",
  noteTextColor: "#cdd6f4",
  noteBorderColor: "#89b4fa",
  titleColor: "#cdd6f4",
  edgeLabelBackground: "#181825",
  clusterBkg: "#181825",
  clusterBorder: "#45475a",
  defaultLinkColor: "#7f849c",
  nodeTextColor: "#cdd6f4",
  errorBkgColor: "#f38ba8",
  errorTextColor: "#11111b",
};

const LATTE_THEME_VARIABLES = {
  darkMode: false,
  background: "#eff1f5",
  primaryColor: "#ccd0da",
  primaryTextColor: "#4c4f69",
  primaryBorderColor: "#1e66f5",
  secondaryColor: "#bcc0cc",
  secondaryTextColor: "#4c4f69",
  secondaryBorderColor: "#1e66f5",
  tertiaryColor: "#acb0be",
  tertiaryTextColor: "#4c4f69",
  tertiaryBorderColor: "#209fb5",
  lineColor: "#8c8fa1",
  textColor: "#4c4f69",
  mainBkg: "#ccd0da",
  secondBkg: "#bcc0cc",
  noteBkgColor: "#bcc0cc",
  noteTextColor: "#4c4f69",
  noteBorderColor: "#1e66f5",
  titleColor: "#4c4f69",
  edgeLabelBackground: "#e6e9ef",
  clusterBkg: "#e6e9ef",
  clusterBorder: "#bcc0cc",
  defaultLinkColor: "#8c8fa1",
  nodeTextColor: "#4c4f69",
  errorBkgColor: "#d20f39",
  errorTextColor: "#eff1f5",
};

const MERMAID_FONT_FAMILY = '"Public Sans", "Segoe UI", "Helvetica Neue", Arial, sans-serif';

export default function MermaidDiagram({ code }: { code: string }) {
  const reactId = useId();
  const safeId = reactId.replace(/[^a-zA-Z0-9]/g, "");
  const prefersDark = usePrefersDarkMode();
  const [svg, setSvg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const seqRef = useRef(0);

  useEffect(() => {
    const seq = ++seqRef.current;
    let cancelled = false;

    void (async () => {
      try {
        const mermaid = (await import("mermaid")).default;
        if (cancelled || seq !== seqRef.current) return;

        mermaid.initialize({
          startOnLoad: false,
          securityLevel: "strict",
          theme: "base",
          themeVariables: prefersDark ? MOCHA_THEME_VARIABLES : LATTE_THEME_VARIABLES,
          fontFamily: MERMAID_FONT_FAMILY,
        });

        const parsed = await mermaid.parse(code, { suppressErrors: true });
        if (cancelled || seq !== seqRef.current) return;
        if (parsed === false) {
          setError("Invalid Mermaid diagram");
          setSvg(null);
          return;
        }

        const renderId = `mermaid-${safeId}-${seq}`;
        const result = await mermaid.render(renderId, code);
        if (cancelled || seq !== seqRef.current) return;

        setSvg(result.svg);
        setError(null);
      } catch (err) {
        if (cancelled || seq !== seqRef.current) return;
        setError(err instanceof Error ? err.message : "Failed to render Mermaid diagram");
        setSvg(null);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [code, prefersDark, safeId]);

  if (error !== null) {
    return (
      <div className="mermaid-diagram mermaid-diagram-error" role="alert">
        <div className="mermaid-diagram-error-banner">{error}</div>
        <pre className="mermaid-diagram-source">{code}</pre>
      </div>
    );
  }

  if (svg === null) {
    return (
      <div
        className="mermaid-diagram mermaid-diagram-loading"
        aria-busy="true"
        aria-label="Rendering Mermaid diagram"
      />
    );
  }

  return (
    <div
      className="mermaid-diagram"
      role="img"
      aria-label="Mermaid diagram"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}
