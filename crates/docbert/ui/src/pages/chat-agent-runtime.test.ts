import { describe, expect, test } from "bun:test";

import {
  FILE_ANALYSIS_SYSTEM_PROMPT,
  chatAgentTools,
  createSubagentContext,
} from "./chat-agent-runtime";

describe("createSubagentContext", () => {
  test("builds an evidence-rich prompt with document details and structure", () => {
    const context = createSubagentContext(
      "How does the rollout work?",
      {
        collection: "notes",
        path: "plan.md",
        title: "Rollout Plan",
        content: "# Rollout\n\nStep 1\n\nStep 2",
        metadata: { owner: "ops", status: "draft" },
      },
      "Look for implementation steps.",
    );

    expect(context.systemPrompt).toBe(FILE_ANALYSIS_SYSTEM_PROMPT);
    expect(context.messages).toHaveLength(1);
    const message = context.messages[0];
    expect(message.role).toBe("user");
    expect(message.content).toContain("Analyze exactly this file: notes/plan.md");
    expect(message.content).toContain("Document title: Rollout Plan");
    expect(message.content).toContain('"owner": "ops"');
    expect(message.content).toContain("Extra focus: Look for implementation steps.");
    expect(message.content).toContain("## Key findings");
    expect(message.content).toContain("## Supporting evidence");
    expect(message.content).toContain("Be specific and information-dense");
    expect(message.content).toContain("# Rollout");
  });
});

describe("FILE_ANALYSIS_SYSTEM_PROMPT", () => {
  test("pushes the subagent toward evidence instead of generic brevity", () => {
    expect(FILE_ANALYSIS_SYSTEM_PROMPT).toContain("evidence-rich analysis");
    expect(FILE_ANALYSIS_SYSTEM_PROMPT).toContain("Do not optimize for brevity");
    expect(FILE_ANALYSIS_SYSTEM_PROMPT).toContain("short quoted phrases");
    expect(FILE_ANALYSIS_SYSTEM_PROMPT).toContain("weakly relevant");
  });
});

describe("chatAgentTools", () => {
  test("describes analyze_document as synthesis-friendly", () => {
    const analyzeTool = chatAgentTools.find((tool) => tool.name === "analyze_document");
    expect(analyzeTool).toBeDefined();
    expect(analyzeTool?.description).toContain("evidence-rich analysis");
    expect(analyzeTool?.description).toContain("final answer");
  });
});
