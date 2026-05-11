import { describe, expect, test } from "bun:test";

import type { Message } from "./chat-message-codec";
import { groupMessagesForDisplay } from "./chat-message-groups";

function subagentMessage(id: string, status: "queued" | "running" | "done" | "error"): Message {
  return {
    id,
    role: "assistant",
    content: `subagent ${id}`,
    actor: {
      type: "subagent",
      id,
      collection: "notes",
      path: `${id}.md`,
      status,
    },
  };
}

describe("groupMessagesForDisplay", () => {
  test("group_messages_nests_subagents_under_last_parent_assistant", () => {
    const groups = groupMessagesForDisplay([
      { id: "user-1", role: "user", content: "Question" },
      { id: "assistant-1", role: "assistant", content: "Answer" },
      subagentMessage("sub-1", "queued"),
    ]);

    expect(groups).toHaveLength(2);
    expect(groups[1].message.id).toBe("assistant-1");
    expect(groups[1].nestedSubagents.map((message) => message.id)).toEqual(["sub-1"]);
  });

  test("group_messages_keeps_leading_subagent_top_level", () => {
    const groups = groupMessagesForDisplay([subagentMessage("sub-1", "queued")]);

    expect(groups).toHaveLength(1);
    expect(groups[0].message.id).toBe("sub-1");
    expect(groups[0].nestedSubagents).toEqual([]);
  });

  test("group_messages_resets_nesting_after_non_parent_message", () => {
    const groups = groupMessagesForDisplay([
      { id: "assistant-1", role: "assistant", content: "Answer" },
      { id: "user-2", role: "user", content: "Follow-up" },
      subagentMessage("sub-1", "done"),
    ]);

    expect(groups).toHaveLength(3);
    expect(groups[0].nestedSubagents).toEqual([]);
    expect(groups[2].message.id).toBe("sub-1");
    expect(groups[2].nestedSubagents).toEqual([]);
  });
});
