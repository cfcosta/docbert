import { Type } from "@mariozechner/pi-ai";
import type { Context, Message as PiMessage, Tool } from "@mariozechner/pi-ai";

import { api } from "../lib/api";
import type { SearchResult } from "../lib/api";
import { createToolResultMessage } from "./chat-context";
import type { ToolCallInfo } from "./chat-message-codec";
import { mergeCurrentTurnSearchResults, type ChatToolRuntimeState } from "./chat-subagents";

export const searchChatTools: Tool[] = [
  {
    name: "search_semantic",
    description:
      "Search the document store using semantic (ColBERT) search. Best for meaning-based queries where wording may differ from the target documents. Searches all collections by default.",
    parameters: Type.Object({
      query: Type.String({ description: "The search query" }),
      collection: Type.Optional(
        Type.String({
          description: "Restrict search to this collection. Omit to search all collections.",
        }),
      ),
      count: Type.Optional(Type.Number({ description: "Number of results to return (default 5)" })),
    }),
  },
  {
    name: "search_hybrid",
    description:
      "Search the document store using hybrid BM25 + semantic search. Best when the query shares keywords with the target documents. Faster on large collections.",
    parameters: Type.Object({
      query: Type.String({ description: "The search query" }),
      collection: Type.Optional(Type.String({ description: "Restrict search to this collection" })),
      count: Type.Optional(Type.Number({ description: "Number of results to return (default 5)" })),
    }),
  },
];

type ToolCall = { id: string; name: string; arguments: Record<string, unknown> };
type SearchApi = typeof api.search;

function searchModeForTool(name: string): "semantic" | "hybrid" | null {
  switch (name) {
    case "search_semantic":
      return "semantic";
    case "search_hybrid":
      return "hybrid";
    default:
      return null;
  }
}

async function executeSearchTool(
  call: ToolCall,
  runtimeState: ChatToolRuntimeState,
  search: SearchApi,
): Promise<{ text: string; sources?: SearchResult[] }> {
  const mode = searchModeForTool(call.name);
  if (!mode) {
    return { text: `Unknown tool: ${call.name}` };
  }

  const results = await search({
    query: call.arguments.query as string,
    mode,
    collection: call.arguments.collection as string | undefined,
    count: (call.arguments.count as number) ?? 5,
  });

  runtimeState.currentTurnSearchResults = mergeCurrentTurnSearchResults(
    runtimeState.currentTurnSearchResults,
    results.results,
  );

  return {
    text: JSON.stringify(results.results, null, 2),
    sources: results.results,
  };
}

export async function executeSearchToolCall({
  call,
  piContext,
  runtimeState,
  search = api.search,
}: {
  call: ToolCall;
  piContext: Context;
  runtimeState: ChatToolRuntimeState;
  search?: SearchApi;
}): Promise<ToolCallInfo> {
  const callInfo: ToolCallInfo = { name: call.name, args: call.arguments };

  try {
    const toolResult = await executeSearchTool(call, runtimeState, search);
    callInfo.result = toolResult.text;
    piContext.messages.push(
      createToolResultMessage(call.id, call.name, toolResult.text, false) as PiMessage,
    );
  } catch (error) {
    const errText = error instanceof Error ? error.message : "unknown error";
    callInfo.result = `Error: ${errText}`;
    callInfo.isError = true;
    piContext.messages.push(
      createToolResultMessage(call.id, call.name, `Error: ${errText}`, true) as PiMessage,
    );
  }

  return callInfo;
}
