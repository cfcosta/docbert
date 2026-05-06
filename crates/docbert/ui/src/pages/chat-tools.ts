import { Type } from "@mariozechner/pi-ai";
import type { Context, Message as PiMessage, Tool } from "@mariozechner/pi-ai";

import { api } from "../lib/api";
import type { SearchMode, SearchResult } from "../lib/api";
import { createToolResultMessage } from "./chat-context";
import type { ToolCallInfo } from "./chat-message-codec";
import { mergeCurrentTurnSearchResults, type ChatToolRuntimeState } from "./chat-subagents";

export const searchChatTools: Tool[] = [
  {
    name: "search_bm25",
    description:
      "Search the document store using BM25 keyword search only (no semantic / ColBERT leg). Best for exact terms, identifiers, symbols, file names, error strings, or any query where the user's wording is expected to appear verbatim in the target documents.",
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
    name: "search_semantic",
    description:
      "Search the document store using semantic (ColBERT) search only. Best for general concepts, meaning-based queries, or topics where the user's wording is unlikely to match the documents verbatim. Searches all collections by default.",
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
      "Search the document store using hybrid BM25 + semantic search fused with RRF. Best when the query mixes specific keywords and a general concept, or when you are unsure which signal matters more. Use search_bm25 or search_semantic when one signal is clearly the right one.",
    parameters: Type.Object({
      query: Type.String({ description: "The search query" }),
      collection: Type.Optional(Type.String({ description: "Restrict search to this collection" })),
      count: Type.Optional(Type.Number({ description: "Number of results to return (default 5)" })),
    }),
  },
];

type ToolCall = { id: string; name: string; arguments: Record<string, unknown> };
type SearchApi = typeof api.search;

export function searchModeForTool(name: string): SearchMode | null {
  switch (name) {
    case "search_semantic":
      return "semantic";
    case "search_hybrid":
      return "hybrid";
    case "search_bm25":
      return "bm25";
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
