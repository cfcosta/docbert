function e(e) {
  return e.replace(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g, ``);
}
function t(e, t, n) {
  return {
    temperature: t?.temperature,
    maxTokens: t?.maxTokens || Math.min(e.maxTokens, 32e3),
    signal: t?.signal,
    apiKey: n || t?.apiKey,
    cacheRetention: t?.cacheRetention,
    sessionId: t?.sessionId,
    headers: t?.headers,
    onPayload: t?.onPayload,
    maxRetryDelayMs: t?.maxRetryDelayMs,
    metadata: t?.metadata,
  };
}
function n(e) {
  return e === `xhigh` ? `high` : e;
}
function r(e, t, r, i) {
  let a = { minimal: 1024, low: 2048, medium: 8192, high: 16384, ...i }[n(r)],
    o = Math.min(e + a, t);
  return (o <= a && (a = Math.max(0, o - 1024)), { maxTokens: o, thinkingBudget: a });
}
function i(e, t, n) {
  let r = new Map(),
    i = e.map((e) => {
      if (e.role === `user`) return e;
      if (e.role === `toolResult`) {
        let t = r.get(e.toolCallId);
        return t && t !== e.toolCallId ? { ...e, toolCallId: t } : e;
      }
      if (e.role === `assistant`) {
        let i = e,
          a = i.provider === t.provider && i.api === t.api && i.model === t.id,
          o = i.content.flatMap((e) => {
            if (e.type === `thinking`)
              return e.redacted
                ? a
                  ? e
                  : []
                : a && e.thinkingSignature
                  ? e
                  : !e.thinking || e.thinking.trim() === ``
                    ? []
                    : a
                      ? e
                      : { type: `text`, text: e.thinking };
            if (e.type === `text`) return a ? e : { type: `text`, text: e.text };
            if (e.type === `toolCall`) {
              let o = e,
                s = o;
              if (
                (!a && o.thoughtSignature && ((s = { ...o }), delete s.thoughtSignature), !a && n)
              ) {
                let e = n(o.id, t, i);
                e !== o.id && (r.set(o.id, e), (s = { ...s, id: e }));
              }
              return s;
            }
            return e;
          });
        return { ...i, content: o };
      }
      return e;
    }),
    a = [],
    o = [],
    s = new Set();
  for (let e = 0; e < i.length; e++) {
    let t = i[e];
    if (t.role === `assistant`) {
      if (o.length > 0) {
        for (let e of o)
          s.has(e.id) ||
            a.push({
              role: `toolResult`,
              toolCallId: e.id,
              toolName: e.name,
              content: [{ type: `text`, text: `No result provided` }],
              isError: !0,
              timestamp: Date.now(),
            });
        ((o = []), (s = new Set()));
      }
      let e = t;
      if (e.stopReason === `error` || e.stopReason === `aborted`) continue;
      let n = e.content.filter((e) => e.type === `toolCall`);
      (n.length > 0 && ((o = n), (s = new Set())), a.push(t));
    } else if (t.role === `toolResult`) (s.add(t.toolCallId), a.push(t));
    else if (t.role === `user`) {
      if (o.length > 0) {
        for (let e of o)
          s.has(e.id) ||
            a.push({
              role: `toolResult`,
              toolCallId: e.id,
              toolName: e.name,
              content: [{ type: `text`, text: `No result provided` }],
              isError: !0,
              timestamp: Date.now(),
            });
        ((o = []), (s = new Set()));
      }
      a.push(t);
    } else a.push(t);
  }
  return a;
}
export { e as a, n as i, r as n, t as r, i as t };
