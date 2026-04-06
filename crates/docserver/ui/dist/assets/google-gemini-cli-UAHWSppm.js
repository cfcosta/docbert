import { n as e, t } from "./event-stream-BhluV9ep.js";
import { a as n, i as r, r as i } from "./transform-messages-B9BoyWEJ.js";
import { a, n as o, o as s, r as c, s as l, t as u } from "./google-shared-CXnighqp.js";
var d = `https://cloudcode-pa.googleapis.com`,
  f = [
    `https://daily-cloudcode-pa.sandbox.googleapis.com`,
    `https://autopush-cloudcode-pa.sandbox.googleapis.com`,
    d,
  ],
  p = {
    "User-Agent": `google-cloud-sdk vscode_cloudshelleditor/0.1`,
    "X-Goog-Api-Client": `gl-node/22.17.0`,
    "Client-Metadata": JSON.stringify({
      ideType: `IDE_UNSPECIFIED`,
      platform: `PLATFORM_UNSPECIFIED`,
      pluginType: `GEMINI`,
    }),
  },
  m = `1.18.4`;
function h() {
  let e = {}.PI_AI_ANTIGRAVITY_VERSION || m;
  return { "User-Agent": `antigravity/${e} darwin/arm64` };
}
var g = `You are Antigravity, a powerful agentic AI coding assistant designed by the Google Deepmind team working on Advanced Agentic Coding.You are pair programming with a USER to solve their coding task. The task may require creating a new codebase, modifying or debugging an existing codebase, or simply answering a question.**Absolute paths only****Proactiveness**`,
  _ = 0,
  v = 3,
  y = 1e3,
  b = 2,
  x = 500,
  S = `interleaved-thinking-2025-05-14`;
function C(e, t) {
  let n = (e) => (e > 0 ? Math.ceil(e + 1e3) : void 0),
    r = t instanceof Headers ? t : t?.headers;
  if (r) {
    let e = r.get(`retry-after`);
    if (e) {
      let t = Number(e);
      if (Number.isFinite(t)) {
        let e = n(t * 1e3);
        if (e !== void 0) return e;
      }
      let r = new Date(e).getTime();
      if (!Number.isNaN(r)) {
        let e = n(r - Date.now());
        if (e !== void 0) return e;
      }
    }
    let t = r.get(`x-ratelimit-reset`);
    if (t) {
      let e = Number.parseInt(t, 10);
      if (!Number.isNaN(e)) {
        let t = n(e * 1e3 - Date.now());
        if (t !== void 0) return t;
      }
    }
    let i = r.get(`x-ratelimit-reset-after`);
    if (i) {
      let e = Number(i);
      if (Number.isFinite(e)) {
        let t = n(e * 1e3);
        if (t !== void 0) return t;
      }
    }
  }
  let i = e.match(/reset after (?:(\d+)h)?(?:(\d+)m)?(\d+(?:\.\d+)?)s/i);
  if (i) {
    let e = i[1] ? parseInt(i[1], 10) : 0,
      t = i[2] ? parseInt(i[2], 10) : 0,
      r = parseFloat(i[3]);
    if (!Number.isNaN(r)) {
      let i = n(((e * 60 + t) * 60 + r) * 1e3);
      if (i !== void 0) return i;
    }
  }
  let a = e.match(/Please retry in ([0-9.]+)(ms|s)/i);
  if (a?.[1]) {
    let e = parseFloat(a[1]);
    if (!Number.isNaN(e) && e > 0) {
      let t = n(a[2].toLowerCase() === `ms` ? e : e * 1e3);
      if (t !== void 0) return t;
    }
  }
  let o = e.match(/"retryDelay":\s*"([0-9.]+)(ms|s)"/i);
  if (o?.[1]) {
    let e = parseFloat(o[1]);
    if (!Number.isNaN(e) && e > 0) {
      let t = n(o[2].toLowerCase() === `ms` ? e : e * 1e3);
      if (t !== void 0) return t;
    }
  }
}
function w(e) {
  return e.provider === `google-antigravity` && e.id.startsWith(`claude-`) && e.reasoning;
}
function T(e) {
  return /gemini-3(?:\.1)?-pro/.test(e.toLowerCase());
}
function E(e) {
  return /gemini-3(?:\.1)?-flash/.test(e.toLowerCase());
}
function D(e) {
  return T(e) || E(e);
}
function O(e, t) {
  return e === 429 || e === 500 || e === 502 || e === 503 || e === 504
    ? !0
    : /resource.?exhausted|rate.?limit|overloaded|service.?unavailable|other.?side.?closed/i.test(
        t,
      );
}
function k(e) {
  try {
    let t = JSON.parse(e);
    if (t.error?.message) return t.error.message;
  } catch {}
  return e;
}
function A(e, t) {
  return new Promise((n, r) => {
    if (t?.aborted) {
      r(Error(`Request was aborted`));
      return;
    }
    let i = setTimeout(n, e);
    t?.addEventListener(`abort`, () => {
      (clearTimeout(i), r(Error(`Request was aborted`)));
    });
  });
}
var j = (n, r, i) => {
    let o = new t();
    return (
      (async () => {
        let t = {
          role: `assistant`,
          content: [],
          api: `google-gemini-cli`,
          provider: n.provider,
          model: n.id,
          usage: {
            input: 0,
            output: 0,
            cacheRead: 0,
            cacheWrite: 0,
            totalTokens: 0,
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
          },
          stopReason: `stop`,
          timestamp: Date.now(),
        };
        try {
          let s = i?.apiKey;
          if (!s)
            throw Error(
              `Google Cloud Code Assist requires OAuth authentication. Use /login to authenticate.`,
            );
          let u, m;
          try {
            let e = JSON.parse(s);
            ((u = e.token), (m = e.projectId));
          } catch {
            throw Error(
              `Invalid Google Cloud Code Assist credentials. Use /login to re-authenticate.`,
            );
          }
          if (!u || !m)
            throw Error(
              `Missing token or projectId in Google Cloud credentials. Use /login to re-authenticate.`,
            );
          let g = n.provider === `google-antigravity`,
            T = n.baseUrl?.trim(),
            E = T ? [T] : g ? f : [d],
            D = N(n, r, m, i, g),
            j = await i?.onPayload?.(D, n);
          j !== void 0 && (D = j);
          let M = g ? h() : p,
            P = {
              Authorization: `Bearer ${u}`,
              "Content-Type": `application/json`,
              Accept: `text/event-stream`,
              ...M,
              ...(w(n) ? { "anthropic-beta": S } : {}),
              ...i?.headers,
            },
            F = JSON.stringify(D),
            I,
            L,
            R,
            z = 0;
          for (let e = 0; e <= v; e++) {
            if (i?.signal?.aborted) throw Error(`Request was aborted`);
            try {
              if (
                ((R = `${E[z]}/v1internal:streamGenerateContent?alt=sse`),
                (I = await fetch(R, { method: `POST`, headers: P, body: F, signal: i?.signal })),
                I.ok)
              )
                break;
              let t = await I.text();
              if ((I.status === 403 || I.status === 404) && z < E.length - 1) {
                z++;
                continue;
              }
              if (e < v && O(I.status, t)) {
                z < E.length - 1 && z++;
                let n = C(t, I),
                  r = n ?? y * 2 ** e,
                  a = i?.maxRetryDelayMs ?? 6e4;
                if (a > 0 && n && n > a) {
                  let e = Math.ceil(n / 1e3);
                  throw Error(
                    `Server requested ${e}s retry delay (max: ${Math.ceil(a / 1e3)}s). ${k(t)}`,
                  );
                }
                await A(r, i?.signal);
                continue;
              }
              throw Error(`Cloud Code Assist API error (${I.status}): ${k(t)}`);
            } catch (t) {
              if (
                t instanceof Error &&
                (t.name === `AbortError` || t.message === `Request was aborted`)
              )
                throw Error(`Request was aborted`);
              if (
                ((L = t instanceof Error ? t : Error(String(t))),
                L.message === `fetch failed` &&
                  L.cause instanceof Error &&
                  (L = Error(`Network error: ${L.cause.message}`)),
                e < v)
              ) {
                await A(y * 2 ** e, i?.signal);
                continue;
              }
              throw L;
            }
          }
          if (!I || !I.ok) throw L ?? Error(`Failed to get response after retries`);
          let B = !1,
            V = () => {
              B ||= (o.push({ type: `start`, partial: t }), !0);
            },
            H = () => {
              ((t.content = []),
                (t.usage = {
                  input: 0,
                  output: 0,
                  cacheRead: 0,
                  cacheWrite: 0,
                  totalTokens: 0,
                  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
                }),
                (t.stopReason = `stop`),
                (t.errorMessage = void 0),
                (t.timestamp = Date.now()),
                (B = !1));
            },
            U = async (r) => {
              if (!r.body) throw Error(`No response body`);
              let s = !1,
                u = null,
                d = t.content,
                f = () => d.length - 1,
                p = r.body.getReader(),
                m = new TextDecoder(),
                h = ``,
                g = () => {
                  p.cancel().catch(() => {});
                };
              i?.signal?.addEventListener(`abort`, g);
              try {
                for (;;) {
                  if (i?.signal?.aborted) throw Error(`Request was aborted`);
                  let { done: r, value: g } = await p.read();
                  if (r) break;
                  h += m.decode(g, { stream: !0 });
                  let v = h.split(`
`);
                  h = v.pop() || ``;
                  for (let r of v) {
                    if (!r.startsWith(`data:`)) continue;
                    let i = r.slice(5).trim();
                    if (!i) continue;
                    let p;
                    try {
                      p = JSON.parse(i);
                    } catch {
                      continue;
                    }
                    let m = p.response;
                    if (!m) continue;
                    t.responseId ||= m.responseId;
                    let h = m.candidates?.[0];
                    if (h?.content?.parts)
                      for (let e of h.content.parts) {
                        if (e.text !== void 0) {
                          s = !0;
                          let n = c(e);
                          ((!u || (n && u.type !== `thinking`) || (!n && u.type !== `text`)) &&
                            (u &&
                              (u.type === `text`
                                ? o.push({
                                    type: `text_end`,
                                    contentIndex: d.length - 1,
                                    content: u.text,
                                    partial: t,
                                  })
                                : o.push({
                                    type: `thinking_end`,
                                    contentIndex: f(),
                                    content: u.thinking,
                                    partial: t,
                                  })),
                            n
                              ? ((u = {
                                  type: `thinking`,
                                  thinking: ``,
                                  thinkingSignature: void 0,
                                }),
                                t.content.push(u),
                                V(),
                                o.push({ type: `thinking_start`, contentIndex: f(), partial: t }))
                              : ((u = { type: `text`, text: `` }),
                                t.content.push(u),
                                V(),
                                o.push({ type: `text_start`, contentIndex: f(), partial: t }))),
                            u.type === `thinking`
                              ? ((u.thinking += e.text),
                                (u.thinkingSignature = l(u.thinkingSignature, e.thoughtSignature)),
                                o.push({
                                  type: `thinking_delta`,
                                  contentIndex: f(),
                                  delta: e.text,
                                  partial: t,
                                }))
                              : ((u.text += e.text),
                                (u.textSignature = l(u.textSignature, e.thoughtSignature)),
                                o.push({
                                  type: `text_delta`,
                                  contentIndex: f(),
                                  delta: e.text,
                                  partial: t,
                                })));
                        }
                        if (e.functionCall) {
                          ((s = !0),
                            (u &&=
                              (u.type === `text`
                                ? o.push({
                                    type: `text_end`,
                                    contentIndex: f(),
                                    content: u.text,
                                    partial: t,
                                  })
                                : o.push({
                                    type: `thinking_end`,
                                    contentIndex: f(),
                                    content: u.thinking,
                                    partial: t,
                                  }),
                              null)));
                          let n = e.functionCall.id,
                            r = {
                              type: `toolCall`,
                              id:
                                !n || t.content.some((e) => e.type === `toolCall` && e.id === n)
                                  ? `${e.functionCall.name}_${Date.now()}_${++_}`
                                  : n,
                              name: e.functionCall.name || ``,
                              arguments: e.functionCall.args ?? {},
                              ...(e.thoughtSignature && { thoughtSignature: e.thoughtSignature }),
                            };
                          (t.content.push(r),
                            V(),
                            o.push({ type: `toolcall_start`, contentIndex: f(), partial: t }),
                            o.push({
                              type: `toolcall_delta`,
                              contentIndex: f(),
                              delta: JSON.stringify(r.arguments),
                              partial: t,
                            }),
                            o.push({
                              type: `toolcall_end`,
                              contentIndex: f(),
                              toolCall: r,
                              partial: t,
                            }));
                        }
                      }
                    if (
                      (h?.finishReason &&
                        ((t.stopReason = a(h.finishReason)),
                        t.content.some((e) => e.type === `toolCall`) && (t.stopReason = `toolUse`)),
                      m.usageMetadata)
                    ) {
                      let r = m.usageMetadata.promptTokenCount || 0,
                        i = m.usageMetadata.cachedContentTokenCount || 0;
                      ((t.usage = {
                        input: r - i,
                        output:
                          (m.usageMetadata.candidatesTokenCount || 0) +
                          (m.usageMetadata.thoughtsTokenCount || 0),
                        cacheRead: i,
                        cacheWrite: 0,
                        totalTokens: m.usageMetadata.totalTokenCount || 0,
                        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
                      }),
                        e(n, t.usage));
                    }
                  }
                }
              } finally {
                i?.signal?.removeEventListener(`abort`, g);
              }
              return (
                u &&
                  (u.type === `text`
                    ? o.push({ type: `text_end`, contentIndex: f(), content: u.text, partial: t })
                    : o.push({
                        type: `thinking_end`,
                        contentIndex: f(),
                        content: u.thinking,
                        partial: t,
                      })),
                s
              );
            },
            W = !1,
            G = I;
          for (let e = 0; e <= b; e++) {
            if (i?.signal?.aborted) throw Error(`Request was aborted`);
            if (e > 0) {
              if ((await A(x * 2 ** (e - 1), i?.signal), !R)) throw Error(`Missing request URL`);
              if (
                ((G = await fetch(R, { method: `POST`, headers: P, body: F, signal: i?.signal })),
                !G.ok)
              ) {
                let e = await G.text();
                throw Error(`Cloud Code Assist API error (${G.status}): ${e}`);
              }
            }
            if (await U(G)) {
              W = !0;
              break;
            }
            e < b && H();
          }
          if (!W) throw Error(`Cloud Code Assist API returned an empty response`);
          if (i?.signal?.aborted) throw Error(`Request was aborted`);
          if (t.stopReason === `aborted` || t.stopReason === `error`)
            throw Error(`An unknown error occurred`);
          (o.push({ type: `done`, reason: t.stopReason, message: t }), o.end());
        } catch (e) {
          for (let e of t.content) `index` in e && delete e.index;
          ((t.stopReason = i?.signal?.aborted ? `aborted` : `error`),
            (t.errorMessage = e instanceof Error ? e.message : JSON.stringify(e)),
            o.push({ type: `error`, reason: t.stopReason, error: t }),
            o.end());
        }
      })(),
      o
    );
  },
  M = (e, t, n) => {
    let a = n?.apiKey;
    if (!a)
      throw Error(
        `Google Cloud Code Assist requires OAuth authentication. Use /login to authenticate.`,
      );
    let o = i(e, n, a);
    if (!n?.reasoning) return j(e, t, { ...o, thinking: { enabled: !1 } });
    let s = r(n.reasoning);
    if (D(e.id)) return j(e, t, { ...o, thinking: { enabled: !0, level: F(s, e.id) } });
    let c = { minimal: 1024, low: 2048, medium: 8192, high: 16384, ...n.thinkingBudgets }[s],
      l = Math.min((o.maxTokens || 0) + c, e.maxTokens);
    return (
      l <= c && (c = Math.max(0, l - 1024)),
      j(e, t, { ...o, maxTokens: l, thinking: { enabled: !0, budgetTokens: c } })
    );
  };
function N(e, t, r, i = {}, a = !1) {
  let c = u(e, t),
    l = {};
  (i.temperature !== void 0 && (l.temperature = i.temperature),
    i.maxTokens !== void 0 && (l.maxOutputTokens = i.maxTokens),
    i.thinking?.enabled && e.reasoning
      ? ((l.thinkingConfig = { includeThoughts: !0 }),
        i.thinking.level === void 0
          ? i.thinking.budgetTokens !== void 0 &&
            (l.thinkingConfig.thinkingBudget = i.thinking.budgetTokens)
          : (l.thinkingConfig.thinkingLevel = i.thinking.level))
      : e.reasoning && i.thinking && !i.thinking.enabled && (l.thinkingConfig = P(e.id)));
  let d = { contents: c };
  if (
    ((d.sessionId = i.sessionId),
    t.systemPrompt && (d.systemInstruction = { parts: [{ text: n(t.systemPrompt) }] }),
    Object.keys(l).length > 0 && (d.generationConfig = l),
    t.tools && t.tools.length > 0)
  ) {
    let n = e.id.startsWith(`claude-`);
    ((d.tools = o(t.tools, n)),
      i.toolChoice && (d.toolConfig = { functionCallingConfig: { mode: s(i.toolChoice) } }));
  }
  if (a) {
    let e = d.systemInstruction?.parts ?? [];
    d.systemInstruction = {
      role: `user`,
      parts: [{ text: g }, { text: `Please ignore following [ignore]${g}[/ignore]` }, ...e],
    };
  }
  return {
    project: r,
    model: e.id,
    request: d,
    ...(a ? { requestType: `agent` } : {}),
    userAgent: a ? `antigravity` : `pi-coding-agent`,
    requestId: `${a ? `agent` : `pi`}-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`,
  };
}
function P(e) {
  return T(e)
    ? { thinkingLevel: `LOW` }
    : E(e)
      ? { thinkingLevel: `MINIMAL` }
      : { thinkingBudget: 0 };
}
function F(e, t) {
  if (T(t))
    switch (e) {
      case `minimal`:
      case `low`:
        return `LOW`;
      case `medium`:
      case `high`:
        return `HIGH`;
    }
  switch (e) {
    case `minimal`:
      return `MINIMAL`;
    case `low`:
      return `LOW`;
    case `medium`:
      return `MEDIUM`;
    case `high`:
      return `HIGH`;
  }
}
export {
  N as buildRequest,
  C as extractRetryDelay,
  j as streamGoogleGeminiCli,
  M as streamSimpleGoogleGeminiCli,
};
