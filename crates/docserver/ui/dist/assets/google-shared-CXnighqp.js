import { d as e, s as t } from "./event-stream-BhluV9ep.js";
import { a as n, t as r } from "./transform-messages-B9BoyWEJ.js";
var i = t((e, t) => {
    function n(e, t) {
      (typeof t == `boolean` && (t = { forever: t }),
        (this._originalTimeouts = JSON.parse(JSON.stringify(e))),
        (this._timeouts = e),
        (this._options = t || {}),
        (this._maxRetryTime = (t && t.maxRetryTime) || 1 / 0),
        (this._fn = null),
        (this._errors = []),
        (this._attempts = 1),
        (this._operationTimeout = null),
        (this._operationTimeoutCb = null),
        (this._timeout = null),
        (this._operationStart = null),
        (this._timer = null),
        this._options.forever && (this._cachedTimeouts = this._timeouts.slice(0)));
    }
    ((t.exports = n),
      (n.prototype.reset = function () {
        ((this._attempts = 1), (this._timeouts = this._originalTimeouts.slice(0)));
      }),
      (n.prototype.stop = function () {
        (this._timeout && clearTimeout(this._timeout),
          this._timer && clearTimeout(this._timer),
          (this._timeouts = []),
          (this._cachedTimeouts = null));
      }),
      (n.prototype.retry = function (e) {
        if ((this._timeout && clearTimeout(this._timeout), !e)) return !1;
        var t = new Date().getTime();
        if (e && t - this._operationStart >= this._maxRetryTime)
          return (
            this._errors.push(e), this._errors.unshift(Error(`RetryOperation timeout occurred`)), !1
          );
        this._errors.push(e);
        var n = this._timeouts.shift();
        if (n === void 0)
          if (this._cachedTimeouts)
            (this._errors.splice(0, this._errors.length - 1), (n = this._cachedTimeouts.slice(-1)));
          else return !1;
        var r = this;
        return (
          (this._timer = setTimeout(function () {
            (r._attempts++,
              r._operationTimeoutCb &&
                ((r._timeout = setTimeout(function () {
                  r._operationTimeoutCb(r._attempts);
                }, r._operationTimeout)),
                r._options.unref && r._timeout.unref()),
              r._fn(r._attempts));
          }, n)),
          this._options.unref && this._timer.unref(),
          !0
        );
      }),
      (n.prototype.attempt = function (e, t) {
        ((this._fn = e),
          t &&
            (t.timeout && (this._operationTimeout = t.timeout),
            t.cb && (this._operationTimeoutCb = t.cb)));
        var n = this;
        (this._operationTimeoutCb &&
          (this._timeout = setTimeout(function () {
            n._operationTimeoutCb();
          }, n._operationTimeout)),
          (this._operationStart = new Date().getTime()),
          this._fn(this._attempts));
      }),
      (n.prototype.try = function (e) {
        (console.log(`Using RetryOperation.try() is deprecated`), this.attempt(e));
      }),
      (n.prototype.start = function (e) {
        (console.log(`Using RetryOperation.start() is deprecated`), this.attempt(e));
      }),
      (n.prototype.start = n.prototype.try),
      (n.prototype.errors = function () {
        return this._errors;
      }),
      (n.prototype.attempts = function () {
        return this._attempts;
      }),
      (n.prototype.mainError = function () {
        if (this._errors.length === 0) return null;
        for (var e = {}, t = null, n = 0, r = 0; r < this._errors.length; r++) {
          var i = this._errors[r],
            a = i.message,
            o = (e[a] || 0) + 1;
          ((e[a] = o), o >= n && ((t = i), (n = o)));
        }
        return t;
      }));
  }),
  a = t((e) => {
    var t = i();
    ((e.operation = function (n) {
      return new t(e.timeouts(n), {
        forever: n && (n.forever || n.retries === 1 / 0),
        unref: n && n.unref,
        maxRetryTime: n && n.maxRetryTime,
      });
    }),
      (e.timeouts = function (e) {
        if (e instanceof Array) return [].concat(e);
        var t = { retries: 10, factor: 2, minTimeout: 1 * 1e3, maxTimeout: 1 / 0, randomize: !1 };
        for (var n in e) t[n] = e[n];
        if (t.minTimeout > t.maxTimeout) throw Error(`minTimeout is greater than maxTimeout`);
        for (var r = [], i = 0; i < t.retries; i++) r.push(this.createTimeout(i, t));
        return (
          e && e.forever && !r.length && r.push(this.createTimeout(i, t)),
          r.sort(function (e, t) {
            return e - t;
          }),
          r
        );
      }),
      (e.createTimeout = function (e, t) {
        var n = t.randomize ? Math.random() + 1 : 1,
          r = Math.round(n * Math.max(t.minTimeout, 1) * t.factor ** +e);
        return ((r = Math.min(r, t.maxTimeout)), r);
      }),
      (e.wrap = function (t, n, r) {
        if ((n instanceof Array && ((r = n), (n = null)), !r))
          for (var i in ((r = []), t)) typeof t[i] == `function` && r.push(i);
        for (var a = 0; a < r.length; a++) {
          var o = r[a],
            s = t[o];
          ((t[o] = function (r) {
            var i = e.operation(n),
              a = Array.prototype.slice.call(arguments, 1),
              o = a.pop();
            (a.push(function (e) {
              i.retry(e) || (e && (arguments[0] = i.mainError()), o.apply(this, arguments));
            }),
              i.attempt(function () {
                r.apply(t, a);
              }));
          }.bind(t, s)),
            (t[o].options = n));
        }
      }));
  }),
  o = t((e, t) => {
    t.exports = a();
  }),
  s = e(
    t((e, t) => {
      var n = o(),
        r = [
          `Failed to fetch`,
          `NetworkError when attempting to fetch resource.`,
          `The Internet connection appears to be offline.`,
          `Network request failed`,
        ],
        i = class extends Error {
          constructor(e) {
            (super(),
              e instanceof Error
                ? ((this.originalError = e), ({ message: e } = e))
                : ((this.originalError = Error(e)), (this.originalError.stack = this.stack)),
              (this.name = `AbortError`),
              (this.message = e));
          }
        },
        a = (e, t, n) => {
          let r = n.retries - (t - 1);
          return ((e.attemptNumber = t), (e.retriesLeft = r), e);
        },
        s = (e) => r.includes(e),
        c = (e, t) =>
          new Promise((r, o) => {
            t = { onFailedAttempt: () => {}, retries: 10, ...t };
            let c = n.operation(t);
            c.attempt(async (n) => {
              try {
                r(await e(n));
              } catch (e) {
                if (!(e instanceof Error)) {
                  o(TypeError(`Non-error was thrown: "${e}". You should only throw errors.`));
                  return;
                }
                if (e instanceof i) (c.stop(), o(e.originalError));
                else if (e instanceof TypeError && !s(e.message)) (c.stop(), o(e));
                else {
                  a(e, n, t);
                  try {
                    await t.onFailedAttempt(e);
                  } catch (e) {
                    o(e);
                    return;
                  }
                  c.retry(e) || o(c.mainError());
                }
              }
            });
          });
      ((t.exports = c), (t.exports.default = c), (t.exports.AbortError = i));
    })(),
    1,
  ),
  c = void 0,
  l = void 0;
function u() {
  return { geminiUrl: c, vertexUrl: l };
}
function d(e, t, n, r) {
  if (!e?.baseUrl) {
    let e = u();
    return t ? (e.vertexUrl ?? n) : (e.geminiUrl ?? r);
  }
  return e.baseUrl;
}
var f = class {};
function p(e, t) {
  return e.replace(/\{([^}]+)\}/g, (e, n) => {
    if (Object.prototype.hasOwnProperty.call(t, n)) {
      let e = t[n];
      return e == null ? `` : String(e);
    } else throw Error(`Key '${n}' not found in valueMap.`);
  });
}
function m(e, t, n) {
  for (let r = 0; r < t.length - 1; r++) {
    let i = t[r];
    if (i.endsWith(`[]`)) {
      let a = i.slice(0, -2);
      if (!(a in e))
        if (Array.isArray(n)) e[a] = Array.from({ length: n.length }, () => ({}));
        else throw Error(`Value must be a list given an array path ${i}`);
      if (Array.isArray(e[a])) {
        let i = e[a];
        if (Array.isArray(n))
          for (let e = 0; e < i.length; e++) {
            let a = i[e];
            m(a, t.slice(r + 1), n[e]);
          }
        else for (let e of i) m(e, t.slice(r + 1), n);
      }
      return;
    } else if (i.endsWith(`[0]`)) {
      let a = i.slice(0, -3);
      a in e || (e[a] = [{}]);
      let o = e[a];
      m(o[0], t.slice(r + 1), n);
      return;
    }
    ((!e[i] || typeof e[i] != `object`) && (e[i] = {}), (e = e[i]));
  }
  let r = t[t.length - 1],
    i = e[r];
  if (i !== void 0) {
    if (!n || (typeof n == `object` && Object.keys(n).length === 0) || n === i) return;
    if (typeof i == `object` && typeof n == `object` && i !== null && n !== null)
      Object.assign(i, n);
    else throw Error(`Cannot set value for an existing key. Key: ${r}`);
  } else
    r === `_self` && typeof n == `object` && n && !Array.isArray(n)
      ? Object.assign(e, n)
      : (e[r] = n);
}
function h(e, t, n = void 0) {
  try {
    if (t.length === 1 && t[0] === `_self`) return e;
    for (let r = 0; r < t.length; r++) {
      if (typeof e != `object` || !e) return n;
      let i = t[r];
      if (i.endsWith(`[]`)) {
        let a = i.slice(0, -2);
        if (a in e) {
          let i = e[a];
          return Array.isArray(i) ? i.map((e) => h(e, t.slice(r + 1), n)) : n;
        } else return n;
      } else e = e[i];
    }
    return e;
  } catch (e) {
    if (e instanceof TypeError) return n;
    throw e;
  }
}
function g(e, t) {
  for (let [n, r] of Object.entries(t)) {
    let t = n.split(`.`),
      i = r.split(`.`),
      a = new Set(),
      o = -1;
    for (let e = 0; e < t.length; e++)
      if (t[e] === `*`) {
        o = e;
        break;
      }
    if (o !== -1 && i.length > o)
      for (let e = o; e < i.length; e++) {
        let t = i[e];
        t !== `*` && !t.endsWith(`[]`) && !t.endsWith(`[0]`) && a.add(t);
      }
    _(e, t, i, 0, a);
  }
}
function _(e, t, n, r, i) {
  if (r >= t.length || typeof e != `object` || !e) return;
  let a = t[r];
  if (a.endsWith(`[]`)) {
    let o = a.slice(0, -2),
      s = e;
    if (o in s && Array.isArray(s[o])) for (let e of s[o]) _(e, t, n, r + 1, i);
  } else if (a === `*`) {
    if (typeof e == `object` && e && !Array.isArray(e)) {
      let t = e,
        a = Object.keys(t).filter((e) => !e.startsWith(`_`) && !i.has(e)),
        o = {};
      for (let e of a) o[e] = t[e];
      for (let [e, i] of Object.entries(o)) {
        let a = [];
        for (let t of n.slice(r)) t === `*` ? a.push(e) : a.push(t);
        m(t, a, i);
      }
      for (let e of a) delete t[e];
    }
  } else {
    let o = e;
    a in o && _(o[a], t, n, r + 1, i);
  }
}
function v(e) {
  if (typeof e != `string`) throw Error(`fromImageBytes must be a string`);
  return e;
}
function y(e) {
  let t = {},
    n = h(e, [`operationName`]);
  n != null && m(t, [`operationName`], n);
  let r = h(e, [`resourceName`]);
  return (r != null && m(t, [`_url`, `resourceName`], r), t);
}
function b(e) {
  let t = {},
    n = h(e, [`name`]);
  n != null && m(t, [`name`], n);
  let r = h(e, [`metadata`]);
  r != null && m(t, [`metadata`], r);
  let i = h(e, [`done`]);
  i != null && m(t, [`done`], i);
  let a = h(e, [`error`]);
  a != null && m(t, [`error`], a);
  let o = h(e, [`response`, `generateVideoResponse`]);
  return (o != null && m(t, [`response`], S(o)), t);
}
function x(e) {
  let t = {},
    n = h(e, [`name`]);
  n != null && m(t, [`name`], n);
  let r = h(e, [`metadata`]);
  r != null && m(t, [`metadata`], r);
  let i = h(e, [`done`]);
  i != null && m(t, [`done`], i);
  let a = h(e, [`error`]);
  a != null && m(t, [`error`], a);
  let o = h(e, [`response`]);
  return (o != null && m(t, [`response`], C(o)), t);
}
function S(e) {
  let t = {},
    n = h(e, [`generatedSamples`]);
  if (n != null) {
    let e = n;
    (Array.isArray(e) && (e = e.map((e) => w(e))), m(t, [`generatedVideos`], e));
  }
  let r = h(e, [`raiMediaFilteredCount`]);
  r != null && m(t, [`raiMediaFilteredCount`], r);
  let i = h(e, [`raiMediaFilteredReasons`]);
  return (i != null && m(t, [`raiMediaFilteredReasons`], i), t);
}
function C(e) {
  let t = {},
    n = h(e, [`videos`]);
  if (n != null) {
    let e = n;
    (Array.isArray(e) && (e = e.map((e) => T(e))), m(t, [`generatedVideos`], e));
  }
  let r = h(e, [`raiMediaFilteredCount`]);
  r != null && m(t, [`raiMediaFilteredCount`], r);
  let i = h(e, [`raiMediaFilteredReasons`]);
  return (i != null && m(t, [`raiMediaFilteredReasons`], i), t);
}
function w(e) {
  let t = {},
    n = h(e, [`video`]);
  return (n != null && m(t, [`video`], ee(n)), t);
}
function T(e) {
  let t = {},
    n = h(e, [`_self`]);
  return (n != null && m(t, [`video`], te(n)), t);
}
function E(e) {
  let t = {},
    n = h(e, [`operationName`]);
  return (n != null && m(t, [`_url`, `operationName`], n), t);
}
function D(e) {
  let t = {},
    n = h(e, [`operationName`]);
  return (n != null && m(t, [`_url`, `operationName`], n), t);
}
function O(e) {
  let t = {},
    n = h(e, [`name`]);
  n != null && m(t, [`name`], n);
  let r = h(e, [`metadata`]);
  r != null && m(t, [`metadata`], r);
  let i = h(e, [`done`]);
  i != null && m(t, [`done`], i);
  let a = h(e, [`error`]);
  a != null && m(t, [`error`], a);
  let o = h(e, [`response`]);
  return (o != null && m(t, [`response`], k(o)), t);
}
function k(e) {
  let t = {},
    n = h(e, [`sdkHttpResponse`]);
  n != null && m(t, [`sdkHttpResponse`], n);
  let r = h(e, [`parent`]);
  r != null && m(t, [`parent`], r);
  let i = h(e, [`documentName`]);
  return (i != null && m(t, [`documentName`], i), t);
}
function A(e) {
  let t = {},
    n = h(e, [`name`]);
  n != null && m(t, [`name`], n);
  let r = h(e, [`metadata`]);
  r != null && m(t, [`metadata`], r);
  let i = h(e, [`done`]);
  i != null && m(t, [`done`], i);
  let a = h(e, [`error`]);
  a != null && m(t, [`error`], a);
  let o = h(e, [`response`]);
  return (o != null && m(t, [`response`], j(o)), t);
}
function j(e) {
  let t = {},
    n = h(e, [`sdkHttpResponse`]);
  n != null && m(t, [`sdkHttpResponse`], n);
  let r = h(e, [`parent`]);
  r != null && m(t, [`parent`], r);
  let i = h(e, [`documentName`]);
  return (i != null && m(t, [`documentName`], i), t);
}
function ee(e) {
  let t = {},
    n = h(e, [`uri`]);
  n != null && m(t, [`uri`], n);
  let r = h(e, [`encodedVideo`]);
  r != null && m(t, [`videoBytes`], v(r));
  let i = h(e, [`encoding`]);
  return (i != null && m(t, [`mimeType`], i), t);
}
function te(e) {
  let t = {},
    n = h(e, [`gcsUri`]);
  n != null && m(t, [`uri`], n);
  let r = h(e, [`bytesBase64Encoded`]);
  r != null && m(t, [`videoBytes`], v(r));
  let i = h(e, [`mimeType`]);
  return (i != null && m(t, [`mimeType`], i), t);
}
var ne;
(function (e) {
  ((e.LANGUAGE_UNSPECIFIED = `LANGUAGE_UNSPECIFIED`), (e.PYTHON = `PYTHON`));
})((ne ||= {}));
var re;
(function (e) {
  ((e.OUTCOME_UNSPECIFIED = `OUTCOME_UNSPECIFIED`),
    (e.OUTCOME_OK = `OUTCOME_OK`),
    (e.OUTCOME_FAILED = `OUTCOME_FAILED`),
    (e.OUTCOME_DEADLINE_EXCEEDED = `OUTCOME_DEADLINE_EXCEEDED`));
})((re ||= {}));
var ie;
(function (e) {
  ((e.SCHEDULING_UNSPECIFIED = `SCHEDULING_UNSPECIFIED`),
    (e.SILENT = `SILENT`),
    (e.WHEN_IDLE = `WHEN_IDLE`),
    (e.INTERRUPT = `INTERRUPT`));
})((ie ||= {}));
var M;
(function (e) {
  ((e.TYPE_UNSPECIFIED = `TYPE_UNSPECIFIED`),
    (e.STRING = `STRING`),
    (e.NUMBER = `NUMBER`),
    (e.INTEGER = `INTEGER`),
    (e.BOOLEAN = `BOOLEAN`),
    (e.ARRAY = `ARRAY`),
    (e.OBJECT = `OBJECT`),
    (e.NULL = `NULL`));
})((M ||= {}));
var ae;
(function (e) {
  ((e.PHISH_BLOCK_THRESHOLD_UNSPECIFIED = `PHISH_BLOCK_THRESHOLD_UNSPECIFIED`),
    (e.BLOCK_LOW_AND_ABOVE = `BLOCK_LOW_AND_ABOVE`),
    (e.BLOCK_MEDIUM_AND_ABOVE = `BLOCK_MEDIUM_AND_ABOVE`),
    (e.BLOCK_HIGH_AND_ABOVE = `BLOCK_HIGH_AND_ABOVE`),
    (e.BLOCK_HIGHER_AND_ABOVE = `BLOCK_HIGHER_AND_ABOVE`),
    (e.BLOCK_VERY_HIGH_AND_ABOVE = `BLOCK_VERY_HIGH_AND_ABOVE`),
    (e.BLOCK_ONLY_EXTREMELY_HIGH = `BLOCK_ONLY_EXTREMELY_HIGH`));
})((ae ||= {}));
var oe;
(function (e) {
  ((e.AUTH_TYPE_UNSPECIFIED = `AUTH_TYPE_UNSPECIFIED`),
    (e.NO_AUTH = `NO_AUTH`),
    (e.API_KEY_AUTH = `API_KEY_AUTH`),
    (e.HTTP_BASIC_AUTH = `HTTP_BASIC_AUTH`),
    (e.GOOGLE_SERVICE_ACCOUNT_AUTH = `GOOGLE_SERVICE_ACCOUNT_AUTH`),
    (e.OAUTH = `OAUTH`),
    (e.OIDC_AUTH = `OIDC_AUTH`));
})((oe ||= {}));
var se;
(function (e) {
  ((e.HTTP_IN_UNSPECIFIED = `HTTP_IN_UNSPECIFIED`),
    (e.HTTP_IN_QUERY = `HTTP_IN_QUERY`),
    (e.HTTP_IN_HEADER = `HTTP_IN_HEADER`),
    (e.HTTP_IN_PATH = `HTTP_IN_PATH`),
    (e.HTTP_IN_BODY = `HTTP_IN_BODY`),
    (e.HTTP_IN_COOKIE = `HTTP_IN_COOKIE`));
})((se ||= {}));
var ce;
(function (e) {
  ((e.API_SPEC_UNSPECIFIED = `API_SPEC_UNSPECIFIED`),
    (e.SIMPLE_SEARCH = `SIMPLE_SEARCH`),
    (e.ELASTIC_SEARCH = `ELASTIC_SEARCH`));
})((ce ||= {}));
var le;
(function (e) {
  ((e.UNSPECIFIED = `UNSPECIFIED`), (e.BLOCKING = `BLOCKING`), (e.NON_BLOCKING = `NON_BLOCKING`));
})((le ||= {}));
var ue;
(function (e) {
  ((e.MODE_UNSPECIFIED = `MODE_UNSPECIFIED`), (e.MODE_DYNAMIC = `MODE_DYNAMIC`));
})((ue ||= {}));
var de;
(function (e) {
  ((e.MODE_UNSPECIFIED = `MODE_UNSPECIFIED`),
    (e.AUTO = `AUTO`),
    (e.ANY = `ANY`),
    (e.NONE = `NONE`),
    (e.VALIDATED = `VALIDATED`));
})((de ||= {}));
var fe;
(function (e) {
  ((e.THINKING_LEVEL_UNSPECIFIED = `THINKING_LEVEL_UNSPECIFIED`),
    (e.LOW = `LOW`),
    (e.MEDIUM = `MEDIUM`),
    (e.HIGH = `HIGH`),
    (e.MINIMAL = `MINIMAL`));
})((fe ||= {}));
var pe;
(function (e) {
  ((e.DONT_ALLOW = `DONT_ALLOW`), (e.ALLOW_ADULT = `ALLOW_ADULT`), (e.ALLOW_ALL = `ALLOW_ALL`));
})((pe ||= {}));
var me;
(function (e) {
  ((e.HARM_CATEGORY_UNSPECIFIED = `HARM_CATEGORY_UNSPECIFIED`),
    (e.HARM_CATEGORY_HARASSMENT = `HARM_CATEGORY_HARASSMENT`),
    (e.HARM_CATEGORY_HATE_SPEECH = `HARM_CATEGORY_HATE_SPEECH`),
    (e.HARM_CATEGORY_SEXUALLY_EXPLICIT = `HARM_CATEGORY_SEXUALLY_EXPLICIT`),
    (e.HARM_CATEGORY_DANGEROUS_CONTENT = `HARM_CATEGORY_DANGEROUS_CONTENT`),
    (e.HARM_CATEGORY_CIVIC_INTEGRITY = `HARM_CATEGORY_CIVIC_INTEGRITY`),
    (e.HARM_CATEGORY_IMAGE_HATE = `HARM_CATEGORY_IMAGE_HATE`),
    (e.HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT = `HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT`),
    (e.HARM_CATEGORY_IMAGE_HARASSMENT = `HARM_CATEGORY_IMAGE_HARASSMENT`),
    (e.HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT = `HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT`),
    (e.HARM_CATEGORY_JAILBREAK = `HARM_CATEGORY_JAILBREAK`));
})((me ||= {}));
var he;
(function (e) {
  ((e.HARM_BLOCK_METHOD_UNSPECIFIED = `HARM_BLOCK_METHOD_UNSPECIFIED`),
    (e.SEVERITY = `SEVERITY`),
    (e.PROBABILITY = `PROBABILITY`));
})((he ||= {}));
var ge;
(function (e) {
  ((e.HARM_BLOCK_THRESHOLD_UNSPECIFIED = `HARM_BLOCK_THRESHOLD_UNSPECIFIED`),
    (e.BLOCK_LOW_AND_ABOVE = `BLOCK_LOW_AND_ABOVE`),
    (e.BLOCK_MEDIUM_AND_ABOVE = `BLOCK_MEDIUM_AND_ABOVE`),
    (e.BLOCK_ONLY_HIGH = `BLOCK_ONLY_HIGH`),
    (e.BLOCK_NONE = `BLOCK_NONE`),
    (e.OFF = `OFF`));
})((ge ||= {}));
var N;
(function (e) {
  ((e.FINISH_REASON_UNSPECIFIED = `FINISH_REASON_UNSPECIFIED`),
    (e.STOP = `STOP`),
    (e.MAX_TOKENS = `MAX_TOKENS`),
    (e.SAFETY = `SAFETY`),
    (e.RECITATION = `RECITATION`),
    (e.LANGUAGE = `LANGUAGE`),
    (e.OTHER = `OTHER`),
    (e.BLOCKLIST = `BLOCKLIST`),
    (e.PROHIBITED_CONTENT = `PROHIBITED_CONTENT`),
    (e.SPII = `SPII`),
    (e.MALFORMED_FUNCTION_CALL = `MALFORMED_FUNCTION_CALL`),
    (e.IMAGE_SAFETY = `IMAGE_SAFETY`),
    (e.UNEXPECTED_TOOL_CALL = `UNEXPECTED_TOOL_CALL`),
    (e.IMAGE_PROHIBITED_CONTENT = `IMAGE_PROHIBITED_CONTENT`),
    (e.NO_IMAGE = `NO_IMAGE`),
    (e.IMAGE_RECITATION = `IMAGE_RECITATION`),
    (e.IMAGE_OTHER = `IMAGE_OTHER`));
})((N ||= {}));
var _e;
(function (e) {
  ((e.HARM_PROBABILITY_UNSPECIFIED = `HARM_PROBABILITY_UNSPECIFIED`),
    (e.NEGLIGIBLE = `NEGLIGIBLE`),
    (e.LOW = `LOW`),
    (e.MEDIUM = `MEDIUM`),
    (e.HIGH = `HIGH`));
})((_e ||= {}));
var ve;
(function (e) {
  ((e.HARM_SEVERITY_UNSPECIFIED = `HARM_SEVERITY_UNSPECIFIED`),
    (e.HARM_SEVERITY_NEGLIGIBLE = `HARM_SEVERITY_NEGLIGIBLE`),
    (e.HARM_SEVERITY_LOW = `HARM_SEVERITY_LOW`),
    (e.HARM_SEVERITY_MEDIUM = `HARM_SEVERITY_MEDIUM`),
    (e.HARM_SEVERITY_HIGH = `HARM_SEVERITY_HIGH`));
})((ve ||= {}));
var ye;
(function (e) {
  ((e.URL_RETRIEVAL_STATUS_UNSPECIFIED = `URL_RETRIEVAL_STATUS_UNSPECIFIED`),
    (e.URL_RETRIEVAL_STATUS_SUCCESS = `URL_RETRIEVAL_STATUS_SUCCESS`),
    (e.URL_RETRIEVAL_STATUS_ERROR = `URL_RETRIEVAL_STATUS_ERROR`),
    (e.URL_RETRIEVAL_STATUS_PAYWALL = `URL_RETRIEVAL_STATUS_PAYWALL`),
    (e.URL_RETRIEVAL_STATUS_UNSAFE = `URL_RETRIEVAL_STATUS_UNSAFE`));
})((ye ||= {}));
var be;
(function (e) {
  ((e.BLOCKED_REASON_UNSPECIFIED = `BLOCKED_REASON_UNSPECIFIED`),
    (e.SAFETY = `SAFETY`),
    (e.OTHER = `OTHER`),
    (e.BLOCKLIST = `BLOCKLIST`),
    (e.PROHIBITED_CONTENT = `PROHIBITED_CONTENT`),
    (e.IMAGE_SAFETY = `IMAGE_SAFETY`),
    (e.MODEL_ARMOR = `MODEL_ARMOR`),
    (e.JAILBREAK = `JAILBREAK`));
})((be ||= {}));
var xe;
(function (e) {
  ((e.TRAFFIC_TYPE_UNSPECIFIED = `TRAFFIC_TYPE_UNSPECIFIED`),
    (e.ON_DEMAND = `ON_DEMAND`),
    (e.ON_DEMAND_PRIORITY = `ON_DEMAND_PRIORITY`),
    (e.ON_DEMAND_FLEX = `ON_DEMAND_FLEX`),
    (e.PROVISIONED_THROUGHPUT = `PROVISIONED_THROUGHPUT`));
})((xe ||= {}));
var Se;
(function (e) {
  ((e.MODALITY_UNSPECIFIED = `MODALITY_UNSPECIFIED`),
    (e.TEXT = `TEXT`),
    (e.IMAGE = `IMAGE`),
    (e.AUDIO = `AUDIO`));
})((Se ||= {}));
var Ce;
(function (e) {
  ((e.MEDIA_RESOLUTION_UNSPECIFIED = `MEDIA_RESOLUTION_UNSPECIFIED`),
    (e.MEDIA_RESOLUTION_LOW = `MEDIA_RESOLUTION_LOW`),
    (e.MEDIA_RESOLUTION_MEDIUM = `MEDIA_RESOLUTION_MEDIUM`),
    (e.MEDIA_RESOLUTION_HIGH = `MEDIA_RESOLUTION_HIGH`));
})((Ce ||= {}));
var we;
(function (e) {
  ((e.TUNING_MODE_UNSPECIFIED = `TUNING_MODE_UNSPECIFIED`),
    (e.TUNING_MODE_FULL = `TUNING_MODE_FULL`),
    (e.TUNING_MODE_PEFT_ADAPTER = `TUNING_MODE_PEFT_ADAPTER`));
})((we ||= {}));
var Te;
(function (e) {
  ((e.ADAPTER_SIZE_UNSPECIFIED = `ADAPTER_SIZE_UNSPECIFIED`),
    (e.ADAPTER_SIZE_ONE = `ADAPTER_SIZE_ONE`),
    (e.ADAPTER_SIZE_TWO = `ADAPTER_SIZE_TWO`),
    (e.ADAPTER_SIZE_FOUR = `ADAPTER_SIZE_FOUR`),
    (e.ADAPTER_SIZE_EIGHT = `ADAPTER_SIZE_EIGHT`),
    (e.ADAPTER_SIZE_SIXTEEN = `ADAPTER_SIZE_SIXTEEN`),
    (e.ADAPTER_SIZE_THIRTY_TWO = `ADAPTER_SIZE_THIRTY_TWO`));
})((Te ||= {}));
var Ee;
(function (e) {
  ((e.JOB_STATE_UNSPECIFIED = `JOB_STATE_UNSPECIFIED`),
    (e.JOB_STATE_QUEUED = `JOB_STATE_QUEUED`),
    (e.JOB_STATE_PENDING = `JOB_STATE_PENDING`),
    (e.JOB_STATE_RUNNING = `JOB_STATE_RUNNING`),
    (e.JOB_STATE_SUCCEEDED = `JOB_STATE_SUCCEEDED`),
    (e.JOB_STATE_FAILED = `JOB_STATE_FAILED`),
    (e.JOB_STATE_CANCELLING = `JOB_STATE_CANCELLING`),
    (e.JOB_STATE_CANCELLED = `JOB_STATE_CANCELLED`),
    (e.JOB_STATE_PAUSED = `JOB_STATE_PAUSED`),
    (e.JOB_STATE_EXPIRED = `JOB_STATE_EXPIRED`),
    (e.JOB_STATE_UPDATING = `JOB_STATE_UPDATING`),
    (e.JOB_STATE_PARTIALLY_SUCCEEDED = `JOB_STATE_PARTIALLY_SUCCEEDED`));
})((Ee ||= {}));
var De;
(function (e) {
  ((e.TUNING_JOB_STATE_UNSPECIFIED = `TUNING_JOB_STATE_UNSPECIFIED`),
    (e.TUNING_JOB_STATE_WAITING_FOR_QUOTA = `TUNING_JOB_STATE_WAITING_FOR_QUOTA`),
    (e.TUNING_JOB_STATE_PROCESSING_DATASET = `TUNING_JOB_STATE_PROCESSING_DATASET`),
    (e.TUNING_JOB_STATE_WAITING_FOR_CAPACITY = `TUNING_JOB_STATE_WAITING_FOR_CAPACITY`),
    (e.TUNING_JOB_STATE_TUNING = `TUNING_JOB_STATE_TUNING`),
    (e.TUNING_JOB_STATE_POST_PROCESSING = `TUNING_JOB_STATE_POST_PROCESSING`));
})((De ||= {}));
var Oe;
(function (e) {
  ((e.AGGREGATION_METRIC_UNSPECIFIED = `AGGREGATION_METRIC_UNSPECIFIED`),
    (e.AVERAGE = `AVERAGE`),
    (e.MODE = `MODE`),
    (e.STANDARD_DEVIATION = `STANDARD_DEVIATION`),
    (e.VARIANCE = `VARIANCE`),
    (e.MINIMUM = `MINIMUM`),
    (e.MAXIMUM = `MAXIMUM`),
    (e.MEDIAN = `MEDIAN`),
    (e.PERCENTILE_P90 = `PERCENTILE_P90`),
    (e.PERCENTILE_P95 = `PERCENTILE_P95`),
    (e.PERCENTILE_P99 = `PERCENTILE_P99`));
})((Oe ||= {}));
var ke;
(function (e) {
  ((e.PAIRWISE_CHOICE_UNSPECIFIED = `PAIRWISE_CHOICE_UNSPECIFIED`),
    (e.BASELINE = `BASELINE`),
    (e.CANDIDATE = `CANDIDATE`),
    (e.TIE = `TIE`));
})((ke ||= {}));
var Ae;
(function (e) {
  ((e.TUNING_TASK_UNSPECIFIED = `TUNING_TASK_UNSPECIFIED`),
    (e.TUNING_TASK_I2V = `TUNING_TASK_I2V`),
    (e.TUNING_TASK_T2V = `TUNING_TASK_T2V`),
    (e.TUNING_TASK_R2V = `TUNING_TASK_R2V`));
})((Ae ||= {}));
var je;
(function (e) {
  ((e.MEDIA_RESOLUTION_UNSPECIFIED = `MEDIA_RESOLUTION_UNSPECIFIED`),
    (e.MEDIA_RESOLUTION_LOW = `MEDIA_RESOLUTION_LOW`),
    (e.MEDIA_RESOLUTION_MEDIUM = `MEDIA_RESOLUTION_MEDIUM`),
    (e.MEDIA_RESOLUTION_HIGH = `MEDIA_RESOLUTION_HIGH`),
    (e.MEDIA_RESOLUTION_ULTRA_HIGH = `MEDIA_RESOLUTION_ULTRA_HIGH`));
})((je ||= {}));
var Me;
(function (e) {
  ((e.TOOL_TYPE_UNSPECIFIED = `TOOL_TYPE_UNSPECIFIED`),
    (e.GOOGLE_SEARCH_WEB = `GOOGLE_SEARCH_WEB`),
    (e.GOOGLE_SEARCH_IMAGE = `GOOGLE_SEARCH_IMAGE`),
    (e.URL_CONTEXT = `URL_CONTEXT`),
    (e.GOOGLE_MAPS = `GOOGLE_MAPS`),
    (e.FILE_SEARCH = `FILE_SEARCH`));
})((Me ||= {}));
var Ne;
(function (e) {
  e.COLLECTION = `COLLECTION`;
})((Ne ||= {}));
var Pe;
(function (e) {
  ((e.FEATURE_SELECTION_PREFERENCE_UNSPECIFIED = `FEATURE_SELECTION_PREFERENCE_UNSPECIFIED`),
    (e.PRIORITIZE_QUALITY = `PRIORITIZE_QUALITY`),
    (e.BALANCED = `BALANCED`),
    (e.PRIORITIZE_COST = `PRIORITIZE_COST`));
})((Pe ||= {}));
var Fe;
(function (e) {
  ((e.ENVIRONMENT_UNSPECIFIED = `ENVIRONMENT_UNSPECIFIED`),
    (e.ENVIRONMENT_BROWSER = `ENVIRONMENT_BROWSER`));
})((Fe ||= {}));
var Ie;
(function (e) {
  ((e.PROMINENT_PEOPLE_UNSPECIFIED = `PROMINENT_PEOPLE_UNSPECIFIED`),
    (e.ALLOW_PROMINENT_PEOPLE = `ALLOW_PROMINENT_PEOPLE`),
    (e.BLOCK_PROMINENT_PEOPLE = `BLOCK_PROMINENT_PEOPLE`));
})((Ie ||= {}));
var Le;
(function (e) {
  ((e.PREDICT = `PREDICT`), (e.EMBED_CONTENT = `EMBED_CONTENT`));
})((Le ||= {}));
var Re;
(function (e) {
  ((e.BLOCK_LOW_AND_ABOVE = `BLOCK_LOW_AND_ABOVE`),
    (e.BLOCK_MEDIUM_AND_ABOVE = `BLOCK_MEDIUM_AND_ABOVE`),
    (e.BLOCK_ONLY_HIGH = `BLOCK_ONLY_HIGH`),
    (e.BLOCK_NONE = `BLOCK_NONE`));
})((Re ||= {}));
var ze;
(function (e) {
  ((e.auto = `auto`),
    (e.en = `en`),
    (e.ja = `ja`),
    (e.ko = `ko`),
    (e.hi = `hi`),
    (e.zh = `zh`),
    (e.pt = `pt`),
    (e.es = `es`));
})((ze ||= {}));
var Be;
(function (e) {
  ((e.MASK_MODE_DEFAULT = `MASK_MODE_DEFAULT`),
    (e.MASK_MODE_USER_PROVIDED = `MASK_MODE_USER_PROVIDED`),
    (e.MASK_MODE_BACKGROUND = `MASK_MODE_BACKGROUND`),
    (e.MASK_MODE_FOREGROUND = `MASK_MODE_FOREGROUND`),
    (e.MASK_MODE_SEMANTIC = `MASK_MODE_SEMANTIC`));
})((Be ||= {}));
var Ve;
(function (e) {
  ((e.CONTROL_TYPE_DEFAULT = `CONTROL_TYPE_DEFAULT`),
    (e.CONTROL_TYPE_CANNY = `CONTROL_TYPE_CANNY`),
    (e.CONTROL_TYPE_SCRIBBLE = `CONTROL_TYPE_SCRIBBLE`),
    (e.CONTROL_TYPE_FACE_MESH = `CONTROL_TYPE_FACE_MESH`));
})((Ve ||= {}));
var He;
(function (e) {
  ((e.SUBJECT_TYPE_DEFAULT = `SUBJECT_TYPE_DEFAULT`),
    (e.SUBJECT_TYPE_PERSON = `SUBJECT_TYPE_PERSON`),
    (e.SUBJECT_TYPE_ANIMAL = `SUBJECT_TYPE_ANIMAL`),
    (e.SUBJECT_TYPE_PRODUCT = `SUBJECT_TYPE_PRODUCT`));
})((He ||= {}));
var Ue;
(function (e) {
  ((e.EDIT_MODE_DEFAULT = `EDIT_MODE_DEFAULT`),
    (e.EDIT_MODE_INPAINT_REMOVAL = `EDIT_MODE_INPAINT_REMOVAL`),
    (e.EDIT_MODE_INPAINT_INSERTION = `EDIT_MODE_INPAINT_INSERTION`),
    (e.EDIT_MODE_OUTPAINT = `EDIT_MODE_OUTPAINT`),
    (e.EDIT_MODE_CONTROLLED_EDITING = `EDIT_MODE_CONTROLLED_EDITING`),
    (e.EDIT_MODE_STYLE = `EDIT_MODE_STYLE`),
    (e.EDIT_MODE_BGSWAP = `EDIT_MODE_BGSWAP`),
    (e.EDIT_MODE_PRODUCT_IMAGE = `EDIT_MODE_PRODUCT_IMAGE`));
})((Ue ||= {}));
var We;
(function (e) {
  ((e.FOREGROUND = `FOREGROUND`),
    (e.BACKGROUND = `BACKGROUND`),
    (e.PROMPT = `PROMPT`),
    (e.SEMANTIC = `SEMANTIC`),
    (e.INTERACTIVE = `INTERACTIVE`));
})((We ||= {}));
var Ge;
(function (e) {
  ((e.ASSET = `ASSET`), (e.STYLE = `STYLE`));
})((Ge ||= {}));
var Ke;
(function (e) {
  ((e.INSERT = `INSERT`),
    (e.REMOVE = `REMOVE`),
    (e.REMOVE_STATIC = `REMOVE_STATIC`),
    (e.OUTPAINT = `OUTPAINT`));
})((Ke ||= {}));
var qe;
(function (e) {
  ((e.OPTIMIZED = `OPTIMIZED`), (e.LOSSLESS = `LOSSLESS`));
})((qe ||= {}));
var Je;
(function (e) {
  ((e.SUPERVISED_FINE_TUNING = `SUPERVISED_FINE_TUNING`),
    (e.PREFERENCE_TUNING = `PREFERENCE_TUNING`),
    (e.DISTILLATION = `DISTILLATION`));
})((Je ||= {}));
var Ye;
(function (e) {
  ((e.STATE_UNSPECIFIED = `STATE_UNSPECIFIED`),
    (e.STATE_PENDING = `STATE_PENDING`),
    (e.STATE_ACTIVE = `STATE_ACTIVE`),
    (e.STATE_FAILED = `STATE_FAILED`));
})((Ye ||= {}));
var Xe;
(function (e) {
  ((e.STATE_UNSPECIFIED = `STATE_UNSPECIFIED`),
    (e.PROCESSING = `PROCESSING`),
    (e.ACTIVE = `ACTIVE`),
    (e.FAILED = `FAILED`));
})((Xe ||= {}));
var Ze;
(function (e) {
  ((e.SOURCE_UNSPECIFIED = `SOURCE_UNSPECIFIED`),
    (e.UPLOADED = `UPLOADED`),
    (e.GENERATED = `GENERATED`),
    (e.REGISTERED = `REGISTERED`));
})((Ze ||= {}));
var Qe;
(function (e) {
  ((e.TURN_COMPLETE_REASON_UNSPECIFIED = `TURN_COMPLETE_REASON_UNSPECIFIED`),
    (e.MALFORMED_FUNCTION_CALL = `MALFORMED_FUNCTION_CALL`),
    (e.RESPONSE_REJECTED = `RESPONSE_REJECTED`),
    (e.NEED_MORE_INPUT = `NEED_MORE_INPUT`));
})((Qe ||= {}));
var $e;
(function (e) {
  ((e.MODALITY_UNSPECIFIED = `MODALITY_UNSPECIFIED`),
    (e.TEXT = `TEXT`),
    (e.IMAGE = `IMAGE`),
    (e.VIDEO = `VIDEO`),
    (e.AUDIO = `AUDIO`),
    (e.DOCUMENT = `DOCUMENT`));
})(($e ||= {}));
var et;
(function (e) {
  ((e.VAD_SIGNAL_TYPE_UNSPECIFIED = `VAD_SIGNAL_TYPE_UNSPECIFIED`),
    (e.VAD_SIGNAL_TYPE_SOS = `VAD_SIGNAL_TYPE_SOS`),
    (e.VAD_SIGNAL_TYPE_EOS = `VAD_SIGNAL_TYPE_EOS`));
})((et ||= {}));
var tt;
(function (e) {
  ((e.TYPE_UNSPECIFIED = `TYPE_UNSPECIFIED`),
    (e.ACTIVITY_START = `ACTIVITY_START`),
    (e.ACTIVITY_END = `ACTIVITY_END`));
})((tt ||= {}));
var nt;
(function (e) {
  ((e.START_SENSITIVITY_UNSPECIFIED = `START_SENSITIVITY_UNSPECIFIED`),
    (e.START_SENSITIVITY_HIGH = `START_SENSITIVITY_HIGH`),
    (e.START_SENSITIVITY_LOW = `START_SENSITIVITY_LOW`));
})((nt ||= {}));
var rt;
(function (e) {
  ((e.END_SENSITIVITY_UNSPECIFIED = `END_SENSITIVITY_UNSPECIFIED`),
    (e.END_SENSITIVITY_HIGH = `END_SENSITIVITY_HIGH`),
    (e.END_SENSITIVITY_LOW = `END_SENSITIVITY_LOW`));
})((rt ||= {}));
var it;
(function (e) {
  ((e.ACTIVITY_HANDLING_UNSPECIFIED = `ACTIVITY_HANDLING_UNSPECIFIED`),
    (e.START_OF_ACTIVITY_INTERRUPTS = `START_OF_ACTIVITY_INTERRUPTS`),
    (e.NO_INTERRUPTION = `NO_INTERRUPTION`));
})((it ||= {}));
var at;
(function (e) {
  ((e.TURN_COVERAGE_UNSPECIFIED = `TURN_COVERAGE_UNSPECIFIED`),
    (e.TURN_INCLUDES_ONLY_ACTIVITY = `TURN_INCLUDES_ONLY_ACTIVITY`),
    (e.TURN_INCLUDES_ALL_INPUT = `TURN_INCLUDES_ALL_INPUT`));
})((at ||= {}));
var ot;
(function (e) {
  ((e.SCALE_UNSPECIFIED = `SCALE_UNSPECIFIED`),
    (e.C_MAJOR_A_MINOR = `C_MAJOR_A_MINOR`),
    (e.D_FLAT_MAJOR_B_FLAT_MINOR = `D_FLAT_MAJOR_B_FLAT_MINOR`),
    (e.D_MAJOR_B_MINOR = `D_MAJOR_B_MINOR`),
    (e.E_FLAT_MAJOR_C_MINOR = `E_FLAT_MAJOR_C_MINOR`),
    (e.E_MAJOR_D_FLAT_MINOR = `E_MAJOR_D_FLAT_MINOR`),
    (e.F_MAJOR_D_MINOR = `F_MAJOR_D_MINOR`),
    (e.G_FLAT_MAJOR_E_FLAT_MINOR = `G_FLAT_MAJOR_E_FLAT_MINOR`),
    (e.G_MAJOR_E_MINOR = `G_MAJOR_E_MINOR`),
    (e.A_FLAT_MAJOR_F_MINOR = `A_FLAT_MAJOR_F_MINOR`),
    (e.A_MAJOR_G_FLAT_MINOR = `A_MAJOR_G_FLAT_MINOR`),
    (e.B_FLAT_MAJOR_G_MINOR = `B_FLAT_MAJOR_G_MINOR`),
    (e.B_MAJOR_A_FLAT_MINOR = `B_MAJOR_A_FLAT_MINOR`));
})((ot ||= {}));
var st;
(function (e) {
  ((e.MUSIC_GENERATION_MODE_UNSPECIFIED = `MUSIC_GENERATION_MODE_UNSPECIFIED`),
    (e.QUALITY = `QUALITY`),
    (e.DIVERSITY = `DIVERSITY`),
    (e.VOCALIZATION = `VOCALIZATION`));
})((st ||= {}));
var ct;
(function (e) {
  ((e.PLAYBACK_CONTROL_UNSPECIFIED = `PLAYBACK_CONTROL_UNSPECIFIED`),
    (e.PLAY = `PLAY`),
    (e.PAUSE = `PAUSE`),
    (e.STOP = `STOP`),
    (e.RESET_CONTEXT = `RESET_CONTEXT`));
})((ct ||= {}));
var lt = class {
    constructor(e) {
      let t = {};
      for (let n of e.headers.entries()) t[n[0]] = n[1];
      ((this.headers = t), (this.responseInternal = e));
    }
    json() {
      return this.responseInternal.json();
    }
  },
  ut = class {
    get text() {
      if (this.candidates?.[0]?.content?.parts?.length === 0) return;
      this.candidates &&
        this.candidates.length > 1 &&
        console.warn(
          `there are multiple candidates in the response, returning text from the first one.`,
        );
      let e = ``,
        t = !1,
        n = [];
      for (let r of this.candidates?.[0]?.content?.parts ?? []) {
        for (let [e, t] of Object.entries(r))
          e !== `text` &&
            e !== `thought` &&
            e !== `thoughtSignature` &&
            (t !== null || t !== void 0) &&
            n.push(e);
        if (typeof r.text == `string`) {
          if (typeof r.thought == `boolean` && r.thought) continue;
          ((t = !0), (e += r.text));
        }
      }
      return (
        n.length > 0 &&
          console.warn(
            `there are non-text parts ${n} in the response, returning concatenation of all text parts. Please refer to the non text parts for a full response from model.`,
          ),
        t ? e : void 0
      );
    }
    get data() {
      if (this.candidates?.[0]?.content?.parts?.length === 0) return;
      this.candidates &&
        this.candidates.length > 1 &&
        console.warn(
          `there are multiple candidates in the response, returning data from the first one.`,
        );
      let e = ``,
        t = [];
      for (let n of this.candidates?.[0]?.content?.parts ?? []) {
        for (let [e, r] of Object.entries(n))
          e !== `inlineData` && (r !== null || r !== void 0) && t.push(e);
        n.inlineData && typeof n.inlineData.data == `string` && (e += atob(n.inlineData.data));
      }
      return (
        t.length > 0 &&
          console.warn(
            `there are non-data parts ${t} in the response, returning concatenation of all data parts. Please refer to the non data parts for a full response from model.`,
          ),
        e.length > 0 ? btoa(e) : void 0
      );
    }
    get functionCalls() {
      if (this.candidates?.[0]?.content?.parts?.length === 0) return;
      this.candidates &&
        this.candidates.length > 1 &&
        console.warn(
          `there are multiple candidates in the response, returning function calls from the first one.`,
        );
      let e = this.candidates?.[0]?.content?.parts
        ?.filter((e) => e.functionCall)
        .map((e) => e.functionCall)
        .filter((e) => e !== void 0);
      if (e?.length !== 0) return e;
    }
    get executableCode() {
      if (this.candidates?.[0]?.content?.parts?.length === 0) return;
      this.candidates &&
        this.candidates.length > 1 &&
        console.warn(
          `there are multiple candidates in the response, returning executable code from the first one.`,
        );
      let e = this.candidates?.[0]?.content?.parts
        ?.filter((e) => e.executableCode)
        .map((e) => e.executableCode)
        .filter((e) => e !== void 0);
      if (e?.length !== 0) return e?.[0]?.code;
    }
    get codeExecutionResult() {
      if (this.candidates?.[0]?.content?.parts?.length === 0) return;
      this.candidates &&
        this.candidates.length > 1 &&
        console.warn(
          `there are multiple candidates in the response, returning code execution result from the first one.`,
        );
      let e = this.candidates?.[0]?.content?.parts
        ?.filter((e) => e.codeExecutionResult)
        .map((e) => e.codeExecutionResult)
        .filter((e) => e !== void 0);
      if (e?.length !== 0) return e?.[0]?.output;
    }
  },
  dt = class {},
  ft = class {},
  pt = class {},
  mt = class {},
  ht = class {},
  gt = class {},
  _t = class {},
  vt = class {},
  yt = class {},
  bt = class {},
  xt = class e {
    _fromAPIResponse({ apiResponse: t, _isVertexAI: n }) {
      let r = new e(),
        i,
        a = t;
      return ((i = n ? x(a) : b(a)), Object.assign(r, i), r);
    }
  },
  St = class {},
  Ct = class {},
  wt = class {},
  Tt = class {},
  Et = class {},
  Dt = class {},
  Ot = class {},
  kt = class e {
    _fromAPIResponse({ apiResponse: t, _isVertexAI: n }) {
      let r = new e(),
        i = O(t);
      return (Object.assign(r, i), r);
    }
  },
  At = class {},
  jt = class {},
  Mt = class {},
  Nt = class {},
  Pt = class {},
  Ft = class {
    get text() {
      let e = ``,
        t = !1,
        n = [];
      for (let r of this.serverContent?.modelTurn?.parts ?? []) {
        for (let [e, t] of Object.entries(r))
          e !== `text` && e !== `thought` && t !== null && n.push(e);
        if (typeof r.text == `string`) {
          if (typeof r.thought == `boolean` && r.thought) continue;
          ((t = !0), (e += r.text));
        }
      }
      return (
        n.length > 0 &&
          console.warn(
            `there are non-text parts ${n} in the response, returning concatenation of all text parts. Please refer to the non text parts for a full response from model.`,
          ),
        t ? e : void 0
      );
    }
    get data() {
      let e = ``,
        t = [];
      for (let n of this.serverContent?.modelTurn?.parts ?? []) {
        for (let [e, r] of Object.entries(n)) e !== `inlineData` && r !== null && t.push(e);
        n.inlineData && typeof n.inlineData.data == `string` && (e += atob(n.inlineData.data));
      }
      return (
        t.length > 0 &&
          console.warn(
            `there are non-data parts ${t} in the response, returning concatenation of all data parts. Please refer to the non data parts for a full response from model.`,
          ),
        e.length > 0 ? btoa(e) : void 0
      );
    }
  },
  It = class {
    get audioChunk() {
      if (
        this.serverContent &&
        this.serverContent.audioChunks &&
        this.serverContent.audioChunks.length > 0
      )
        return this.serverContent.audioChunks[0];
    }
  },
  Lt = class e {
    _fromAPIResponse({ apiResponse: t, _isVertexAI: n }) {
      let r = new e(),
        i = A(t);
      return (Object.assign(r, i), r);
    }
  };
function P(e, t) {
  if (!t || typeof t != `string`) throw Error(`model is required and must be a string`);
  if (t.includes(`..`) || t.includes(`?`) || t.includes(`&`))
    throw Error(`invalid model parameter`);
  if (e.isVertexAI()) {
    if (t.startsWith(`publishers/`) || t.startsWith(`projects/`) || t.startsWith(`models/`))
      return t;
    if (t.indexOf(`/`) >= 0) {
      let e = t.split(`/`, 2);
      return `publishers/${e[0]}/models/${e[1]}`;
    } else return `publishers/google/models/${t}`;
  } else if (t.startsWith(`models/`) || t.startsWith(`tunedModels/`)) return t;
  else return `models/${t}`;
}
function Rt(e, t) {
  let n = P(e, t);
  return n
    ? n.startsWith(`publishers/`) && e.isVertexAI()
      ? `projects/${e.getProject()}/locations/${e.getLocation()}/${n}`
      : n.startsWith(`models/`) && e.isVertexAI()
        ? `projects/${e.getProject()}/locations/${e.getLocation()}/publishers/google/${n}`
        : n
    : ``;
}
function zt(e) {
  return Array.isArray(e) ? e.map((e) => Bt(e)) : [Bt(e)];
}
function Bt(e) {
  if (typeof e == `object` && e) return e;
  throw Error(`Could not parse input as Blob. Unsupported blob type: ${typeof e}`);
}
function Vt(e) {
  let t = Bt(e);
  if (t.mimeType && t.mimeType.startsWith(`image/`)) return t;
  throw Error(`Unsupported mime type: ${t.mimeType}`);
}
function Ht(e) {
  let t = Bt(e);
  if (t.mimeType && t.mimeType.startsWith(`audio/`)) return t;
  throw Error(`Unsupported mime type: ${t.mimeType}`);
}
function Ut(e) {
  if (e == null) throw Error(`PartUnion is required`);
  if (typeof e == `object`) return e;
  if (typeof e == `string`) return { text: e };
  throw Error(`Unsupported part type: ${typeof e}`);
}
function Wt(e) {
  if (e == null || (Array.isArray(e) && e.length === 0)) throw Error(`PartListUnion is required`);
  return Array.isArray(e) ? e.map((e) => Ut(e)) : [Ut(e)];
}
function Gt(e) {
  return typeof e == `object` && !!e && `parts` in e && Array.isArray(e.parts);
}
function Kt(e) {
  return typeof e == `object` && !!e && `functionCall` in e;
}
function qt(e) {
  return typeof e == `object` && !!e && `functionResponse` in e;
}
function F(e) {
  if (e == null) throw Error(`ContentUnion is required`);
  return Gt(e) ? e : { role: `user`, parts: Wt(e) };
}
function Jt(e, t) {
  if (!t) return [];
  if (e.isVertexAI() && Array.isArray(t))
    return t.flatMap((e) => {
      let t = F(e);
      return t.parts && t.parts.length > 0 && t.parts[0].text !== void 0 ? [t.parts[0].text] : [];
    });
  if (e.isVertexAI()) {
    let e = F(t);
    return e.parts && e.parts.length > 0 && e.parts[0].text !== void 0 ? [e.parts[0].text] : [];
  }
  return Array.isArray(t) ? t.map((e) => F(e)) : [F(t)];
}
function I(e) {
  if (e == null || (Array.isArray(e) && e.length === 0)) throw Error(`contents are required`);
  if (!Array.isArray(e)) {
    if (Kt(e) || qt(e))
      throw Error(
        `To specify functionCall or functionResponse parts, please wrap them in a Content object, specifying the role for them`,
      );
    return [F(e)];
  }
  let t = [],
    n = [],
    r = Gt(e[0]);
  for (let i of e) {
    let e = Gt(i);
    if (e != r)
      throw Error(
        `Mixing Content and Parts is not supported, please group the parts into a the appropriate Content objects and specify the roles for them`,
      );
    if (e) t.push(i);
    else if (Kt(i) || qt(i))
      throw Error(
        `To specify functionCall or functionResponse parts, please wrap them, and any other parts, in Content objects as appropriate, specifying the role for them`,
      );
    else n.push(i);
  }
  return (r || t.push({ role: `user`, parts: Wt(n) }), t);
}
function Yt(e, t) {
  e.includes(`null`) && (t.nullable = !0);
  let n = e.filter((e) => e !== `null`);
  if (n.length === 1)
    t.type = Object.values(M).includes(n[0].toUpperCase())
      ? n[0].toUpperCase()
      : M.TYPE_UNSPECIFIED;
  else {
    t.anyOf = [];
    for (let e of n)
      t.anyOf.push({
        type: Object.values(M).includes(e.toUpperCase()) ? e.toUpperCase() : M.TYPE_UNSPECIFIED,
      });
  }
}
function L(e) {
  let t = {},
    n = [`items`],
    r = [`anyOf`],
    i = [`properties`];
  if (e.type && e.anyOf) throw Error(`type and anyOf cannot be both populated.`);
  let a = e.anyOf;
  (a != null &&
    a.length == 2 &&
    (a[0].type === `null`
      ? ((t.nullable = !0), (e = a[1]))
      : a[1].type === `null` && ((t.nullable = !0), (e = a[0]))),
    e.type instanceof Array && Yt(e.type, t));
  for (let [a, o] of Object.entries(e))
    if (o != null)
      if (a == `type`) {
        if (o === `null`)
          throw Error(`type: null can not be the only possible type for the field.`);
        if (o instanceof Array) continue;
        t.type = Object.values(M).includes(o.toUpperCase()) ? o.toUpperCase() : M.TYPE_UNSPECIFIED;
      } else if (n.includes(a)) t[a] = L(o);
      else if (r.includes(a)) {
        let e = [];
        for (let n of o) {
          if (n.type == `null`) {
            t.nullable = !0;
            continue;
          }
          e.push(L(n));
        }
        t[a] = e;
      } else if (i.includes(a)) {
        let e = {};
        for (let [t, n] of Object.entries(o)) e[t] = L(n);
        t[a] = e;
      } else {
        if (a === `additionalProperties`) continue;
        t[a] = o;
      }
  return t;
}
function Xt(e) {
  return L(e);
}
function Zt(e) {
  if (typeof e == `object`) return e;
  if (typeof e == `string`) return { voiceConfig: { prebuiltVoiceConfig: { voiceName: e } } };
  throw Error(`Unsupported speechConfig type: ${typeof e}`);
}
function Qt(e) {
  if (`multiSpeakerVoiceConfig` in e)
    throw Error(`multiSpeakerVoiceConfig is not supported in the live API.`);
  return e;
}
function R(e) {
  if (e.functionDeclarations)
    for (let t of e.functionDeclarations)
      (t.parameters &&
        (Object.keys(t.parameters).includes(`$schema`)
          ? t.parametersJsonSchema || ((t.parametersJsonSchema = t.parameters), delete t.parameters)
          : (t.parameters = L(t.parameters))),
        t.response &&
          (Object.keys(t.response).includes(`$schema`)
            ? t.responseJsonSchema || ((t.responseJsonSchema = t.response), delete t.response)
            : (t.response = L(t.response))));
  return e;
}
function z(e) {
  if (e == null) throw Error(`tools is required`);
  if (!Array.isArray(e)) throw Error(`tools is required and must be an array of Tools`);
  let t = [];
  for (let n of e) t.push(n);
  return t;
}
function $t(e, t, n, r = 1) {
  let i = !t.startsWith(`${n}/`) && t.split(`/`).length === r;
  return e.isVertexAI()
    ? t.startsWith(`projects/`)
      ? t
      : t.startsWith(`locations/`)
        ? `projects/${e.getProject()}/${t}`
        : t.startsWith(`${n}/`)
          ? `projects/${e.getProject()}/locations/${e.getLocation()}/${t}`
          : i
            ? `projects/${e.getProject()}/locations/${e.getLocation()}/${n}/${t}`
            : t
    : i
      ? `${n}/${t}`
      : t;
}
function B(e, t) {
  if (typeof t != `string`) throw Error(`name must be a string`);
  return $t(e, t, `cachedContents`);
}
function en(e) {
  switch (e) {
    case `STATE_UNSPECIFIED`:
      return `JOB_STATE_UNSPECIFIED`;
    case `CREATING`:
      return `JOB_STATE_RUNNING`;
    case `ACTIVE`:
      return `JOB_STATE_SUCCEEDED`;
    case `FAILED`:
      return `JOB_STATE_FAILED`;
    default:
      return e;
  }
}
function V(e) {
  return v(e);
}
function tn(e) {
  return typeof e == `object` && !!e && `name` in e;
}
function nn(e) {
  return typeof e == `object` && !!e && `video` in e;
}
function rn(e) {
  return typeof e == `object` && !!e && `uri` in e;
}
function an(e) {
  let t;
  if (
    (tn(e) && (t = e.name),
    !(rn(e) && ((t = e.uri), t === void 0)) && !(nn(e) && ((t = e.video?.uri), t === void 0)))
  ) {
    if ((typeof e == `string` && (t = e), t === void 0))
      throw Error(`Could not extract file name from the provided input.`);
    if (t.startsWith(`https://`)) {
      let e = t.split(`files/`)[1].match(/[a-z0-9]+/);
      if (e === null) throw Error(`Could not extract file name from URI ${t}`);
      t = e[0];
    } else t.startsWith(`files/`) && (t = t.split(`files/`)[1]);
    return t;
  }
}
function on(e, t) {
  let n;
  return (
    (n = e.isVertexAI()
      ? t
        ? `publishers/google/models`
        : `models`
      : t
        ? `models`
        : `tunedModels`),
    n
  );
}
function sn(e) {
  for (let t of [`models`, `tunedModels`, `publisherModels`]) if (cn(e, t)) return e[t];
  return [];
}
function cn(e, t) {
  return typeof e == `object` && !!e && t in e;
}
function ln(e, t = {}) {
  let n = e,
    r = { name: n.name, description: n.description, parametersJsonSchema: n.inputSchema };
  return (
    n.outputSchema && (r.responseJsonSchema = n.outputSchema),
    t.behavior && (r.behavior = t.behavior),
    { functionDeclarations: [r] }
  );
}
function un(e, t = {}) {
  let n = [],
    r = new Set();
  for (let i of e) {
    let e = i.name;
    if (r.has(e))
      throw Error(
        `Duplicate function name ${e} found in MCP tools. Please ensure function names are unique.`,
      );
    r.add(e);
    let a = ln(i, t);
    a.functionDeclarations && n.push(...a.functionDeclarations);
  }
  return { functionDeclarations: n };
}
function dn(e, t) {
  let n;
  if (typeof t == `string`)
    if (e.isVertexAI())
      if (t.startsWith(`gs://`)) n = { format: `jsonl`, gcsUri: [t] };
      else if (t.startsWith(`bq://`)) n = { format: `bigquery`, bigqueryUri: t };
      else throw Error(`Unsupported string source for Vertex AI: ${t}`);
    else if (t.startsWith(`files/`)) n = { fileName: t };
    else throw Error(`Unsupported string source for Gemini API: ${t}`);
  else if (Array.isArray(t)) {
    if (e.isVertexAI()) throw Error(`InlinedRequest[] is not supported in Vertex AI.`);
    n = { inlinedRequests: t };
  } else n = t;
  let r = [n.gcsUri, n.bigqueryUri].filter(Boolean).length,
    i = [n.inlinedRequests, n.fileName].filter(Boolean).length;
  if (e.isVertexAI()) {
    if (i > 0 || r !== 1)
      throw Error("Exactly one of `gcsUri` or `bigqueryUri` must be set for Vertex AI.");
  } else if (r > 0 || i !== 1)
    throw Error("Exactly one of `inlinedRequests`, `fileName`, must be set for Gemini API.");
  return n;
}
function fn(e) {
  if (typeof e != `string`) return e;
  let t = e;
  if (t.startsWith(`gs://`)) return { format: `jsonl`, gcsUri: t };
  if (t.startsWith(`bq://`)) return { format: `bigquery`, bigqueryUri: t };
  throw Error(`Unsupported destination: ${t}`);
}
function pn(e) {
  if (typeof e != `object` || !e) return {};
  let t = e,
    n = t.inlinedResponses;
  if (typeof n != `object` || !n) return e;
  let r = n.inlinedResponses;
  if (!Array.isArray(r) || r.length === 0) return e;
  let i = !1;
  for (let e of r) {
    if (typeof e != `object` || !e) continue;
    let t = e.response;
    if (!(typeof t != `object` || !t) && t.embedding !== void 0) {
      i = !0;
      break;
    }
  }
  return (
    i && ((t.inlinedEmbedContentResponses = t.inlinedResponses), delete t.inlinedResponses), e
  );
}
function mn(e, t) {
  let n = t;
  if (!e.isVertexAI()) {
    if (/batches\/[^/]+$/.test(n)) return n.split(`/`).pop();
    throw Error(`Invalid batch job name: ${n}.`);
  }
  if (/^projects\/[^/]+\/locations\/[^/]+\/batchPredictionJobs\/[^/]+$/.test(n))
    return n.split(`/`).pop();
  if (/^\d+$/.test(n)) return n;
  throw Error(`Invalid batch job name: ${n}.`);
}
function hn(e) {
  let t = e;
  return t === `BATCH_STATE_UNSPECIFIED`
    ? `JOB_STATE_UNSPECIFIED`
    : t === `BATCH_STATE_PENDING`
      ? `JOB_STATE_PENDING`
      : t === `BATCH_STATE_RUNNING`
        ? `JOB_STATE_RUNNING`
        : t === `BATCH_STATE_SUCCEEDED`
          ? `JOB_STATE_SUCCEEDED`
          : t === `BATCH_STATE_FAILED`
            ? `JOB_STATE_FAILED`
            : t === `BATCH_STATE_CANCELLED`
              ? `JOB_STATE_CANCELLED`
              : t === `BATCH_STATE_EXPIRED`
                ? `JOB_STATE_EXPIRED`
                : t;
}
function gn(e) {
  return (e.includes(`gemini`) && e !== `gemini-embedding-001`) || e.includes(`maas`);
}
function _n(e) {
  let t = {},
    n = h(e, [`apiKey`]);
  if ((n != null && m(t, [`apiKey`], n), h(e, [`apiKeyConfig`]) !== void 0))
    throw Error(`apiKeyConfig parameter is not supported in Gemini API.`);
  if (h(e, [`authType`]) !== void 0)
    throw Error(`authType parameter is not supported in Gemini API.`);
  if (h(e, [`googleServiceAccountConfig`]) !== void 0)
    throw Error(`googleServiceAccountConfig parameter is not supported in Gemini API.`);
  if (h(e, [`httpBasicAuthConfig`]) !== void 0)
    throw Error(`httpBasicAuthConfig parameter is not supported in Gemini API.`);
  if (h(e, [`oauthConfig`]) !== void 0)
    throw Error(`oauthConfig parameter is not supported in Gemini API.`);
  if (h(e, [`oidcConfig`]) !== void 0)
    throw Error(`oidcConfig parameter is not supported in Gemini API.`);
  return t;
}
function vn(e) {
  let t = {},
    n = h(e, [`responsesFile`]);
  n != null && m(t, [`fileName`], n);
  let r = h(e, [`inlinedResponses`, `inlinedResponses`]);
  if (r != null) {
    let e = r;
    (Array.isArray(e) && (e = e.map((e) => nr(e))), m(t, [`inlinedResponses`], e));
  }
  let i = h(e, [`inlinedEmbedContentResponses`, `inlinedResponses`]);
  if (i != null) {
    let e = i;
    (Array.isArray(e) && (e = e.map((e) => e)), m(t, [`inlinedEmbedContentResponses`], e));
  }
  return t;
}
function yn(e) {
  let t = {},
    n = h(e, [`predictionsFormat`]);
  n != null && m(t, [`format`], n);
  let r = h(e, [`gcsDestination`, `outputUriPrefix`]);
  r != null && m(t, [`gcsUri`], r);
  let i = h(e, [`bigqueryDestination`, `outputUri`]);
  return (i != null && m(t, [`bigqueryUri`], i), t);
}
function bn(e) {
  let t = {},
    n = h(e, [`format`]);
  n != null && m(t, [`predictionsFormat`], n);
  let r = h(e, [`gcsUri`]);
  r != null && m(t, [`gcsDestination`, `outputUriPrefix`], r);
  let i = h(e, [`bigqueryUri`]);
  if ((i != null && m(t, [`bigqueryDestination`, `outputUri`], i), h(e, [`fileName`]) !== void 0))
    throw Error(`fileName parameter is not supported in Vertex AI.`);
  if (h(e, [`inlinedResponses`]) !== void 0)
    throw Error(`inlinedResponses parameter is not supported in Vertex AI.`);
  if (h(e, [`inlinedEmbedContentResponses`]) !== void 0)
    throw Error(`inlinedEmbedContentResponses parameter is not supported in Vertex AI.`);
  return t;
}
function xn(e) {
  let t = {},
    n = h(e, [`name`]);
  n != null && m(t, [`name`], n);
  let r = h(e, [`metadata`, `displayName`]);
  r != null && m(t, [`displayName`], r);
  let i = h(e, [`metadata`, `state`]);
  i != null && m(t, [`state`], hn(i));
  let a = h(e, [`metadata`, `createTime`]);
  a != null && m(t, [`createTime`], a);
  let o = h(e, [`metadata`, `endTime`]);
  o != null && m(t, [`endTime`], o);
  let s = h(e, [`metadata`, `updateTime`]);
  s != null && m(t, [`updateTime`], s);
  let c = h(e, [`metadata`, `model`]);
  c != null && m(t, [`model`], c);
  let l = h(e, [`metadata`, `output`]);
  return (l != null && m(t, [`dest`], vn(pn(l))), t);
}
function Sn(e) {
  let t = {},
    n = h(e, [`name`]);
  n != null && m(t, [`name`], n);
  let r = h(e, [`displayName`]);
  r != null && m(t, [`displayName`], r);
  let i = h(e, [`state`]);
  i != null && m(t, [`state`], hn(i));
  let a = h(e, [`error`]);
  a != null && m(t, [`error`], a);
  let o = h(e, [`createTime`]);
  o != null && m(t, [`createTime`], o);
  let s = h(e, [`startTime`]);
  s != null && m(t, [`startTime`], s);
  let c = h(e, [`endTime`]);
  c != null && m(t, [`endTime`], c);
  let l = h(e, [`updateTime`]);
  l != null && m(t, [`updateTime`], l);
  let u = h(e, [`model`]);
  u != null && m(t, [`model`], u);
  let d = h(e, [`inputConfig`]);
  d != null && m(t, [`src`], Cn(d));
  let f = h(e, [`outputConfig`]);
  f != null && m(t, [`dest`], yn(pn(f)));
  let p = h(e, [`completionStats`]);
  return (p != null && m(t, [`completionStats`], p), t);
}
function Cn(e) {
  let t = {},
    n = h(e, [`instancesFormat`]);
  n != null && m(t, [`format`], n);
  let r = h(e, [`gcsSource`, `uris`]);
  r != null && m(t, [`gcsUri`], r);
  let i = h(e, [`bigquerySource`, `inputUri`]);
  return (i != null && m(t, [`bigqueryUri`], i), t);
}
function wn(e, t) {
  let n = {};
  if (h(t, [`format`]) !== void 0) throw Error(`format parameter is not supported in Gemini API.`);
  if (h(t, [`gcsUri`]) !== void 0) throw Error(`gcsUri parameter is not supported in Gemini API.`);
  if (h(t, [`bigqueryUri`]) !== void 0)
    throw Error(`bigqueryUri parameter is not supported in Gemini API.`);
  let r = h(t, [`fileName`]);
  r != null && m(n, [`fileName`], r);
  let i = h(t, [`inlinedRequests`]);
  if (i != null) {
    let t = i;
    (Array.isArray(t) && (t = t.map((t) => tr(e, t))), m(n, [`requests`, `requests`], t));
  }
  return n;
}
function Tn(e) {
  let t = {},
    n = h(e, [`format`]);
  n != null && m(t, [`instancesFormat`], n);
  let r = h(e, [`gcsUri`]);
  r != null && m(t, [`gcsSource`, `uris`], r);
  let i = h(e, [`bigqueryUri`]);
  if ((i != null && m(t, [`bigquerySource`, `inputUri`], i), h(e, [`fileName`]) !== void 0))
    throw Error(`fileName parameter is not supported in Vertex AI.`);
  if (h(e, [`inlinedRequests`]) !== void 0)
    throw Error(`inlinedRequests parameter is not supported in Vertex AI.`);
  return t;
}
function En(e) {
  let t = {},
    n = h(e, [`data`]);
  if ((n != null && m(t, [`data`], n), h(e, [`displayName`]) !== void 0))
    throw Error(`displayName parameter is not supported in Gemini API.`);
  let r = h(e, [`mimeType`]);
  return (r != null && m(t, [`mimeType`], r), t);
}
function Dn(e, t) {
  let n = {},
    r = h(t, [`name`]);
  return (r != null && m(n, [`_url`, `name`], mn(e, r)), n);
}
function On(e, t) {
  let n = {},
    r = h(t, [`name`]);
  return (r != null && m(n, [`_url`, `name`], mn(e, r)), n);
}
function kn(e) {
  let t = {},
    n = h(e, [`content`]);
  n != null && m(t, [`content`], n);
  let r = h(e, [`citationMetadata`]);
  r != null && m(t, [`citationMetadata`], An(r));
  let i = h(e, [`tokenCount`]);
  i != null && m(t, [`tokenCount`], i);
  let a = h(e, [`finishReason`]);
  a != null && m(t, [`finishReason`], a);
  let o = h(e, [`groundingMetadata`]);
  o != null && m(t, [`groundingMetadata`], o);
  let s = h(e, [`avgLogprobs`]);
  s != null && m(t, [`avgLogprobs`], s);
  let c = h(e, [`index`]);
  c != null && m(t, [`index`], c);
  let l = h(e, [`logprobsResult`]);
  l != null && m(t, [`logprobsResult`], l);
  let u = h(e, [`safetyRatings`]);
  if (u != null) {
    let e = u;
    (Array.isArray(e) && (e = e.map((e) => e)), m(t, [`safetyRatings`], e));
  }
  let d = h(e, [`urlContextMetadata`]);
  return (d != null && m(t, [`urlContextMetadata`], d), t);
}
function An(e) {
  let t = {},
    n = h(e, [`citationSources`]);
  if (n != null) {
    let e = n;
    (Array.isArray(e) && (e = e.map((e) => e)), m(t, [`citations`], e));
  }
  return t;
}
function jn(e) {
  let t = {},
    n = h(e, [`parts`]);
  if (n != null) {
    let e = n;
    (Array.isArray(e) && (e = e.map((e) => lr(e))), m(t, [`parts`], e));
  }
  let r = h(e, [`role`]);
  return (r != null && m(t, [`role`], r), t);
}
function Mn(e, t) {
  let n = {},
    r = h(e, [`displayName`]);
  if ((t !== void 0 && r != null && m(t, [`batch`, `displayName`], r), h(e, [`dest`]) !== void 0))
    throw Error(`dest parameter is not supported in Gemini API.`);
  return n;
}
function Nn(e, t) {
  let n = {},
    r = h(e, [`displayName`]);
  t !== void 0 && r != null && m(t, [`displayName`], r);
  let i = h(e, [`dest`]);
  return (t !== void 0 && i != null && m(t, [`outputConfig`], bn(fn(i))), n);
}
function Pn(e, t) {
  let n = {},
    r = h(t, [`model`]);
  r != null && m(n, [`_url`, `model`], P(e, r));
  let i = h(t, [`src`]);
  i != null && m(n, [`batch`, `inputConfig`], wn(e, dn(e, i)));
  let a = h(t, [`config`]);
  return (a != null && Mn(a, n), n);
}
function Fn(e, t) {
  let n = {},
    r = h(t, [`model`]);
  r != null && m(n, [`model`], P(e, r));
  let i = h(t, [`src`]);
  i != null && m(n, [`inputConfig`], Tn(dn(e, i)));
  let a = h(t, [`config`]);
  return (a != null && Nn(a, n), n);
}
function In(e, t) {
  let n = {},
    r = h(e, [`displayName`]);
  return (t !== void 0 && r != null && m(t, [`batch`, `displayName`], r), n);
}
function Ln(e, t) {
  let n = {},
    r = h(t, [`model`]);
  r != null && m(n, [`_url`, `model`], P(e, r));
  let i = h(t, [`src`]);
  i != null && m(n, [`batch`, `inputConfig`], Wn(e, i));
  let a = h(t, [`config`]);
  return (a != null && In(a, n), n);
}
function Rn(e, t) {
  let n = {},
    r = h(t, [`name`]);
  return (r != null && m(n, [`_url`, `name`], mn(e, r)), n);
}
function zn(e, t) {
  let n = {},
    r = h(t, [`name`]);
  return (r != null && m(n, [`_url`, `name`], mn(e, r)), n);
}
function Bn(e) {
  let t = {},
    n = h(e, [`sdkHttpResponse`]);
  n != null && m(t, [`sdkHttpResponse`], n);
  let r = h(e, [`name`]);
  r != null && m(t, [`name`], r);
  let i = h(e, [`done`]);
  i != null && m(t, [`done`], i);
  let a = h(e, [`error`]);
  return (a != null && m(t, [`error`], a), t);
}
function Vn(e) {
  let t = {},
    n = h(e, [`sdkHttpResponse`]);
  n != null && m(t, [`sdkHttpResponse`], n);
  let r = h(e, [`name`]);
  r != null && m(t, [`name`], r);
  let i = h(e, [`done`]);
  i != null && m(t, [`done`], i);
  let a = h(e, [`error`]);
  return (a != null && m(t, [`error`], a), t);
}
function Hn(e, t) {
  let n = {},
    r = h(t, [`contents`]);
  if (r != null) {
    let t = Jt(e, r);
    (Array.isArray(t) && (t = t.map((e) => e)), m(n, [`requests[]`, `request`, `content`], t));
  }
  let i = h(t, [`config`]);
  return (
    i != null && (m(n, [`_self`], Un(i, n)), g(n, { "requests[].*": `requests[].request.*` })), n
  );
}
function Un(e, t) {
  let n = {},
    r = h(e, [`taskType`]);
  t !== void 0 && r != null && m(t, [`requests[]`, `taskType`], r);
  let i = h(e, [`title`]);
  t !== void 0 && i != null && m(t, [`requests[]`, `title`], i);
  let a = h(e, [`outputDimensionality`]);
  if (
    (t !== void 0 && a != null && m(t, [`requests[]`, `outputDimensionality`], a),
    h(e, [`mimeType`]) !== void 0)
  )
    throw Error(`mimeType parameter is not supported in Gemini API.`);
  if (h(e, [`autoTruncate`]) !== void 0)
    throw Error(`autoTruncate parameter is not supported in Gemini API.`);
  return n;
}
function Wn(e, t) {
  let n = {},
    r = h(t, [`fileName`]);
  r != null && m(n, [`file_name`], r);
  let i = h(t, [`inlinedRequests`]);
  return (i != null && m(n, [`requests`], Hn(e, i)), n);
}
function Gn(e) {
  let t = {};
  if (h(e, [`displayName`]) !== void 0)
    throw Error(`displayName parameter is not supported in Gemini API.`);
  let n = h(e, [`fileUri`]);
  n != null && m(t, [`fileUri`], n);
  let r = h(e, [`mimeType`]);
  return (r != null && m(t, [`mimeType`], r), t);
}
function Kn(e) {
  let t = {},
    n = h(e, [`id`]);
  n != null && m(t, [`id`], n);
  let r = h(e, [`args`]);
  r != null && m(t, [`args`], r);
  let i = h(e, [`name`]);
  if ((i != null && m(t, [`name`], i), h(e, [`partialArgs`]) !== void 0))
    throw Error(`partialArgs parameter is not supported in Gemini API.`);
  if (h(e, [`willContinue`]) !== void 0)
    throw Error(`willContinue parameter is not supported in Gemini API.`);
  return t;
}
function qn(e) {
  let t = {},
    n = h(e, [`allowedFunctionNames`]);
  n != null && m(t, [`allowedFunctionNames`], n);
  let r = h(e, [`mode`]);
  if ((r != null && m(t, [`mode`], r), h(e, [`streamFunctionCallArguments`]) !== void 0))
    throw Error(`streamFunctionCallArguments parameter is not supported in Gemini API.`);
  return t;
}
function Jn(e, t, n) {
  let r = {},
    i = h(t, [`systemInstruction`]);
  n !== void 0 && i != null && m(n, [`systemInstruction`], jn(F(i)));
  let a = h(t, [`temperature`]);
  a != null && m(r, [`temperature`], a);
  let o = h(t, [`topP`]);
  o != null && m(r, [`topP`], o);
  let s = h(t, [`topK`]);
  s != null && m(r, [`topK`], s);
  let c = h(t, [`candidateCount`]);
  c != null && m(r, [`candidateCount`], c);
  let l = h(t, [`maxOutputTokens`]);
  l != null && m(r, [`maxOutputTokens`], l);
  let u = h(t, [`stopSequences`]);
  u != null && m(r, [`stopSequences`], u);
  let d = h(t, [`responseLogprobs`]);
  d != null && m(r, [`responseLogprobs`], d);
  let f = h(t, [`logprobs`]);
  f != null && m(r, [`logprobs`], f);
  let p = h(t, [`presencePenalty`]);
  p != null && m(r, [`presencePenalty`], p);
  let g = h(t, [`frequencyPenalty`]);
  g != null && m(r, [`frequencyPenalty`], g);
  let _ = h(t, [`seed`]);
  _ != null && m(r, [`seed`], _);
  let v = h(t, [`responseMimeType`]);
  v != null && m(r, [`responseMimeType`], v);
  let y = h(t, [`responseSchema`]);
  y != null && m(r, [`responseSchema`], Xt(y));
  let b = h(t, [`responseJsonSchema`]);
  if ((b != null && m(r, [`responseJsonSchema`], b), h(t, [`routingConfig`]) !== void 0))
    throw Error(`routingConfig parameter is not supported in Gemini API.`);
  if (h(t, [`modelSelectionConfig`]) !== void 0)
    throw Error(`modelSelectionConfig parameter is not supported in Gemini API.`);
  let x = h(t, [`safetySettings`]);
  if (n !== void 0 && x != null) {
    let e = x;
    (Array.isArray(e) && (e = e.map((e) => ur(e))), m(n, [`safetySettings`], e));
  }
  let S = h(t, [`tools`]);
  if (n !== void 0 && S != null) {
    let e = z(S);
    (Array.isArray(e) && (e = e.map((e) => fr(R(e)))), m(n, [`tools`], e));
  }
  let C = h(t, [`toolConfig`]);
  if ((n !== void 0 && C != null && m(n, [`toolConfig`], dr(C)), h(t, [`labels`]) !== void 0))
    throw Error(`labels parameter is not supported in Gemini API.`);
  let w = h(t, [`cachedContent`]);
  n !== void 0 && w != null && m(n, [`cachedContent`], B(e, w));
  let T = h(t, [`responseModalities`]);
  T != null && m(r, [`responseModalities`], T);
  let E = h(t, [`mediaResolution`]);
  E != null && m(r, [`mediaResolution`], E);
  let D = h(t, [`speechConfig`]);
  if ((D != null && m(r, [`speechConfig`], Zt(D)), h(t, [`audioTimestamp`]) !== void 0))
    throw Error(`audioTimestamp parameter is not supported in Gemini API.`);
  let O = h(t, [`thinkingConfig`]);
  O != null && m(r, [`thinkingConfig`], O);
  let k = h(t, [`imageConfig`]);
  k != null && m(r, [`imageConfig`], er(k));
  let A = h(t, [`enableEnhancedCivicAnswers`]);
  if ((A != null && m(r, [`enableEnhancedCivicAnswers`], A), h(t, [`modelArmorConfig`]) !== void 0))
    throw Error(`modelArmorConfig parameter is not supported in Gemini API.`);
  return r;
}
function Yn(e) {
  let t = {},
    n = h(e, [`sdkHttpResponse`]);
  n != null && m(t, [`sdkHttpResponse`], n);
  let r = h(e, [`candidates`]);
  if (r != null) {
    let e = r;
    (Array.isArray(e) && (e = e.map((e) => kn(e))), m(t, [`candidates`], e));
  }
  let i = h(e, [`modelVersion`]);
  i != null && m(t, [`modelVersion`], i);
  let a = h(e, [`promptFeedback`]);
  a != null && m(t, [`promptFeedback`], a);
  let o = h(e, [`responseId`]);
  o != null && m(t, [`responseId`], o);
  let s = h(e, [`usageMetadata`]);
  return (s != null && m(t, [`usageMetadata`], s), t);
}
function Xn(e, t) {
  let n = {},
    r = h(t, [`name`]);
  return (r != null && m(n, [`_url`, `name`], mn(e, r)), n);
}
function Zn(e, t) {
  let n = {},
    r = h(t, [`name`]);
  return (r != null && m(n, [`_url`, `name`], mn(e, r)), n);
}
function Qn(e) {
  let t = {},
    n = h(e, [`authConfig`]);
  n != null && m(t, [`authConfig`], _n(n));
  let r = h(e, [`enableWidget`]);
  return (r != null && m(t, [`enableWidget`], r), t);
}
function $n(e) {
  let t = {},
    n = h(e, [`searchTypes`]);
  if ((n != null && m(t, [`searchTypes`], n), h(e, [`blockingConfidence`]) !== void 0))
    throw Error(`blockingConfidence parameter is not supported in Gemini API.`);
  if (h(e, [`excludeDomains`]) !== void 0)
    throw Error(`excludeDomains parameter is not supported in Gemini API.`);
  let r = h(e, [`timeRangeFilter`]);
  return (r != null && m(t, [`timeRangeFilter`], r), t);
}
function er(e) {
  let t = {},
    n = h(e, [`aspectRatio`]);
  n != null && m(t, [`aspectRatio`], n);
  let r = h(e, [`imageSize`]);
  if ((r != null && m(t, [`imageSize`], r), h(e, [`personGeneration`]) !== void 0))
    throw Error(`personGeneration parameter is not supported in Gemini API.`);
  if (h(e, [`prominentPeople`]) !== void 0)
    throw Error(`prominentPeople parameter is not supported in Gemini API.`);
  if (h(e, [`outputMimeType`]) !== void 0)
    throw Error(`outputMimeType parameter is not supported in Gemini API.`);
  if (h(e, [`outputCompressionQuality`]) !== void 0)
    throw Error(`outputCompressionQuality parameter is not supported in Gemini API.`);
  if (h(e, [`imageOutputOptions`]) !== void 0)
    throw Error(`imageOutputOptions parameter is not supported in Gemini API.`);
  return t;
}
function tr(e, t) {
  let n = {},
    r = h(t, [`model`]);
  r != null && m(n, [`request`, `model`], P(e, r));
  let i = h(t, [`contents`]);
  if (i != null) {
    let e = I(i);
    (Array.isArray(e) && (e = e.map((e) => jn(e))), m(n, [`request`, `contents`], e));
  }
  let a = h(t, [`metadata`]);
  a != null && m(n, [`metadata`], a);
  let o = h(t, [`config`]);
  return (o != null && m(n, [`request`, `generationConfig`], Jn(e, o, h(n, [`request`], {}))), n);
}
function nr(e) {
  let t = {},
    n = h(e, [`response`]);
  n != null && m(t, [`response`], Yn(n));
  let r = h(e, [`metadata`]);
  r != null && m(t, [`metadata`], r);
  let i = h(e, [`error`]);
  return (i != null && m(t, [`error`], i), t);
}
function rr(e, t) {
  let n = {},
    r = h(e, [`pageSize`]);
  t !== void 0 && r != null && m(t, [`_query`, `pageSize`], r);
  let i = h(e, [`pageToken`]);
  if ((t !== void 0 && i != null && m(t, [`_query`, `pageToken`], i), h(e, [`filter`]) !== void 0))
    throw Error(`filter parameter is not supported in Gemini API.`);
  return n;
}
function ir(e, t) {
  let n = {},
    r = h(e, [`pageSize`]);
  t !== void 0 && r != null && m(t, [`_query`, `pageSize`], r);
  let i = h(e, [`pageToken`]);
  t !== void 0 && i != null && m(t, [`_query`, `pageToken`], i);
  let a = h(e, [`filter`]);
  return (t !== void 0 && a != null && m(t, [`_query`, `filter`], a), n);
}
function ar(e) {
  let t = {},
    n = h(e, [`config`]);
  return (n != null && rr(n, t), t);
}
function or(e) {
  let t = {},
    n = h(e, [`config`]);
  return (n != null && ir(n, t), t);
}
function sr(e) {
  let t = {},
    n = h(e, [`sdkHttpResponse`]);
  n != null && m(t, [`sdkHttpResponse`], n);
  let r = h(e, [`nextPageToken`]);
  r != null && m(t, [`nextPageToken`], r);
  let i = h(e, [`operations`]);
  if (i != null) {
    let e = i;
    (Array.isArray(e) && (e = e.map((e) => xn(e))), m(t, [`batchJobs`], e));
  }
  return t;
}
function cr(e) {
  let t = {},
    n = h(e, [`sdkHttpResponse`]);
  n != null && m(t, [`sdkHttpResponse`], n);
  let r = h(e, [`nextPageToken`]);
  r != null && m(t, [`nextPageToken`], r);
  let i = h(e, [`batchPredictionJobs`]);
  if (i != null) {
    let e = i;
    (Array.isArray(e) && (e = e.map((e) => Sn(e))), m(t, [`batchJobs`], e));
  }
  return t;
}
function lr(e) {
  let t = {},
    n = h(e, [`mediaResolution`]);
  n != null && m(t, [`mediaResolution`], n);
  let r = h(e, [`codeExecutionResult`]);
  r != null && m(t, [`codeExecutionResult`], r);
  let i = h(e, [`executableCode`]);
  i != null && m(t, [`executableCode`], i);
  let a = h(e, [`fileData`]);
  a != null && m(t, [`fileData`], Gn(a));
  let o = h(e, [`functionCall`]);
  o != null && m(t, [`functionCall`], Kn(o));
  let s = h(e, [`functionResponse`]);
  s != null && m(t, [`functionResponse`], s);
  let c = h(e, [`inlineData`]);
  c != null && m(t, [`inlineData`], En(c));
  let l = h(e, [`text`]);
  l != null && m(t, [`text`], l);
  let u = h(e, [`thought`]);
  u != null && m(t, [`thought`], u);
  let d = h(e, [`thoughtSignature`]);
  d != null && m(t, [`thoughtSignature`], d);
  let f = h(e, [`videoMetadata`]);
  f != null && m(t, [`videoMetadata`], f);
  let p = h(e, [`toolCall`]);
  p != null && m(t, [`toolCall`], p);
  let g = h(e, [`toolResponse`]);
  return (g != null && m(t, [`toolResponse`], g), t);
}
function ur(e) {
  let t = {},
    n = h(e, [`category`]);
  if ((n != null && m(t, [`category`], n), h(e, [`method`]) !== void 0))
    throw Error(`method parameter is not supported in Gemini API.`);
  let r = h(e, [`threshold`]);
  return (r != null && m(t, [`threshold`], r), t);
}
function dr(e) {
  let t = {},
    n = h(e, [`retrievalConfig`]);
  n != null && m(t, [`retrievalConfig`], n);
  let r = h(e, [`functionCallingConfig`]);
  r != null && m(t, [`functionCallingConfig`], qn(r));
  let i = h(e, [`includeServerSideToolInvocations`]);
  return (i != null && m(t, [`includeServerSideToolInvocations`], i), t);
}
function fr(e) {
  let t = {};
  if (h(e, [`retrieval`]) !== void 0)
    throw Error(`retrieval parameter is not supported in Gemini API.`);
  let n = h(e, [`computerUse`]);
  n != null && m(t, [`computerUse`], n);
  let r = h(e, [`fileSearch`]);
  r != null && m(t, [`fileSearch`], r);
  let i = h(e, [`googleSearch`]);
  i != null && m(t, [`googleSearch`], $n(i));
  let a = h(e, [`googleMaps`]);
  a != null && m(t, [`googleMaps`], Qn(a));
  let o = h(e, [`codeExecution`]);
  if ((o != null && m(t, [`codeExecution`], o), h(e, [`enterpriseWebSearch`]) !== void 0))
    throw Error(`enterpriseWebSearch parameter is not supported in Gemini API.`);
  let s = h(e, [`functionDeclarations`]);
  if (s != null) {
    let e = s;
    (Array.isArray(e) && (e = e.map((e) => e)), m(t, [`functionDeclarations`], e));
  }
  let c = h(e, [`googleSearchRetrieval`]);
  if ((c != null && m(t, [`googleSearchRetrieval`], c), h(e, [`parallelAiSearch`]) !== void 0))
    throw Error(`parallelAiSearch parameter is not supported in Gemini API.`);
  let l = h(e, [`urlContext`]);
  l != null && m(t, [`urlContext`], l);
  let u = h(e, [`mcpServers`]);
  if (u != null) {
    let e = u;
    (Array.isArray(e) && (e = e.map((e) => e)), m(t, [`mcpServers`], e));
  }
  return t;
}
var H;
(function (e) {
  ((e.PAGED_ITEM_BATCH_JOBS = `batchJobs`),
    (e.PAGED_ITEM_MODELS = `models`),
    (e.PAGED_ITEM_TUNING_JOBS = `tuningJobs`),
    (e.PAGED_ITEM_FILES = `files`),
    (e.PAGED_ITEM_CACHED_CONTENTS = `cachedContents`),
    (e.PAGED_ITEM_FILE_SEARCH_STORES = `fileSearchStores`),
    (e.PAGED_ITEM_DOCUMENTS = `documents`));
})((H ||= {}));
var U = class {
    constructor(e, t, n, r) {
      ((this.pageInternal = []),
        (this.paramsInternal = {}),
        (this.requestInternal = t),
        this.init(e, n, r));
    }
    init(e, t, n) {
      ((this.nameInternal = e),
        (this.pageInternal = t[this.nameInternal] || []),
        (this.sdkHttpResponseInternal = t?.sdkHttpResponse),
        (this.idxInternal = 0));
      let r = { config: {} };
      ((r =
        !n || Object.keys(n).length === 0
          ? { config: {} }
          : typeof n == `object`
            ? Object.assign({}, n)
            : n),
        r.config && (r.config.pageToken = t.nextPageToken),
        (this.paramsInternal = r),
        (this.pageInternalSize = r.config?.pageSize ?? this.pageInternal.length));
    }
    initNextPage(e) {
      this.init(this.nameInternal, e, this.paramsInternal);
    }
    get page() {
      return this.pageInternal;
    }
    get name() {
      return this.nameInternal;
    }
    get pageSize() {
      return this.pageInternalSize;
    }
    get sdkHttpResponse() {
      return this.sdkHttpResponseInternal;
    }
    get params() {
      return this.paramsInternal;
    }
    get pageLength() {
      return this.pageInternal.length;
    }
    getItem(e) {
      return this.pageInternal[e];
    }
    [Symbol.asyncIterator]() {
      return {
        next: async () => {
          if (this.idxInternal >= this.pageLength)
            if (this.hasNextPage()) await this.nextPage();
            else return { value: void 0, done: !0 };
          let e = this.getItem(this.idxInternal);
          return ((this.idxInternal += 1), { value: e, done: !1 });
        },
        return: async () => ({ value: void 0, done: !0 }),
      };
    }
    async nextPage() {
      if (!this.hasNextPage()) throw Error(`No more pages to fetch.`);
      let e = await this.requestInternal(this.params);
      return (this.initNextPage(e), this.page);
    }
    hasNextPage() {
      return this.params.config?.pageToken !== void 0;
    }
  },
  pr = class extends f {
    constructor(e) {
      (super(),
        (this.apiClient = e),
        (this.list = async (e = {}) =>
          new U(
            H.PAGED_ITEM_BATCH_JOBS,
            (e) => this.listInternal(e),
            await this.listInternal(e),
            e,
          )),
        (this.create = async (e) => (
          this.apiClient.isVertexAI() && (e.config = this.formatDestination(e.src, e.config)),
          this.createInternal(e)
        )),
        (this.createEmbeddings = async (e) => {
          if (
            (console.warn(
              `batches.createEmbeddings() is experimental and may change without notice.`,
            ),
            this.apiClient.isVertexAI())
          )
            throw Error(`Vertex AI does not support batches.createEmbeddings.`);
          return this.createEmbeddingsInternal(e);
        }));
    }
    createInlinedGenerateContentRequest(e) {
      let t = Pn(this.apiClient, e),
        n = t._url,
        r = p(`{model}:batchGenerateContent`, n),
        i = t.batch.inputConfig.requests,
        a = i.requests,
        o = [];
      for (let e of a) {
        let t = Object.assign({}, e);
        if (t.systemInstruction) {
          let e = t.systemInstruction;
          delete t.systemInstruction;
          let n = t.request;
          ((n.systemInstruction = e), (t.request = n));
        }
        o.push(t);
      }
      return (
        (i.requests = o), delete t.config, delete t._url, delete t._query, { path: r, body: t }
      );
    }
    getGcsUri(e) {
      if (typeof e == `string`) return e.startsWith(`gs://`) ? e : void 0;
      if (!Array.isArray(e) && e.gcsUri && e.gcsUri.length > 0) return e.gcsUri[0];
    }
    getBigqueryUri(e) {
      if (typeof e == `string`) return e.startsWith(`bq://`) ? e : void 0;
      if (!Array.isArray(e)) return e.bigqueryUri;
    }
    formatDestination(e, t) {
      let n = t ? Object.assign({}, t) : {},
        r = Date.now().toString();
      if (((n.displayName ||= `genaiBatchJob_${r}`), n.dest === void 0)) {
        let t = this.getGcsUri(e),
          i = this.getBigqueryUri(e);
        if (t)
          t.endsWith(`.jsonl`) ? (n.dest = `${t.slice(0, -6)}/dest`) : (n.dest = `${t}_dest_${r}`);
        else if (i) n.dest = `${i}_dest_${r}`;
        else throw Error(`Unsupported source for Vertex AI: No GCS or BigQuery URI found.`);
      }
      return n;
    }
    async createInternal(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI()) {
        let i = Fn(this.apiClient, e);
        return (
          (n = p(`batchPredictionJobs`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `POST`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) => e.json())),
          t.then((e) => Sn(e))
        );
      } else {
        let i = Pn(this.apiClient, e);
        return (
          (n = p(`{model}:batchGenerateContent`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `POST`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) => e.json())),
          t.then((e) => xn(e))
        );
      }
    }
    async createEmbeddingsInternal(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI())
        throw Error(`This method is only supported by the Gemini Developer API.`);
      {
        let i = Ln(this.apiClient, e);
        return (
          (n = p(`{model}:asyncBatchEmbedContent`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `POST`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) => e.json())),
          t.then((e) => xn(e))
        );
      }
    }
    async get(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI()) {
        let i = Zn(this.apiClient, e);
        return (
          (n = p(`batchPredictionJobs/{name}`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `GET`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) => e.json())),
          t.then((e) => Sn(e))
        );
      } else {
        let i = Xn(this.apiClient, e);
        return (
          (n = p(`batches/{name}`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `GET`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) => e.json())),
          t.then((e) => xn(e))
        );
      }
    }
    async cancel(e) {
      let t = ``,
        n = {};
      if (this.apiClient.isVertexAI()) {
        let r = On(this.apiClient, e);
        ((t = p(`batchPredictionJobs/{name}:cancel`, r._url)),
          (n = r._query),
          delete r._url,
          delete r._query,
          await this.apiClient.request({
            path: t,
            queryParams: n,
            body: JSON.stringify(r),
            httpMethod: `POST`,
            httpOptions: e.config?.httpOptions,
            abortSignal: e.config?.abortSignal,
          }));
      } else {
        let r = Dn(this.apiClient, e);
        ((t = p(`batches/{name}:cancel`, r._url)),
          (n = r._query),
          delete r._url,
          delete r._query,
          await this.apiClient.request({
            path: t,
            queryParams: n,
            body: JSON.stringify(r),
            httpMethod: `POST`,
            httpOptions: e.config?.httpOptions,
            abortSignal: e.config?.abortSignal,
          }));
      }
    }
    async listInternal(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI()) {
        let i = or(e);
        return (
          (n = p(`batchPredictionJobs`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `GET`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) =>
              e.json().then((t) => {
                let n = t;
                return ((n.sdkHttpResponse = { headers: e.headers }), n);
              }),
            )),
          t.then((e) => {
            let t = cr(e),
              n = new Pt();
            return (Object.assign(n, t), n);
          })
        );
      } else {
        let i = ar(e);
        return (
          (n = p(`batches`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `GET`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) =>
              e.json().then((t) => {
                let n = t;
                return ((n.sdkHttpResponse = { headers: e.headers }), n);
              }),
            )),
          t.then((e) => {
            let t = sr(e),
              n = new Pt();
            return (Object.assign(n, t), n);
          })
        );
      }
    }
    async delete(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI()) {
        let i = zn(this.apiClient, e);
        return (
          (n = p(`batchPredictionJobs/{name}`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `DELETE`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) =>
              e.json().then((t) => {
                let n = t;
                return ((n.sdkHttpResponse = { headers: e.headers }), n);
              }),
            )),
          t.then((e) => Vn(e))
        );
      } else {
        let i = Rn(this.apiClient, e);
        return (
          (n = p(`batches/{name}`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `DELETE`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) =>
              e.json().then((t) => {
                let n = t;
                return ((n.sdkHttpResponse = { headers: e.headers }), n);
              }),
            )),
          t.then((e) => Bn(e))
        );
      }
    }
  };
function mr(e) {
  let t = {},
    n = h(e, [`apiKey`]);
  if ((n != null && m(t, [`apiKey`], n), h(e, [`apiKeyConfig`]) !== void 0))
    throw Error(`apiKeyConfig parameter is not supported in Gemini API.`);
  if (h(e, [`authType`]) !== void 0)
    throw Error(`authType parameter is not supported in Gemini API.`);
  if (h(e, [`googleServiceAccountConfig`]) !== void 0)
    throw Error(`googleServiceAccountConfig parameter is not supported in Gemini API.`);
  if (h(e, [`httpBasicAuthConfig`]) !== void 0)
    throw Error(`httpBasicAuthConfig parameter is not supported in Gemini API.`);
  if (h(e, [`oauthConfig`]) !== void 0)
    throw Error(`oauthConfig parameter is not supported in Gemini API.`);
  if (h(e, [`oidcConfig`]) !== void 0)
    throw Error(`oidcConfig parameter is not supported in Gemini API.`);
  return t;
}
function hr(e) {
  let t = {},
    n = h(e, [`data`]);
  if ((n != null && m(t, [`data`], n), h(e, [`displayName`]) !== void 0))
    throw Error(`displayName parameter is not supported in Gemini API.`);
  let r = h(e, [`mimeType`]);
  return (r != null && m(t, [`mimeType`], r), t);
}
function gr(e) {
  let t = {},
    n = h(e, [`parts`]);
  if (n != null) {
    let e = n;
    (Array.isArray(e) && (e = e.map((e) => Br(e))), m(t, [`parts`], e));
  }
  let r = h(e, [`role`]);
  return (r != null && m(t, [`role`], r), t);
}
function _r(e) {
  let t = {},
    n = h(e, [`parts`]);
  if (n != null) {
    let e = n;
    (Array.isArray(e) && (e = e.map((e) => Vr(e))), m(t, [`parts`], e));
  }
  let r = h(e, [`role`]);
  return (r != null && m(t, [`role`], r), t);
}
function vr(e, t) {
  let n = {},
    r = h(e, [`ttl`]);
  t !== void 0 && r != null && m(t, [`ttl`], r);
  let i = h(e, [`expireTime`]);
  t !== void 0 && i != null && m(t, [`expireTime`], i);
  let a = h(e, [`displayName`]);
  t !== void 0 && a != null && m(t, [`displayName`], a);
  let o = h(e, [`contents`]);
  if (t !== void 0 && o != null) {
    let e = I(o);
    (Array.isArray(e) && (e = e.map((e) => gr(e))), m(t, [`contents`], e));
  }
  let s = h(e, [`systemInstruction`]);
  t !== void 0 && s != null && m(t, [`systemInstruction`], gr(F(s)));
  let c = h(e, [`tools`]);
  if (t !== void 0 && c != null) {
    let e = c;
    (Array.isArray(e) && (e = e.map((e) => Wr(e))), m(t, [`tools`], e));
  }
  let l = h(e, [`toolConfig`]);
  if ((t !== void 0 && l != null && m(t, [`toolConfig`], Hr(l)), h(e, [`kmsKeyName`]) !== void 0))
    throw Error(`kmsKeyName parameter is not supported in Gemini API.`);
  return n;
}
function yr(e, t) {
  let n = {},
    r = h(e, [`ttl`]);
  t !== void 0 && r != null && m(t, [`ttl`], r);
  let i = h(e, [`expireTime`]);
  t !== void 0 && i != null && m(t, [`expireTime`], i);
  let a = h(e, [`displayName`]);
  t !== void 0 && a != null && m(t, [`displayName`], a);
  let o = h(e, [`contents`]);
  if (t !== void 0 && o != null) {
    let e = I(o);
    (Array.isArray(e) && (e = e.map((e) => _r(e))), m(t, [`contents`], e));
  }
  let s = h(e, [`systemInstruction`]);
  t !== void 0 && s != null && m(t, [`systemInstruction`], _r(F(s)));
  let c = h(e, [`tools`]);
  if (t !== void 0 && c != null) {
    let e = c;
    (Array.isArray(e) && (e = e.map((e) => Gr(e))), m(t, [`tools`], e));
  }
  let l = h(e, [`toolConfig`]);
  t !== void 0 && l != null && m(t, [`toolConfig`], Ur(l));
  let u = h(e, [`kmsKeyName`]);
  return (t !== void 0 && u != null && m(t, [`encryption_spec`, `kmsKeyName`], u), n);
}
function br(e, t) {
  let n = {},
    r = h(t, [`model`]);
  r != null && m(n, [`model`], Rt(e, r));
  let i = h(t, [`config`]);
  return (i != null && vr(i, n), n);
}
function xr(e, t) {
  let n = {},
    r = h(t, [`model`]);
  r != null && m(n, [`model`], Rt(e, r));
  let i = h(t, [`config`]);
  return (i != null && yr(i, n), n);
}
function Sr(e, t) {
  let n = {},
    r = h(t, [`name`]);
  return (r != null && m(n, [`_url`, `name`], B(e, r)), n);
}
function Cr(e, t) {
  let n = {},
    r = h(t, [`name`]);
  return (r != null && m(n, [`_url`, `name`], B(e, r)), n);
}
function wr(e) {
  let t = {},
    n = h(e, [`sdkHttpResponse`]);
  return (n != null && m(t, [`sdkHttpResponse`], n), t);
}
function Tr(e) {
  let t = {},
    n = h(e, [`sdkHttpResponse`]);
  return (n != null && m(t, [`sdkHttpResponse`], n), t);
}
function Er(e) {
  let t = {};
  if (h(e, [`displayName`]) !== void 0)
    throw Error(`displayName parameter is not supported in Gemini API.`);
  let n = h(e, [`fileUri`]);
  n != null && m(t, [`fileUri`], n);
  let r = h(e, [`mimeType`]);
  return (r != null && m(t, [`mimeType`], r), t);
}
function Dr(e) {
  let t = {},
    n = h(e, [`id`]);
  n != null && m(t, [`id`], n);
  let r = h(e, [`args`]);
  r != null && m(t, [`args`], r);
  let i = h(e, [`name`]);
  if ((i != null && m(t, [`name`], i), h(e, [`partialArgs`]) !== void 0))
    throw Error(`partialArgs parameter is not supported in Gemini API.`);
  if (h(e, [`willContinue`]) !== void 0)
    throw Error(`willContinue parameter is not supported in Gemini API.`);
  return t;
}
function Or(e) {
  let t = {},
    n = h(e, [`allowedFunctionNames`]);
  n != null && m(t, [`allowedFunctionNames`], n);
  let r = h(e, [`mode`]);
  if ((r != null && m(t, [`mode`], r), h(e, [`streamFunctionCallArguments`]) !== void 0))
    throw Error(`streamFunctionCallArguments parameter is not supported in Gemini API.`);
  return t;
}
function kr(e) {
  let t = {},
    n = h(e, [`description`]);
  n != null && m(t, [`description`], n);
  let r = h(e, [`name`]);
  r != null && m(t, [`name`], r);
  let i = h(e, [`parameters`]);
  i != null && m(t, [`parameters`], i);
  let a = h(e, [`parametersJsonSchema`]);
  a != null && m(t, [`parametersJsonSchema`], a);
  let o = h(e, [`response`]);
  o != null && m(t, [`response`], o);
  let s = h(e, [`responseJsonSchema`]);
  if ((s != null && m(t, [`responseJsonSchema`], s), h(e, [`behavior`]) !== void 0))
    throw Error(`behavior parameter is not supported in Vertex AI.`);
  return t;
}
function Ar(e, t) {
  let n = {},
    r = h(t, [`name`]);
  return (r != null && m(n, [`_url`, `name`], B(e, r)), n);
}
function jr(e, t) {
  let n = {},
    r = h(t, [`name`]);
  return (r != null && m(n, [`_url`, `name`], B(e, r)), n);
}
function Mr(e) {
  let t = {},
    n = h(e, [`authConfig`]);
  n != null && m(t, [`authConfig`], mr(n));
  let r = h(e, [`enableWidget`]);
  return (r != null && m(t, [`enableWidget`], r), t);
}
function Nr(e) {
  let t = {},
    n = h(e, [`searchTypes`]);
  if ((n != null && m(t, [`searchTypes`], n), h(e, [`blockingConfidence`]) !== void 0))
    throw Error(`blockingConfidence parameter is not supported in Gemini API.`);
  if (h(e, [`excludeDomains`]) !== void 0)
    throw Error(`excludeDomains parameter is not supported in Gemini API.`);
  let r = h(e, [`timeRangeFilter`]);
  return (r != null && m(t, [`timeRangeFilter`], r), t);
}
function Pr(e, t) {
  let n = {},
    r = h(e, [`pageSize`]);
  t !== void 0 && r != null && m(t, [`_query`, `pageSize`], r);
  let i = h(e, [`pageToken`]);
  return (t !== void 0 && i != null && m(t, [`_query`, `pageToken`], i), n);
}
function Fr(e, t) {
  let n = {},
    r = h(e, [`pageSize`]);
  t !== void 0 && r != null && m(t, [`_query`, `pageSize`], r);
  let i = h(e, [`pageToken`]);
  return (t !== void 0 && i != null && m(t, [`_query`, `pageToken`], i), n);
}
function Ir(e) {
  let t = {},
    n = h(e, [`config`]);
  return (n != null && Pr(n, t), t);
}
function Lr(e) {
  let t = {},
    n = h(e, [`config`]);
  return (n != null && Fr(n, t), t);
}
function Rr(e) {
  let t = {},
    n = h(e, [`sdkHttpResponse`]);
  n != null && m(t, [`sdkHttpResponse`], n);
  let r = h(e, [`nextPageToken`]);
  r != null && m(t, [`nextPageToken`], r);
  let i = h(e, [`cachedContents`]);
  if (i != null) {
    let e = i;
    (Array.isArray(e) && (e = e.map((e) => e)), m(t, [`cachedContents`], e));
  }
  return t;
}
function zr(e) {
  let t = {},
    n = h(e, [`sdkHttpResponse`]);
  n != null && m(t, [`sdkHttpResponse`], n);
  let r = h(e, [`nextPageToken`]);
  r != null && m(t, [`nextPageToken`], r);
  let i = h(e, [`cachedContents`]);
  if (i != null) {
    let e = i;
    (Array.isArray(e) && (e = e.map((e) => e)), m(t, [`cachedContents`], e));
  }
  return t;
}
function Br(e) {
  let t = {},
    n = h(e, [`mediaResolution`]);
  n != null && m(t, [`mediaResolution`], n);
  let r = h(e, [`codeExecutionResult`]);
  r != null && m(t, [`codeExecutionResult`], r);
  let i = h(e, [`executableCode`]);
  i != null && m(t, [`executableCode`], i);
  let a = h(e, [`fileData`]);
  a != null && m(t, [`fileData`], Er(a));
  let o = h(e, [`functionCall`]);
  o != null && m(t, [`functionCall`], Dr(o));
  let s = h(e, [`functionResponse`]);
  s != null && m(t, [`functionResponse`], s);
  let c = h(e, [`inlineData`]);
  c != null && m(t, [`inlineData`], hr(c));
  let l = h(e, [`text`]);
  l != null && m(t, [`text`], l);
  let u = h(e, [`thought`]);
  u != null && m(t, [`thought`], u);
  let d = h(e, [`thoughtSignature`]);
  d != null && m(t, [`thoughtSignature`], d);
  let f = h(e, [`videoMetadata`]);
  f != null && m(t, [`videoMetadata`], f);
  let p = h(e, [`toolCall`]);
  p != null && m(t, [`toolCall`], p);
  let g = h(e, [`toolResponse`]);
  return (g != null && m(t, [`toolResponse`], g), t);
}
function Vr(e) {
  let t = {},
    n = h(e, [`mediaResolution`]);
  n != null && m(t, [`mediaResolution`], n);
  let r = h(e, [`codeExecutionResult`]);
  r != null && m(t, [`codeExecutionResult`], r);
  let i = h(e, [`executableCode`]);
  i != null && m(t, [`executableCode`], i);
  let a = h(e, [`fileData`]);
  a != null && m(t, [`fileData`], a);
  let o = h(e, [`functionCall`]);
  o != null && m(t, [`functionCall`], o);
  let s = h(e, [`functionResponse`]);
  s != null && m(t, [`functionResponse`], s);
  let c = h(e, [`inlineData`]);
  c != null && m(t, [`inlineData`], c);
  let l = h(e, [`text`]);
  l != null && m(t, [`text`], l);
  let u = h(e, [`thought`]);
  u != null && m(t, [`thought`], u);
  let d = h(e, [`thoughtSignature`]);
  d != null && m(t, [`thoughtSignature`], d);
  let f = h(e, [`videoMetadata`]);
  if ((f != null && m(t, [`videoMetadata`], f), h(e, [`toolCall`]) !== void 0))
    throw Error(`toolCall parameter is not supported in Vertex AI.`);
  if (h(e, [`toolResponse`]) !== void 0)
    throw Error(`toolResponse parameter is not supported in Vertex AI.`);
  return t;
}
function Hr(e) {
  let t = {},
    n = h(e, [`retrievalConfig`]);
  n != null && m(t, [`retrievalConfig`], n);
  let r = h(e, [`functionCallingConfig`]);
  r != null && m(t, [`functionCallingConfig`], Or(r));
  let i = h(e, [`includeServerSideToolInvocations`]);
  return (i != null && m(t, [`includeServerSideToolInvocations`], i), t);
}
function Ur(e) {
  let t = {},
    n = h(e, [`retrievalConfig`]);
  n != null && m(t, [`retrievalConfig`], n);
  let r = h(e, [`functionCallingConfig`]);
  if (
    (r != null && m(t, [`functionCallingConfig`], r),
    h(e, [`includeServerSideToolInvocations`]) !== void 0)
  )
    throw Error(`includeServerSideToolInvocations parameter is not supported in Vertex AI.`);
  return t;
}
function Wr(e) {
  let t = {};
  if (h(e, [`retrieval`]) !== void 0)
    throw Error(`retrieval parameter is not supported in Gemini API.`);
  let n = h(e, [`computerUse`]);
  n != null && m(t, [`computerUse`], n);
  let r = h(e, [`fileSearch`]);
  r != null && m(t, [`fileSearch`], r);
  let i = h(e, [`googleSearch`]);
  i != null && m(t, [`googleSearch`], Nr(i));
  let a = h(e, [`googleMaps`]);
  a != null && m(t, [`googleMaps`], Mr(a));
  let o = h(e, [`codeExecution`]);
  if ((o != null && m(t, [`codeExecution`], o), h(e, [`enterpriseWebSearch`]) !== void 0))
    throw Error(`enterpriseWebSearch parameter is not supported in Gemini API.`);
  let s = h(e, [`functionDeclarations`]);
  if (s != null) {
    let e = s;
    (Array.isArray(e) && (e = e.map((e) => e)), m(t, [`functionDeclarations`], e));
  }
  let c = h(e, [`googleSearchRetrieval`]);
  if ((c != null && m(t, [`googleSearchRetrieval`], c), h(e, [`parallelAiSearch`]) !== void 0))
    throw Error(`parallelAiSearch parameter is not supported in Gemini API.`);
  let l = h(e, [`urlContext`]);
  l != null && m(t, [`urlContext`], l);
  let u = h(e, [`mcpServers`]);
  if (u != null) {
    let e = u;
    (Array.isArray(e) && (e = e.map((e) => e)), m(t, [`mcpServers`], e));
  }
  return t;
}
function Gr(e) {
  let t = {},
    n = h(e, [`retrieval`]);
  n != null && m(t, [`retrieval`], n);
  let r = h(e, [`computerUse`]);
  if ((r != null && m(t, [`computerUse`], r), h(e, [`fileSearch`]) !== void 0))
    throw Error(`fileSearch parameter is not supported in Vertex AI.`);
  let i = h(e, [`googleSearch`]);
  i != null && m(t, [`googleSearch`], i);
  let a = h(e, [`googleMaps`]);
  a != null && m(t, [`googleMaps`], a);
  let o = h(e, [`codeExecution`]);
  o != null && m(t, [`codeExecution`], o);
  let s = h(e, [`enterpriseWebSearch`]);
  s != null && m(t, [`enterpriseWebSearch`], s);
  let c = h(e, [`functionDeclarations`]);
  if (c != null) {
    let e = c;
    (Array.isArray(e) && (e = e.map((e) => kr(e))), m(t, [`functionDeclarations`], e));
  }
  let l = h(e, [`googleSearchRetrieval`]);
  l != null && m(t, [`googleSearchRetrieval`], l);
  let u = h(e, [`parallelAiSearch`]);
  u != null && m(t, [`parallelAiSearch`], u);
  let d = h(e, [`urlContext`]);
  if ((d != null && m(t, [`urlContext`], d), h(e, [`mcpServers`]) !== void 0))
    throw Error(`mcpServers parameter is not supported in Vertex AI.`);
  return t;
}
function Kr(e, t) {
  let n = {},
    r = h(e, [`ttl`]);
  t !== void 0 && r != null && m(t, [`ttl`], r);
  let i = h(e, [`expireTime`]);
  return (t !== void 0 && i != null && m(t, [`expireTime`], i), n);
}
function qr(e, t) {
  let n = {},
    r = h(e, [`ttl`]);
  t !== void 0 && r != null && m(t, [`ttl`], r);
  let i = h(e, [`expireTime`]);
  return (t !== void 0 && i != null && m(t, [`expireTime`], i), n);
}
function Jr(e, t) {
  let n = {},
    r = h(t, [`name`]);
  r != null && m(n, [`_url`, `name`], B(e, r));
  let i = h(t, [`config`]);
  return (i != null && Kr(i, n), n);
}
function Yr(e, t) {
  let n = {},
    r = h(t, [`name`]);
  r != null && m(n, [`_url`, `name`], B(e, r));
  let i = h(t, [`config`]);
  return (i != null && qr(i, n), n);
}
var Xr = class extends f {
  constructor(e) {
    (super(),
      (this.apiClient = e),
      (this.list = async (e = {}) =>
        new U(
          H.PAGED_ITEM_CACHED_CONTENTS,
          (e) => this.listInternal(e),
          await this.listInternal(e),
          e,
        )));
  }
  async create(e) {
    let t,
      n = ``,
      r = {};
    if (this.apiClient.isVertexAI()) {
      let i = xr(this.apiClient, e);
      return (
        (n = p(`cachedContents`, i._url)),
        (r = i._query),
        delete i._url,
        delete i._query,
        (t = this.apiClient
          .request({
            path: n,
            queryParams: r,
            body: JSON.stringify(i),
            httpMethod: `POST`,
            httpOptions: e.config?.httpOptions,
            abortSignal: e.config?.abortSignal,
          })
          .then((e) => e.json())),
        t.then((e) => e)
      );
    } else {
      let i = br(this.apiClient, e);
      return (
        (n = p(`cachedContents`, i._url)),
        (r = i._query),
        delete i._url,
        delete i._query,
        (t = this.apiClient
          .request({
            path: n,
            queryParams: r,
            body: JSON.stringify(i),
            httpMethod: `POST`,
            httpOptions: e.config?.httpOptions,
            abortSignal: e.config?.abortSignal,
          })
          .then((e) => e.json())),
        t.then((e) => e)
      );
    }
  }
  async get(e) {
    let t,
      n = ``,
      r = {};
    if (this.apiClient.isVertexAI()) {
      let i = jr(this.apiClient, e);
      return (
        (n = p(`{name}`, i._url)),
        (r = i._query),
        delete i._url,
        delete i._query,
        (t = this.apiClient
          .request({
            path: n,
            queryParams: r,
            body: JSON.stringify(i),
            httpMethod: `GET`,
            httpOptions: e.config?.httpOptions,
            abortSignal: e.config?.abortSignal,
          })
          .then((e) => e.json())),
        t.then((e) => e)
      );
    } else {
      let i = Ar(this.apiClient, e);
      return (
        (n = p(`{name}`, i._url)),
        (r = i._query),
        delete i._url,
        delete i._query,
        (t = this.apiClient
          .request({
            path: n,
            queryParams: r,
            body: JSON.stringify(i),
            httpMethod: `GET`,
            httpOptions: e.config?.httpOptions,
            abortSignal: e.config?.abortSignal,
          })
          .then((e) => e.json())),
        t.then((e) => e)
      );
    }
  }
  async delete(e) {
    let t,
      n = ``,
      r = {};
    if (this.apiClient.isVertexAI()) {
      let i = Cr(this.apiClient, e);
      return (
        (n = p(`{name}`, i._url)),
        (r = i._query),
        delete i._url,
        delete i._query,
        (t = this.apiClient
          .request({
            path: n,
            queryParams: r,
            body: JSON.stringify(i),
            httpMethod: `DELETE`,
            httpOptions: e.config?.httpOptions,
            abortSignal: e.config?.abortSignal,
          })
          .then((e) =>
            e.json().then((t) => {
              let n = t;
              return ((n.sdkHttpResponse = { headers: e.headers }), n);
            }),
          )),
        t.then((e) => {
          let t = Tr(e),
            n = new wt();
          return (Object.assign(n, t), n);
        })
      );
    } else {
      let i = Sr(this.apiClient, e);
      return (
        (n = p(`{name}`, i._url)),
        (r = i._query),
        delete i._url,
        delete i._query,
        (t = this.apiClient
          .request({
            path: n,
            queryParams: r,
            body: JSON.stringify(i),
            httpMethod: `DELETE`,
            httpOptions: e.config?.httpOptions,
            abortSignal: e.config?.abortSignal,
          })
          .then((e) =>
            e.json().then((t) => {
              let n = t;
              return ((n.sdkHttpResponse = { headers: e.headers }), n);
            }),
          )),
        t.then((e) => {
          let t = wr(e),
            n = new wt();
          return (Object.assign(n, t), n);
        })
      );
    }
  }
  async update(e) {
    let t,
      n = ``,
      r = {};
    if (this.apiClient.isVertexAI()) {
      let i = Yr(this.apiClient, e);
      return (
        (n = p(`{name}`, i._url)),
        (r = i._query),
        delete i._url,
        delete i._query,
        (t = this.apiClient
          .request({
            path: n,
            queryParams: r,
            body: JSON.stringify(i),
            httpMethod: `PATCH`,
            httpOptions: e.config?.httpOptions,
            abortSignal: e.config?.abortSignal,
          })
          .then((e) => e.json())),
        t.then((e) => e)
      );
    } else {
      let i = Jr(this.apiClient, e);
      return (
        (n = p(`{name}`, i._url)),
        (r = i._query),
        delete i._url,
        delete i._query,
        (t = this.apiClient
          .request({
            path: n,
            queryParams: r,
            body: JSON.stringify(i),
            httpMethod: `PATCH`,
            httpOptions: e.config?.httpOptions,
            abortSignal: e.config?.abortSignal,
          })
          .then((e) => e.json())),
        t.then((e) => e)
      );
    }
  }
  async listInternal(e) {
    let t,
      n = ``,
      r = {};
    if (this.apiClient.isVertexAI()) {
      let i = Lr(e);
      return (
        (n = p(`cachedContents`, i._url)),
        (r = i._query),
        delete i._url,
        delete i._query,
        (t = this.apiClient
          .request({
            path: n,
            queryParams: r,
            body: JSON.stringify(i),
            httpMethod: `GET`,
            httpOptions: e.config?.httpOptions,
            abortSignal: e.config?.abortSignal,
          })
          .then((e) =>
            e.json().then((t) => {
              let n = t;
              return ((n.sdkHttpResponse = { headers: e.headers }), n);
            }),
          )),
        t.then((e) => {
          let t = zr(e),
            n = new Tt();
          return (Object.assign(n, t), n);
        })
      );
    } else {
      let i = Ir(e);
      return (
        (n = p(`cachedContents`, i._url)),
        (r = i._query),
        delete i._url,
        delete i._query,
        (t = this.apiClient
          .request({
            path: n,
            queryParams: r,
            body: JSON.stringify(i),
            httpMethod: `GET`,
            httpOptions: e.config?.httpOptions,
            abortSignal: e.config?.abortSignal,
          })
          .then((e) =>
            e.json().then((t) => {
              let n = t;
              return ((n.sdkHttpResponse = { headers: e.headers }), n);
            }),
          )),
        t.then((e) => {
          let t = Rr(e),
            n = new Tt();
          return (Object.assign(n, t), n);
        })
      );
    }
  }
};
function Zr(e, t) {
  var n = {};
  for (var r in e) Object.prototype.hasOwnProperty.call(e, r) && t.indexOf(r) < 0 && (n[r] = e[r]);
  if (e != null && typeof Object.getOwnPropertySymbols == `function`)
    for (var i = 0, r = Object.getOwnPropertySymbols(e); i < r.length; i++)
      t.indexOf(r[i]) < 0 &&
        Object.prototype.propertyIsEnumerable.call(e, r[i]) &&
        (n[r[i]] = e[r[i]]);
  return n;
}
function Qr(e) {
  var t = typeof Symbol == `function` && Symbol.iterator,
    n = t && e[t],
    r = 0;
  if (n) return n.call(e);
  if (e && typeof e.length == `number`)
    return {
      next: function () {
        return (e && r >= e.length && (e = void 0), { value: e && e[r++], done: !e });
      },
    };
  throw TypeError(t ? `Object is not iterable.` : `Symbol.iterator is not defined.`);
}
function W(e) {
  return this instanceof W ? ((this.v = e), this) : new W(e);
}
function G(e, t, n) {
  if (!Symbol.asyncIterator) throw TypeError(`Symbol.asyncIterator is not defined.`);
  var r = n.apply(e, t || []),
    i,
    a = [];
  return (
    (i = Object.create((typeof AsyncIterator == `function` ? AsyncIterator : Object).prototype)),
    s(`next`),
    s(`throw`),
    s(`return`, o),
    (i[Symbol.asyncIterator] = function () {
      return this;
    }),
    i
  );
  function o(e) {
    return function (t) {
      return Promise.resolve(t).then(e, d);
    };
  }
  function s(e, t) {
    r[e] &&
      ((i[e] = function (t) {
        return new Promise(function (n, r) {
          a.push([e, t, n, r]) > 1 || c(e, t);
        });
      }),
      t && (i[e] = t(i[e])));
  }
  function c(e, t) {
    try {
      l(r[e](t));
    } catch (e) {
      f(a[0][3], e);
    }
  }
  function l(e) {
    e.value instanceof W ? Promise.resolve(e.value.v).then(u, d) : f(a[0][2], e);
  }
  function u(e) {
    c(`next`, e);
  }
  function d(e) {
    c(`throw`, e);
  }
  function f(e, t) {
    (e(t), a.shift(), a.length && c(a[0][0], a[0][1]));
  }
}
function K(e) {
  if (!Symbol.asyncIterator) throw TypeError(`Symbol.asyncIterator is not defined.`);
  var t = e[Symbol.asyncIterator],
    n;
  return t
    ? t.call(e)
    : ((e = typeof Qr == `function` ? Qr(e) : e[Symbol.iterator]()),
      (n = {}),
      r(`next`),
      r(`throw`),
      r(`return`),
      (n[Symbol.asyncIterator] = function () {
        return this;
      }),
      n);
  function r(t) {
    n[t] =
      e[t] &&
      function (n) {
        return new Promise(function (r, a) {
          ((n = e[t](n)), i(r, a, n.done, n.value));
        });
      };
  }
  function i(e, t, n, r) {
    Promise.resolve(r).then(function (t) {
      e({ value: t, done: n });
    }, t);
  }
}
function $r(e) {
  if (e.candidates == null || e.candidates.length === 0) return !1;
  let t = e.candidates[0]?.content;
  return t === void 0 ? !1 : ei(t);
}
function ei(e) {
  if (e.parts === void 0 || e.parts.length === 0) return !1;
  for (let t of e.parts) if (t === void 0 || Object.keys(t).length === 0) return !1;
  return !0;
}
function ti(e) {
  if (e.length !== 0) {
    for (let t of e)
      if (t.role !== `user` && t.role !== `model`)
        throw Error(`Role must be user or model, but got ${t.role}.`);
  }
}
function ni(e) {
  if (e === void 0 || e.length === 0) return [];
  let t = [],
    n = e.length,
    r = 0;
  for (; r < n; )
    if (e[r].role === `user`) (t.push(e[r]), r++);
    else {
      let i = [],
        a = !0;
      for (; r < n && e[r].role === `model`; ) (i.push(e[r]), a && !ei(e[r]) && (a = !1), r++);
      a ? t.push(...i) : t.pop();
    }
  return t;
}
var ri = class {
    constructor(e, t) {
      ((this.modelsModule = e), (this.apiClient = t));
    }
    create(e) {
      return new ii(
        this.apiClient,
        this.modelsModule,
        e.model,
        e.config,
        structuredClone(e.history),
      );
    }
  },
  ii = class {
    constructor(e, t, n, r = {}, i = []) {
      ((this.apiClient = e),
        (this.modelsModule = t),
        (this.model = n),
        (this.config = r),
        (this.history = i),
        (this.sendPromise = Promise.resolve()),
        ti(i));
    }
    async sendMessage(e) {
      await this.sendPromise;
      let t = F(e.message),
        n = this.modelsModule.generateContent({
          model: this.model,
          contents: this.getHistory(!0).concat(t),
          config: e.config ?? this.config,
        });
      return (
        (this.sendPromise = (async () => {
          let e = await n,
            r = e.candidates?.[0]?.content,
            i = e.automaticFunctionCallingHistory,
            a = this.getHistory(!0).length,
            o = [];
          i != null && (o = i.slice(a) ?? []);
          let s = r ? [r] : [];
          this.recordHistory(t, s, o);
        })()),
        await this.sendPromise.catch(() => {
          this.sendPromise = Promise.resolve();
        }),
        n
      );
    }
    async sendMessageStream(e) {
      await this.sendPromise;
      let t = F(e.message),
        n = this.modelsModule.generateContentStream({
          model: this.model,
          contents: this.getHistory(!0).concat(t),
          config: e.config ?? this.config,
        });
      this.sendPromise = n.then(() => void 0).catch(() => void 0);
      let r = await n;
      return this.processStreamResponse(r, t);
    }
    getHistory(e = !1) {
      let t = e ? ni(this.history) : this.history;
      return structuredClone(t);
    }
    processStreamResponse(e, t) {
      return G(this, arguments, function* () {
        var n, r, i, a;
        let o = [];
        try {
          for (var s = !0, c = K(e), l; (l = yield W(c.next())), (n = l.done), !n; s = !0) {
            ((a = l.value), (s = !1));
            let e = a;
            if ($r(e)) {
              let t = e.candidates?.[0]?.content;
              t !== void 0 && o.push(t);
            }
            yield yield W(e);
          }
        } catch (e) {
          r = { error: e };
        } finally {
          try {
            !s && !n && (i = c.return) && (yield W(i.call(c)));
          } finally {
            if (r) throw r.error;
          }
        }
        this.recordHistory(t, o);
      });
    }
    recordHistory(e, t, n) {
      let r = [];
      (t.length > 0 && t.every((e) => e.role !== void 0)
        ? (r = t)
        : r.push({ role: `model`, parts: [] }),
        n && n.length > 0 ? this.history.push(...ni(n)) : this.history.push(e),
        this.history.push(...r));
    }
  },
  ai = class e extends Error {
    constructor(t) {
      (super(t.message),
        (this.name = `ApiError`),
        (this.status = t.status),
        Object.setPrototypeOf(this, e.prototype));
    }
  };
function oi(e) {
  let t = {},
    n = h(e, [`file`]);
  return (n != null && m(t, [`file`], n), t);
}
function si(e) {
  let t = {},
    n = h(e, [`sdkHttpResponse`]);
  return (n != null && m(t, [`sdkHttpResponse`], n), t);
}
function ci(e) {
  let t = {},
    n = h(e, [`name`]);
  return (n != null && m(t, [`_url`, `file`], an(n)), t);
}
function li(e) {
  let t = {},
    n = h(e, [`sdkHttpResponse`]);
  return (n != null && m(t, [`sdkHttpResponse`], n), t);
}
function ui(e) {
  let t = {},
    n = h(e, [`name`]);
  return (n != null && m(t, [`_url`, `file`], an(n)), t);
}
function di(e) {
  let t = {},
    n = h(e, [`uris`]);
  return (n != null && m(t, [`uris`], n), t);
}
function fi(e, t) {
  let n = {},
    r = h(e, [`pageSize`]);
  t !== void 0 && r != null && m(t, [`_query`, `pageSize`], r);
  let i = h(e, [`pageToken`]);
  return (t !== void 0 && i != null && m(t, [`_query`, `pageToken`], i), n);
}
function pi(e) {
  let t = {},
    n = h(e, [`config`]);
  return (n != null && fi(n, t), t);
}
function mi(e) {
  let t = {},
    n = h(e, [`sdkHttpResponse`]);
  n != null && m(t, [`sdkHttpResponse`], n);
  let r = h(e, [`nextPageToken`]);
  r != null && m(t, [`nextPageToken`], r);
  let i = h(e, [`files`]);
  if (i != null) {
    let e = i;
    (Array.isArray(e) && (e = e.map((e) => e)), m(t, [`files`], e));
  }
  return t;
}
function hi(e) {
  let t = {},
    n = h(e, [`sdkHttpResponse`]);
  n != null && m(t, [`sdkHttpResponse`], n);
  let r = h(e, [`files`]);
  if (r != null) {
    let e = r;
    (Array.isArray(e) && (e = e.map((e) => e)), m(t, [`files`], e));
  }
  return t;
}
var gi = class extends f {
  constructor(e) {
    (super(),
      (this.apiClient = e),
      (this.list = async (e = {}) =>
        new U(H.PAGED_ITEM_FILES, (e) => this.listInternal(e), await this.listInternal(e), e)));
  }
  async upload(e) {
    if (this.apiClient.isVertexAI())
      throw Error(
        `Vertex AI does not support uploading files. You can share files through a GCS bucket.`,
      );
    return this.apiClient.uploadFile(e.file, e.config).then((e) => e);
  }
  async download(e) {
    await this.apiClient.downloadFile(e);
  }
  async registerFiles(e) {
    throw Error(`registerFiles is only supported in Node.js environments.`);
  }
  async _registerFiles(e) {
    return this.registerFilesInternal(e);
  }
  async listInternal(e) {
    let t,
      n = ``,
      r = {};
    if (this.apiClient.isVertexAI())
      throw Error(`This method is only supported by the Gemini Developer API.`);
    {
      let i = pi(e);
      return (
        (n = p(`files`, i._url)),
        (r = i._query),
        delete i._url,
        delete i._query,
        (t = this.apiClient
          .request({
            path: n,
            queryParams: r,
            body: JSON.stringify(i),
            httpMethod: `GET`,
            httpOptions: e.config?.httpOptions,
            abortSignal: e.config?.abortSignal,
          })
          .then((e) =>
            e.json().then((t) => {
              let n = t;
              return ((n.sdkHttpResponse = { headers: e.headers }), n);
            }),
          )),
        t.then((e) => {
          let t = mi(e),
            n = new At();
          return (Object.assign(n, t), n);
        })
      );
    }
  }
  async createInternal(e) {
    let t,
      n = ``,
      r = {};
    if (this.apiClient.isVertexAI())
      throw Error(`This method is only supported by the Gemini Developer API.`);
    {
      let i = oi(e);
      return (
        (n = p(`upload/v1beta/files`, i._url)),
        (r = i._query),
        delete i._url,
        delete i._query,
        (t = this.apiClient
          .request({
            path: n,
            queryParams: r,
            body: JSON.stringify(i),
            httpMethod: `POST`,
            httpOptions: e.config?.httpOptions,
            abortSignal: e.config?.abortSignal,
          })
          .then((e) => e.json())),
        t.then((e) => {
          let t = si(e),
            n = new jt();
          return (Object.assign(n, t), n);
        })
      );
    }
  }
  async get(e) {
    let t,
      n = ``,
      r = {};
    if (this.apiClient.isVertexAI())
      throw Error(`This method is only supported by the Gemini Developer API.`);
    {
      let i = ui(e);
      return (
        (n = p(`files/{file}`, i._url)),
        (r = i._query),
        delete i._url,
        delete i._query,
        (t = this.apiClient
          .request({
            path: n,
            queryParams: r,
            body: JSON.stringify(i),
            httpMethod: `GET`,
            httpOptions: e.config?.httpOptions,
            abortSignal: e.config?.abortSignal,
          })
          .then((e) => e.json())),
        t.then((e) => e)
      );
    }
  }
  async delete(e) {
    let t,
      n = ``,
      r = {};
    if (this.apiClient.isVertexAI())
      throw Error(`This method is only supported by the Gemini Developer API.`);
    {
      let i = ci(e);
      return (
        (n = p(`files/{file}`, i._url)),
        (r = i._query),
        delete i._url,
        delete i._query,
        (t = this.apiClient
          .request({
            path: n,
            queryParams: r,
            body: JSON.stringify(i),
            httpMethod: `DELETE`,
            httpOptions: e.config?.httpOptions,
            abortSignal: e.config?.abortSignal,
          })
          .then((e) =>
            e.json().then((t) => {
              let n = t;
              return ((n.sdkHttpResponse = { headers: e.headers }), n);
            }),
          )),
        t.then((e) => {
          let t = li(e),
            n = new Mt();
          return (Object.assign(n, t), n);
        })
      );
    }
  }
  async registerFilesInternal(e) {
    let t,
      n = ``,
      r = {};
    if (this.apiClient.isVertexAI())
      throw Error(`This method is only supported by the Gemini Developer API.`);
    {
      let i = di(e);
      return (
        (n = p(`files:register`, i._url)),
        (r = i._query),
        delete i._url,
        delete i._query,
        (t = this.apiClient
          .request({
            path: n,
            queryParams: r,
            body: JSON.stringify(i),
            httpMethod: `POST`,
            httpOptions: e.config?.httpOptions,
            abortSignal: e.config?.abortSignal,
          })
          .then((e) => e.json())),
        t.then((e) => {
          let t = hi(e),
            n = new Nt();
          return (Object.assign(n, t), n);
        })
      );
    }
  }
};
function _i(e) {
  let t = {};
  if (h(e, [`languageCodes`]) !== void 0)
    throw Error(`languageCodes parameter is not supported in Gemini API.`);
  return t;
}
function vi(e) {
  let t = {},
    n = h(e, [`apiKey`]);
  if ((n != null && m(t, [`apiKey`], n), h(e, [`apiKeyConfig`]) !== void 0))
    throw Error(`apiKeyConfig parameter is not supported in Gemini API.`);
  if (h(e, [`authType`]) !== void 0)
    throw Error(`authType parameter is not supported in Gemini API.`);
  if (h(e, [`googleServiceAccountConfig`]) !== void 0)
    throw Error(`googleServiceAccountConfig parameter is not supported in Gemini API.`);
  if (h(e, [`httpBasicAuthConfig`]) !== void 0)
    throw Error(`httpBasicAuthConfig parameter is not supported in Gemini API.`);
  if (h(e, [`oauthConfig`]) !== void 0)
    throw Error(`oauthConfig parameter is not supported in Gemini API.`);
  if (h(e, [`oidcConfig`]) !== void 0)
    throw Error(`oidcConfig parameter is not supported in Gemini API.`);
  return t;
}
function yi(e) {
  let t = {},
    n = h(e, [`data`]);
  if ((n != null && m(t, [`data`], n), h(e, [`displayName`]) !== void 0))
    throw Error(`displayName parameter is not supported in Gemini API.`);
  let r = h(e, [`mimeType`]);
  return (r != null && m(t, [`mimeType`], r), t);
}
function bi(e) {
  let t = {},
    n = h(e, [`parts`]);
  if (n != null) {
    let e = n;
    (Array.isArray(e) && (e = e.map((e) => Li(e))), m(t, [`parts`], e));
  }
  let r = h(e, [`role`]);
  return (r != null && m(t, [`role`], r), t);
}
function xi(e) {
  let t = {},
    n = h(e, [`parts`]);
  if (n != null) {
    let e = n;
    (Array.isArray(e) && (e = e.map((e) => Ri(e))), m(t, [`parts`], e));
  }
  let r = h(e, [`role`]);
  return (r != null && m(t, [`role`], r), t);
}
function Si(e) {
  let t = {};
  if (h(e, [`displayName`]) !== void 0)
    throw Error(`displayName parameter is not supported in Gemini API.`);
  let n = h(e, [`fileUri`]);
  n != null && m(t, [`fileUri`], n);
  let r = h(e, [`mimeType`]);
  return (r != null && m(t, [`mimeType`], r), t);
}
function Ci(e) {
  let t = {},
    n = h(e, [`id`]);
  n != null && m(t, [`id`], n);
  let r = h(e, [`args`]);
  r != null && m(t, [`args`], r);
  let i = h(e, [`name`]);
  if ((i != null && m(t, [`name`], i), h(e, [`partialArgs`]) !== void 0))
    throw Error(`partialArgs parameter is not supported in Gemini API.`);
  if (h(e, [`willContinue`]) !== void 0)
    throw Error(`willContinue parameter is not supported in Gemini API.`);
  return t;
}
function wi(e) {
  let t = {},
    n = h(e, [`description`]);
  n != null && m(t, [`description`], n);
  let r = h(e, [`name`]);
  r != null && m(t, [`name`], r);
  let i = h(e, [`parameters`]);
  i != null && m(t, [`parameters`], i);
  let a = h(e, [`parametersJsonSchema`]);
  a != null && m(t, [`parametersJsonSchema`], a);
  let o = h(e, [`response`]);
  o != null && m(t, [`response`], o);
  let s = h(e, [`responseJsonSchema`]);
  if ((s != null && m(t, [`responseJsonSchema`], s), h(e, [`behavior`]) !== void 0))
    throw Error(`behavior parameter is not supported in Vertex AI.`);
  return t;
}
function Ti(e) {
  let t = {},
    n = h(e, [`modelSelectionConfig`]);
  n != null && m(t, [`modelConfig`], n);
  let r = h(e, [`responseJsonSchema`]);
  r != null && m(t, [`responseJsonSchema`], r);
  let i = h(e, [`audioTimestamp`]);
  i != null && m(t, [`audioTimestamp`], i);
  let a = h(e, [`candidateCount`]);
  a != null && m(t, [`candidateCount`], a);
  let o = h(e, [`enableAffectiveDialog`]);
  o != null && m(t, [`enableAffectiveDialog`], o);
  let s = h(e, [`frequencyPenalty`]);
  s != null && m(t, [`frequencyPenalty`], s);
  let c = h(e, [`logprobs`]);
  c != null && m(t, [`logprobs`], c);
  let l = h(e, [`maxOutputTokens`]);
  l != null && m(t, [`maxOutputTokens`], l);
  let u = h(e, [`mediaResolution`]);
  u != null && m(t, [`mediaResolution`], u);
  let d = h(e, [`presencePenalty`]);
  d != null && m(t, [`presencePenalty`], d);
  let f = h(e, [`responseLogprobs`]);
  f != null && m(t, [`responseLogprobs`], f);
  let p = h(e, [`responseMimeType`]);
  p != null && m(t, [`responseMimeType`], p);
  let g = h(e, [`responseModalities`]);
  g != null && m(t, [`responseModalities`], g);
  let _ = h(e, [`responseSchema`]);
  _ != null && m(t, [`responseSchema`], _);
  let v = h(e, [`routingConfig`]);
  v != null && m(t, [`routingConfig`], v);
  let y = h(e, [`seed`]);
  y != null && m(t, [`seed`], y);
  let b = h(e, [`speechConfig`]);
  b != null && m(t, [`speechConfig`], b);
  let x = h(e, [`stopSequences`]);
  x != null && m(t, [`stopSequences`], x);
  let S = h(e, [`temperature`]);
  S != null && m(t, [`temperature`], S);
  let C = h(e, [`thinkingConfig`]);
  C != null && m(t, [`thinkingConfig`], C);
  let w = h(e, [`topK`]);
  w != null && m(t, [`topK`], w);
  let T = h(e, [`topP`]);
  if ((T != null && m(t, [`topP`], T), h(e, [`enableEnhancedCivicAnswers`]) !== void 0))
    throw Error(`enableEnhancedCivicAnswers parameter is not supported in Vertex AI.`);
  return t;
}
function Ei(e) {
  let t = {},
    n = h(e, [`authConfig`]);
  n != null && m(t, [`authConfig`], vi(n));
  let r = h(e, [`enableWidget`]);
  return (r != null && m(t, [`enableWidget`], r), t);
}
function Di(e) {
  let t = {},
    n = h(e, [`searchTypes`]);
  if ((n != null && m(t, [`searchTypes`], n), h(e, [`blockingConfidence`]) !== void 0))
    throw Error(`blockingConfidence parameter is not supported in Gemini API.`);
  if (h(e, [`excludeDomains`]) !== void 0)
    throw Error(`excludeDomains parameter is not supported in Gemini API.`);
  let r = h(e, [`timeRangeFilter`]);
  return (r != null && m(t, [`timeRangeFilter`], r), t);
}
function Oi(e, t) {
  let n = {},
    r = h(e, [`generationConfig`]);
  t !== void 0 && r != null && m(t, [`setup`, `generationConfig`], r);
  let i = h(e, [`responseModalities`]);
  t !== void 0 && i != null && m(t, [`setup`, `generationConfig`, `responseModalities`], i);
  let a = h(e, [`temperature`]);
  t !== void 0 && a != null && m(t, [`setup`, `generationConfig`, `temperature`], a);
  let o = h(e, [`topP`]);
  t !== void 0 && o != null && m(t, [`setup`, `generationConfig`, `topP`], o);
  let s = h(e, [`topK`]);
  t !== void 0 && s != null && m(t, [`setup`, `generationConfig`, `topK`], s);
  let c = h(e, [`maxOutputTokens`]);
  t !== void 0 && c != null && m(t, [`setup`, `generationConfig`, `maxOutputTokens`], c);
  let l = h(e, [`mediaResolution`]);
  t !== void 0 && l != null && m(t, [`setup`, `generationConfig`, `mediaResolution`], l);
  let u = h(e, [`seed`]);
  t !== void 0 && u != null && m(t, [`setup`, `generationConfig`, `seed`], u);
  let d = h(e, [`speechConfig`]);
  t !== void 0 && d != null && m(t, [`setup`, `generationConfig`, `speechConfig`], Qt(d));
  let f = h(e, [`thinkingConfig`]);
  t !== void 0 && f != null && m(t, [`setup`, `generationConfig`, `thinkingConfig`], f);
  let p = h(e, [`enableAffectiveDialog`]);
  t !== void 0 && p != null && m(t, [`setup`, `generationConfig`, `enableAffectiveDialog`], p);
  let g = h(e, [`systemInstruction`]);
  t !== void 0 && g != null && m(t, [`setup`, `systemInstruction`], bi(F(g)));
  let _ = h(e, [`tools`]);
  if (t !== void 0 && _ != null) {
    let e = z(_);
    (Array.isArray(e) && (e = e.map((e) => Bi(R(e)))), m(t, [`setup`, `tools`], e));
  }
  let v = h(e, [`sessionResumption`]);
  t !== void 0 && v != null && m(t, [`setup`, `sessionResumption`], zi(v));
  let y = h(e, [`inputAudioTranscription`]);
  t !== void 0 && y != null && m(t, [`setup`, `inputAudioTranscription`], _i(y));
  let b = h(e, [`outputAudioTranscription`]);
  t !== void 0 && b != null && m(t, [`setup`, `outputAudioTranscription`], _i(b));
  let x = h(e, [`realtimeInputConfig`]);
  t !== void 0 && x != null && m(t, [`setup`, `realtimeInputConfig`], x);
  let S = h(e, [`contextWindowCompression`]);
  t !== void 0 && S != null && m(t, [`setup`, `contextWindowCompression`], S);
  let C = h(e, [`proactivity`]);
  if (
    (t !== void 0 && C != null && m(t, [`setup`, `proactivity`], C),
    h(e, [`explicitVadSignal`]) !== void 0)
  )
    throw Error(`explicitVadSignal parameter is not supported in Gemini API.`);
  return n;
}
function ki(e, t) {
  let n = {},
    r = h(e, [`generationConfig`]);
  t !== void 0 && r != null && m(t, [`setup`, `generationConfig`], Ti(r));
  let i = h(e, [`responseModalities`]);
  t !== void 0 && i != null && m(t, [`setup`, `generationConfig`, `responseModalities`], i);
  let a = h(e, [`temperature`]);
  t !== void 0 && a != null && m(t, [`setup`, `generationConfig`, `temperature`], a);
  let o = h(e, [`topP`]);
  t !== void 0 && o != null && m(t, [`setup`, `generationConfig`, `topP`], o);
  let s = h(e, [`topK`]);
  t !== void 0 && s != null && m(t, [`setup`, `generationConfig`, `topK`], s);
  let c = h(e, [`maxOutputTokens`]);
  t !== void 0 && c != null && m(t, [`setup`, `generationConfig`, `maxOutputTokens`], c);
  let l = h(e, [`mediaResolution`]);
  t !== void 0 && l != null && m(t, [`setup`, `generationConfig`, `mediaResolution`], l);
  let u = h(e, [`seed`]);
  t !== void 0 && u != null && m(t, [`setup`, `generationConfig`, `seed`], u);
  let d = h(e, [`speechConfig`]);
  t !== void 0 && d != null && m(t, [`setup`, `generationConfig`, `speechConfig`], Qt(d));
  let f = h(e, [`thinkingConfig`]);
  t !== void 0 && f != null && m(t, [`setup`, `generationConfig`, `thinkingConfig`], f);
  let p = h(e, [`enableAffectiveDialog`]);
  t !== void 0 && p != null && m(t, [`setup`, `generationConfig`, `enableAffectiveDialog`], p);
  let g = h(e, [`systemInstruction`]);
  t !== void 0 && g != null && m(t, [`setup`, `systemInstruction`], xi(F(g)));
  let _ = h(e, [`tools`]);
  if (t !== void 0 && _ != null) {
    let e = z(_);
    (Array.isArray(e) && (e = e.map((e) => Vi(R(e)))), m(t, [`setup`, `tools`], e));
  }
  let v = h(e, [`sessionResumption`]);
  t !== void 0 && v != null && m(t, [`setup`, `sessionResumption`], v);
  let y = h(e, [`inputAudioTranscription`]);
  t !== void 0 && y != null && m(t, [`setup`, `inputAudioTranscription`], y);
  let b = h(e, [`outputAudioTranscription`]);
  t !== void 0 && b != null && m(t, [`setup`, `outputAudioTranscription`], b);
  let x = h(e, [`realtimeInputConfig`]);
  t !== void 0 && x != null && m(t, [`setup`, `realtimeInputConfig`], x);
  let S = h(e, [`contextWindowCompression`]);
  t !== void 0 && S != null && m(t, [`setup`, `contextWindowCompression`], S);
  let C = h(e, [`proactivity`]);
  t !== void 0 && C != null && m(t, [`setup`, `proactivity`], C);
  let w = h(e, [`explicitVadSignal`]);
  return (t !== void 0 && w != null && m(t, [`setup`, `explicitVadSignal`], w), n);
}
function Ai(e, t) {
  let n = {},
    r = h(t, [`model`]);
  r != null && m(n, [`setup`, `model`], P(e, r));
  let i = h(t, [`config`]);
  return (i != null && m(n, [`config`], Oi(i, n)), n);
}
function ji(e, t) {
  let n = {},
    r = h(t, [`model`]);
  r != null && m(n, [`setup`, `model`], P(e, r));
  let i = h(t, [`config`]);
  return (i != null && m(n, [`config`], ki(i, n)), n);
}
function Mi(e) {
  let t = {},
    n = h(e, [`musicGenerationConfig`]);
  return (n != null && m(t, [`musicGenerationConfig`], n), t);
}
function Ni(e) {
  let t = {},
    n = h(e, [`weightedPrompts`]);
  if (n != null) {
    let e = n;
    (Array.isArray(e) && (e = e.map((e) => e)), m(t, [`weightedPrompts`], e));
  }
  return t;
}
function Pi(e) {
  let t = {},
    n = h(e, [`media`]);
  if (n != null) {
    let e = zt(n);
    (Array.isArray(e) && (e = e.map((e) => yi(e))), m(t, [`mediaChunks`], e));
  }
  let r = h(e, [`audio`]);
  r != null && m(t, [`audio`], yi(Ht(r)));
  let i = h(e, [`audioStreamEnd`]);
  i != null && m(t, [`audioStreamEnd`], i);
  let a = h(e, [`video`]);
  a != null && m(t, [`video`], yi(Vt(a)));
  let o = h(e, [`text`]);
  o != null && m(t, [`text`], o);
  let s = h(e, [`activityStart`]);
  s != null && m(t, [`activityStart`], s);
  let c = h(e, [`activityEnd`]);
  return (c != null && m(t, [`activityEnd`], c), t);
}
function Fi(e) {
  let t = {},
    n = h(e, [`media`]);
  if (n != null) {
    let e = zt(n);
    (Array.isArray(e) && (e = e.map((e) => e)), m(t, [`mediaChunks`], e));
  }
  let r = h(e, [`audio`]);
  r != null && m(t, [`audio`], Ht(r));
  let i = h(e, [`audioStreamEnd`]);
  i != null && m(t, [`audioStreamEnd`], i);
  let a = h(e, [`video`]);
  a != null && m(t, [`video`], Vt(a));
  let o = h(e, [`text`]);
  o != null && m(t, [`text`], o);
  let s = h(e, [`activityStart`]);
  s != null && m(t, [`activityStart`], s);
  let c = h(e, [`activityEnd`]);
  return (c != null && m(t, [`activityEnd`], c), t);
}
function Ii(e) {
  let t = {},
    n = h(e, [`setupComplete`]);
  n != null && m(t, [`setupComplete`], n);
  let r = h(e, [`serverContent`]);
  r != null && m(t, [`serverContent`], r);
  let i = h(e, [`toolCall`]);
  i != null && m(t, [`toolCall`], i);
  let a = h(e, [`toolCallCancellation`]);
  a != null && m(t, [`toolCallCancellation`], a);
  let o = h(e, [`usageMetadata`]);
  o != null && m(t, [`usageMetadata`], Hi(o));
  let s = h(e, [`goAway`]);
  s != null && m(t, [`goAway`], s);
  let c = h(e, [`sessionResumptionUpdate`]);
  c != null && m(t, [`sessionResumptionUpdate`], c);
  let l = h(e, [`voiceActivityDetectionSignal`]);
  l != null && m(t, [`voiceActivityDetectionSignal`], l);
  let u = h(e, [`voiceActivity`]);
  return (u != null && m(t, [`voiceActivity`], Ui(u)), t);
}
function Li(e) {
  let t = {},
    n = h(e, [`mediaResolution`]);
  n != null && m(t, [`mediaResolution`], n);
  let r = h(e, [`codeExecutionResult`]);
  r != null && m(t, [`codeExecutionResult`], r);
  let i = h(e, [`executableCode`]);
  i != null && m(t, [`executableCode`], i);
  let a = h(e, [`fileData`]);
  a != null && m(t, [`fileData`], Si(a));
  let o = h(e, [`functionCall`]);
  o != null && m(t, [`functionCall`], Ci(o));
  let s = h(e, [`functionResponse`]);
  s != null && m(t, [`functionResponse`], s);
  let c = h(e, [`inlineData`]);
  c != null && m(t, [`inlineData`], yi(c));
  let l = h(e, [`text`]);
  l != null && m(t, [`text`], l);
  let u = h(e, [`thought`]);
  u != null && m(t, [`thought`], u);
  let d = h(e, [`thoughtSignature`]);
  d != null && m(t, [`thoughtSignature`], d);
  let f = h(e, [`videoMetadata`]);
  f != null && m(t, [`videoMetadata`], f);
  let p = h(e, [`toolCall`]);
  p != null && m(t, [`toolCall`], p);
  let g = h(e, [`toolResponse`]);
  return (g != null && m(t, [`toolResponse`], g), t);
}
function Ri(e) {
  let t = {},
    n = h(e, [`mediaResolution`]);
  n != null && m(t, [`mediaResolution`], n);
  let r = h(e, [`codeExecutionResult`]);
  r != null && m(t, [`codeExecutionResult`], r);
  let i = h(e, [`executableCode`]);
  i != null && m(t, [`executableCode`], i);
  let a = h(e, [`fileData`]);
  a != null && m(t, [`fileData`], a);
  let o = h(e, [`functionCall`]);
  o != null && m(t, [`functionCall`], o);
  let s = h(e, [`functionResponse`]);
  s != null && m(t, [`functionResponse`], s);
  let c = h(e, [`inlineData`]);
  c != null && m(t, [`inlineData`], c);
  let l = h(e, [`text`]);
  l != null && m(t, [`text`], l);
  let u = h(e, [`thought`]);
  u != null && m(t, [`thought`], u);
  let d = h(e, [`thoughtSignature`]);
  d != null && m(t, [`thoughtSignature`], d);
  let f = h(e, [`videoMetadata`]);
  if ((f != null && m(t, [`videoMetadata`], f), h(e, [`toolCall`]) !== void 0))
    throw Error(`toolCall parameter is not supported in Vertex AI.`);
  if (h(e, [`toolResponse`]) !== void 0)
    throw Error(`toolResponse parameter is not supported in Vertex AI.`);
  return t;
}
function zi(e) {
  let t = {},
    n = h(e, [`handle`]);
  if ((n != null && m(t, [`handle`], n), h(e, [`transparent`]) !== void 0))
    throw Error(`transparent parameter is not supported in Gemini API.`);
  return t;
}
function Bi(e) {
  let t = {};
  if (h(e, [`retrieval`]) !== void 0)
    throw Error(`retrieval parameter is not supported in Gemini API.`);
  let n = h(e, [`computerUse`]);
  n != null && m(t, [`computerUse`], n);
  let r = h(e, [`fileSearch`]);
  r != null && m(t, [`fileSearch`], r);
  let i = h(e, [`googleSearch`]);
  i != null && m(t, [`googleSearch`], Di(i));
  let a = h(e, [`googleMaps`]);
  a != null && m(t, [`googleMaps`], Ei(a));
  let o = h(e, [`codeExecution`]);
  if ((o != null && m(t, [`codeExecution`], o), h(e, [`enterpriseWebSearch`]) !== void 0))
    throw Error(`enterpriseWebSearch parameter is not supported in Gemini API.`);
  let s = h(e, [`functionDeclarations`]);
  if (s != null) {
    let e = s;
    (Array.isArray(e) && (e = e.map((e) => e)), m(t, [`functionDeclarations`], e));
  }
  let c = h(e, [`googleSearchRetrieval`]);
  if ((c != null && m(t, [`googleSearchRetrieval`], c), h(e, [`parallelAiSearch`]) !== void 0))
    throw Error(`parallelAiSearch parameter is not supported in Gemini API.`);
  let l = h(e, [`urlContext`]);
  l != null && m(t, [`urlContext`], l);
  let u = h(e, [`mcpServers`]);
  if (u != null) {
    let e = u;
    (Array.isArray(e) && (e = e.map((e) => e)), m(t, [`mcpServers`], e));
  }
  return t;
}
function Vi(e) {
  let t = {},
    n = h(e, [`retrieval`]);
  n != null && m(t, [`retrieval`], n);
  let r = h(e, [`computerUse`]);
  if ((r != null && m(t, [`computerUse`], r), h(e, [`fileSearch`]) !== void 0))
    throw Error(`fileSearch parameter is not supported in Vertex AI.`);
  let i = h(e, [`googleSearch`]);
  i != null && m(t, [`googleSearch`], i);
  let a = h(e, [`googleMaps`]);
  a != null && m(t, [`googleMaps`], a);
  let o = h(e, [`codeExecution`]);
  o != null && m(t, [`codeExecution`], o);
  let s = h(e, [`enterpriseWebSearch`]);
  s != null && m(t, [`enterpriseWebSearch`], s);
  let c = h(e, [`functionDeclarations`]);
  if (c != null) {
    let e = c;
    (Array.isArray(e) && (e = e.map((e) => wi(e))), m(t, [`functionDeclarations`], e));
  }
  let l = h(e, [`googleSearchRetrieval`]);
  l != null && m(t, [`googleSearchRetrieval`], l);
  let u = h(e, [`parallelAiSearch`]);
  u != null && m(t, [`parallelAiSearch`], u);
  let d = h(e, [`urlContext`]);
  if ((d != null && m(t, [`urlContext`], d), h(e, [`mcpServers`]) !== void 0))
    throw Error(`mcpServers parameter is not supported in Vertex AI.`);
  return t;
}
function Hi(e) {
  let t = {},
    n = h(e, [`promptTokenCount`]);
  n != null && m(t, [`promptTokenCount`], n);
  let r = h(e, [`cachedContentTokenCount`]);
  r != null && m(t, [`cachedContentTokenCount`], r);
  let i = h(e, [`candidatesTokenCount`]);
  i != null && m(t, [`responseTokenCount`], i);
  let a = h(e, [`toolUsePromptTokenCount`]);
  a != null && m(t, [`toolUsePromptTokenCount`], a);
  let o = h(e, [`thoughtsTokenCount`]);
  o != null && m(t, [`thoughtsTokenCount`], o);
  let s = h(e, [`totalTokenCount`]);
  s != null && m(t, [`totalTokenCount`], s);
  let c = h(e, [`promptTokensDetails`]);
  if (c != null) {
    let e = c;
    (Array.isArray(e) && (e = e.map((e) => e)), m(t, [`promptTokensDetails`], e));
  }
  let l = h(e, [`cacheTokensDetails`]);
  if (l != null) {
    let e = l;
    (Array.isArray(e) && (e = e.map((e) => e)), m(t, [`cacheTokensDetails`], e));
  }
  let u = h(e, [`candidatesTokensDetails`]);
  if (u != null) {
    let e = u;
    (Array.isArray(e) && (e = e.map((e) => e)), m(t, [`responseTokensDetails`], e));
  }
  let d = h(e, [`toolUsePromptTokensDetails`]);
  if (d != null) {
    let e = d;
    (Array.isArray(e) && (e = e.map((e) => e)), m(t, [`toolUsePromptTokensDetails`], e));
  }
  let f = h(e, [`trafficType`]);
  return (f != null && m(t, [`trafficType`], f), t);
}
function Ui(e) {
  let t = {},
    n = h(e, [`type`]);
  return (n != null && m(t, [`voiceActivityType`], n), t);
}
function Wi(e, t) {
  let n = {},
    r = h(e, [`apiKey`]);
  if ((r != null && m(n, [`apiKey`], r), h(e, [`apiKeyConfig`]) !== void 0))
    throw Error(`apiKeyConfig parameter is not supported in Gemini API.`);
  if (h(e, [`authType`]) !== void 0)
    throw Error(`authType parameter is not supported in Gemini API.`);
  if (h(e, [`googleServiceAccountConfig`]) !== void 0)
    throw Error(`googleServiceAccountConfig parameter is not supported in Gemini API.`);
  if (h(e, [`httpBasicAuthConfig`]) !== void 0)
    throw Error(`httpBasicAuthConfig parameter is not supported in Gemini API.`);
  if (h(e, [`oauthConfig`]) !== void 0)
    throw Error(`oauthConfig parameter is not supported in Gemini API.`);
  if (h(e, [`oidcConfig`]) !== void 0)
    throw Error(`oidcConfig parameter is not supported in Gemini API.`);
  return n;
}
function Gi(e, t) {
  let n = {},
    r = h(e, [`data`]);
  if ((r != null && m(n, [`data`], r), h(e, [`displayName`]) !== void 0))
    throw Error(`displayName parameter is not supported in Gemini API.`);
  let i = h(e, [`mimeType`]);
  return (i != null && m(n, [`mimeType`], i), n);
}
function Ki(e, t) {
  let n = {},
    r = h(e, [`content`]);
  r != null && m(n, [`content`], r);
  let i = h(e, [`citationMetadata`]);
  i != null && m(n, [`citationMetadata`], qi(i));
  let a = h(e, [`tokenCount`]);
  a != null && m(n, [`tokenCount`], a);
  let o = h(e, [`finishReason`]);
  o != null && m(n, [`finishReason`], o);
  let s = h(e, [`groundingMetadata`]);
  s != null && m(n, [`groundingMetadata`], s);
  let c = h(e, [`avgLogprobs`]);
  c != null && m(n, [`avgLogprobs`], c);
  let l = h(e, [`index`]);
  l != null && m(n, [`index`], l);
  let u = h(e, [`logprobsResult`]);
  u != null && m(n, [`logprobsResult`], u);
  let d = h(e, [`safetyRatings`]);
  if (d != null) {
    let e = d;
    (Array.isArray(e) && (e = e.map((e) => e)), m(n, [`safetyRatings`], e));
  }
  let f = h(e, [`urlContextMetadata`]);
  return (f != null && m(n, [`urlContextMetadata`], f), n);
}
function qi(e, t) {
  let n = {},
    r = h(e, [`citationSources`]);
  if (r != null) {
    let e = r;
    (Array.isArray(e) && (e = e.map((e) => e)), m(n, [`citations`], e));
  }
  return n;
}
function Ji(e, t, n) {
  let r = {},
    i = h(t, [`model`]);
  i != null && m(r, [`_url`, `model`], P(e, i));
  let a = h(t, [`contents`]);
  if (a != null) {
    let e = I(a);
    (Array.isArray(e) && (e = e.map((e) => q(e))), m(r, [`contents`], e));
  }
  return r;
}
function Yi(e, t) {
  let n = {},
    r = h(e, [`sdkHttpResponse`]);
  r != null && m(n, [`sdkHttpResponse`], r);
  let i = h(e, [`tokensInfo`]);
  if (i != null) {
    let e = i;
    (Array.isArray(e) && (e = e.map((e) => e)), m(n, [`tokensInfo`], e));
  }
  return n;
}
function Xi(e, t) {
  let n = {},
    r = h(e, [`values`]);
  r != null && m(n, [`values`], r);
  let i = h(e, [`statistics`]);
  return (i != null && m(n, [`statistics`], Zi(i)), n);
}
function Zi(e, t) {
  let n = {},
    r = h(e, [`truncated`]);
  r != null && m(n, [`truncated`], r);
  let i = h(e, [`token_count`]);
  return (i != null && m(n, [`tokenCount`], i), n);
}
function Qi(e, t) {
  let n = {},
    r = h(e, [`parts`]);
  if (r != null) {
    let e = r;
    (Array.isArray(e) && (e = e.map((e) => _o(e))), m(n, [`parts`], e));
  }
  let i = h(e, [`role`]);
  return (i != null && m(n, [`role`], i), n);
}
function q(e, t) {
  let n = {},
    r = h(e, [`parts`]);
  if (r != null) {
    let e = r;
    (Array.isArray(e) && (e = e.map((e) => vo(e))), m(n, [`parts`], e));
  }
  let i = h(e, [`role`]);
  return (i != null && m(n, [`role`], i), n);
}
function $i(e, t) {
  let n = {},
    r = h(e, [`controlType`]);
  r != null && m(n, [`controlType`], r);
  let i = h(e, [`enableControlImageComputation`]);
  return (i != null && m(n, [`computeControl`], i), n);
}
function ea(e, t) {
  let n = {};
  if (h(e, [`systemInstruction`]) !== void 0)
    throw Error(`systemInstruction parameter is not supported in Gemini API.`);
  if (h(e, [`tools`]) !== void 0) throw Error(`tools parameter is not supported in Gemini API.`);
  if (h(e, [`generationConfig`]) !== void 0)
    throw Error(`generationConfig parameter is not supported in Gemini API.`);
  return n;
}
function ta(e, t, n) {
  let r = {},
    i = h(e, [`systemInstruction`]);
  t !== void 0 && i != null && m(t, [`systemInstruction`], q(F(i)));
  let a = h(e, [`tools`]);
  if (t !== void 0 && a != null) {
    let e = a;
    (Array.isArray(e) && (e = e.map((e) => Io(e))), m(t, [`tools`], e));
  }
  let o = h(e, [`generationConfig`]);
  return (t !== void 0 && o != null && m(t, [`generationConfig`], Za(o)), r);
}
function na(e, t, n) {
  let r = {},
    i = h(t, [`model`]);
  i != null && m(r, [`_url`, `model`], P(e, i));
  let a = h(t, [`contents`]);
  if (a != null) {
    let e = I(a);
    (Array.isArray(e) && (e = e.map((e) => Qi(e))), m(r, [`contents`], e));
  }
  let o = h(t, [`config`]);
  return (o != null && ea(o), r);
}
function ra(e, t, n) {
  let r = {},
    i = h(t, [`model`]);
  i != null && m(r, [`_url`, `model`], P(e, i));
  let a = h(t, [`contents`]);
  if (a != null) {
    let e = I(a);
    (Array.isArray(e) && (e = e.map((e) => q(e))), m(r, [`contents`], e));
  }
  let o = h(t, [`config`]);
  return (o != null && ta(o, r), r);
}
function ia(e, t) {
  let n = {},
    r = h(e, [`sdkHttpResponse`]);
  r != null && m(n, [`sdkHttpResponse`], r);
  let i = h(e, [`totalTokens`]);
  i != null && m(n, [`totalTokens`], i);
  let a = h(e, [`cachedContentTokenCount`]);
  return (a != null && m(n, [`cachedContentTokenCount`], a), n);
}
function aa(e, t) {
  let n = {},
    r = h(e, [`sdkHttpResponse`]);
  r != null && m(n, [`sdkHttpResponse`], r);
  let i = h(e, [`totalTokens`]);
  return (i != null && m(n, [`totalTokens`], i), n);
}
function oa(e, t, n) {
  let r = {},
    i = h(t, [`model`]);
  return (i != null && m(r, [`_url`, `name`], P(e, i)), r);
}
function sa(e, t, n) {
  let r = {},
    i = h(t, [`model`]);
  return (i != null && m(r, [`_url`, `name`], P(e, i)), r);
}
function ca(e, t) {
  let n = {},
    r = h(e, [`sdkHttpResponse`]);
  return (r != null && m(n, [`sdkHttpResponse`], r), n);
}
function la(e, t) {
  let n = {},
    r = h(e, [`sdkHttpResponse`]);
  return (r != null && m(n, [`sdkHttpResponse`], r), n);
}
function ua(e, t, n) {
  let r = {},
    i = h(e, [`outputGcsUri`]);
  t !== void 0 && i != null && m(t, [`parameters`, `storageUri`], i);
  let a = h(e, [`negativePrompt`]);
  t !== void 0 && a != null && m(t, [`parameters`, `negativePrompt`], a);
  let o = h(e, [`numberOfImages`]);
  t !== void 0 && o != null && m(t, [`parameters`, `sampleCount`], o);
  let s = h(e, [`aspectRatio`]);
  t !== void 0 && s != null && m(t, [`parameters`, `aspectRatio`], s);
  let c = h(e, [`guidanceScale`]);
  t !== void 0 && c != null && m(t, [`parameters`, `guidanceScale`], c);
  let l = h(e, [`seed`]);
  t !== void 0 && l != null && m(t, [`parameters`, `seed`], l);
  let u = h(e, [`safetyFilterLevel`]);
  t !== void 0 && u != null && m(t, [`parameters`, `safetySetting`], u);
  let d = h(e, [`personGeneration`]);
  t !== void 0 && d != null && m(t, [`parameters`, `personGeneration`], d);
  let f = h(e, [`includeSafetyAttributes`]);
  t !== void 0 && f != null && m(t, [`parameters`, `includeSafetyAttributes`], f);
  let p = h(e, [`includeRaiReason`]);
  t !== void 0 && p != null && m(t, [`parameters`, `includeRaiReason`], p);
  let g = h(e, [`language`]);
  t !== void 0 && g != null && m(t, [`parameters`, `language`], g);
  let _ = h(e, [`outputMimeType`]);
  t !== void 0 && _ != null && m(t, [`parameters`, `outputOptions`, `mimeType`], _);
  let v = h(e, [`outputCompressionQuality`]);
  t !== void 0 && v != null && m(t, [`parameters`, `outputOptions`, `compressionQuality`], v);
  let y = h(e, [`addWatermark`]);
  t !== void 0 && y != null && m(t, [`parameters`, `addWatermark`], y);
  let b = h(e, [`labels`]);
  t !== void 0 && b != null && m(t, [`labels`], b);
  let x = h(e, [`editMode`]);
  t !== void 0 && x != null && m(t, [`parameters`, `editMode`], x);
  let S = h(e, [`baseSteps`]);
  return (t !== void 0 && S != null && m(t, [`parameters`, `editConfig`, `baseSteps`], S), r);
}
function da(e, t, n) {
  let r = {},
    i = h(t, [`model`]);
  i != null && m(r, [`_url`, `model`], P(e, i));
  let a = h(t, [`prompt`]);
  a != null && m(r, [`instances[0]`, `prompt`], a);
  let o = h(t, [`referenceImages`]);
  if (o != null) {
    let e = o;
    (Array.isArray(e) && (e = e.map((e) => wo(e))), m(r, [`instances[0]`, `referenceImages`], e));
  }
  let s = h(t, [`config`]);
  return (s != null && ua(s, r), r);
}
function fa(e, t) {
  let n = {},
    r = h(e, [`sdkHttpResponse`]);
  r != null && m(n, [`sdkHttpResponse`], r);
  let i = h(e, [`predictions`]);
  if (i != null) {
    let e = i;
    (Array.isArray(e) && (e = e.map((e) => qa(e))), m(n, [`generatedImages`], e));
  }
  return n;
}
function pa(e, t, n) {
  let r = {},
    i = h(e, [`taskType`]);
  t !== void 0 && i != null && m(t, [`requests[]`, `taskType`], i);
  let a = h(e, [`title`]);
  t !== void 0 && a != null && m(t, [`requests[]`, `title`], a);
  let o = h(e, [`outputDimensionality`]);
  if (
    (t !== void 0 && o != null && m(t, [`requests[]`, `outputDimensionality`], o),
    h(e, [`mimeType`]) !== void 0)
  )
    throw Error(`mimeType parameter is not supported in Gemini API.`);
  if (h(e, [`autoTruncate`]) !== void 0)
    throw Error(`autoTruncate parameter is not supported in Gemini API.`);
  return r;
}
function ma(e, t, n) {
  let r = {},
    i = h(n, [`embeddingApiType`]);
  if ((i === void 0 && (i = `PREDICT`), i === `PREDICT`)) {
    let n = h(e, [`taskType`]);
    t !== void 0 && n != null && m(t, [`instances[]`, `task_type`], n);
  } else if (i === `EMBED_CONTENT`) {
    let n = h(e, [`taskType`]);
    t !== void 0 && n != null && m(t, [`taskType`], n);
  }
  let a = h(n, [`embeddingApiType`]);
  if ((a === void 0 && (a = `PREDICT`), a === `PREDICT`)) {
    let n = h(e, [`title`]);
    t !== void 0 && n != null && m(t, [`instances[]`, `title`], n);
  } else if (a === `EMBED_CONTENT`) {
    let n = h(e, [`title`]);
    t !== void 0 && n != null && m(t, [`title`], n);
  }
  let o = h(n, [`embeddingApiType`]);
  if ((o === void 0 && (o = `PREDICT`), o === `PREDICT`)) {
    let n = h(e, [`outputDimensionality`]);
    t !== void 0 && n != null && m(t, [`parameters`, `outputDimensionality`], n);
  } else if (o === `EMBED_CONTENT`) {
    let n = h(e, [`outputDimensionality`]);
    t !== void 0 && n != null && m(t, [`outputDimensionality`], n);
  }
  let s = h(n, [`embeddingApiType`]);
  if ((s === void 0 && (s = `PREDICT`), s === `PREDICT`)) {
    let n = h(e, [`mimeType`]);
    t !== void 0 && n != null && m(t, [`instances[]`, `mimeType`], n);
  }
  let c = h(n, [`embeddingApiType`]);
  if ((c === void 0 && (c = `PREDICT`), c === `PREDICT`)) {
    let n = h(e, [`autoTruncate`]);
    t !== void 0 && n != null && m(t, [`parameters`, `autoTruncate`], n);
  } else if (c === `EMBED_CONTENT`) {
    let n = h(e, [`autoTruncate`]);
    t !== void 0 && n != null && m(t, [`autoTruncate`], n);
  }
  return r;
}
function ha(e, t, n) {
  let r = {},
    i = h(t, [`model`]);
  i != null && m(r, [`_url`, `model`], P(e, i));
  let a = h(t, [`contents`]);
  if (a != null) {
    let t = Jt(e, a);
    (Array.isArray(t) && (t = t.map((e) => e)), m(r, [`requests[]`, `content`], t));
  }
  let o = h(t, [`content`]);
  o != null && Qi(F(o));
  let s = h(t, [`config`]);
  s != null && pa(s, r);
  let c = h(t, [`model`]);
  return (c !== void 0 && m(r, [`requests[]`, `model`], P(e, c)), r);
}
function ga(e, t, n) {
  let r = {},
    i = h(t, [`model`]);
  i != null && m(r, [`_url`, `model`], P(e, i));
  let a = h(n, [`embeddingApiType`]);
  if ((a === void 0 && (a = `PREDICT`), a === `PREDICT`)) {
    let n = h(t, [`contents`]);
    if (n != null) {
      let t = Jt(e, n);
      (Array.isArray(t) && (t = t.map((e) => e)), m(r, [`instances[]`, `content`], t));
    }
  }
  let o = h(n, [`embeddingApiType`]);
  if ((o === void 0 && (o = `PREDICT`), o === `EMBED_CONTENT`)) {
    let e = h(t, [`content`]);
    e != null && m(r, [`content`], q(F(e)));
  }
  let s = h(t, [`config`]);
  return (s != null && ma(s, r, n), r);
}
function _a(e, t) {
  let n = {},
    r = h(e, [`sdkHttpResponse`]);
  r != null && m(n, [`sdkHttpResponse`], r);
  let i = h(e, [`embeddings`]);
  if (i != null) {
    let e = i;
    (Array.isArray(e) && (e = e.map((e) => e)), m(n, [`embeddings`], e));
  }
  let a = h(e, [`metadata`]);
  return (a != null && m(n, [`metadata`], a), n);
}
function va(e, t) {
  let n = {},
    r = h(e, [`sdkHttpResponse`]);
  r != null && m(n, [`sdkHttpResponse`], r);
  let i = h(e, [`predictions[]`, `embeddings`]);
  if (i != null) {
    let e = i;
    (Array.isArray(e) && (e = e.map((e) => Xi(e))), m(n, [`embeddings`], e));
  }
  let a = h(e, [`metadata`]);
  if ((a != null && m(n, [`metadata`], a), t && h(t, [`embeddingApiType`]) === `EMBED_CONTENT`)) {
    let t = h(e, [`embedding`]),
      r = h(e, [`usageMetadata`]),
      i = h(e, [`truncated`]);
    if (t) {
      let e = {};
      (r && r.promptTokenCount && (e.tokenCount = r.promptTokenCount),
        i && (e.truncated = i),
        (t.statistics = e),
        m(n, [`embeddings`], [t]));
    }
  }
  return n;
}
function ya(e, t) {
  let n = {},
    r = h(e, [`endpoint`]);
  r != null && m(n, [`name`], r);
  let i = h(e, [`deployedModelId`]);
  return (i != null && m(n, [`deployedModelId`], i), n);
}
function ba(e, t) {
  let n = {};
  if (h(e, [`displayName`]) !== void 0)
    throw Error(`displayName parameter is not supported in Gemini API.`);
  let r = h(e, [`fileUri`]);
  r != null && m(n, [`fileUri`], r);
  let i = h(e, [`mimeType`]);
  return (i != null && m(n, [`mimeType`], i), n);
}
function xa(e, t) {
  let n = {},
    r = h(e, [`id`]);
  r != null && m(n, [`id`], r);
  let i = h(e, [`args`]);
  i != null && m(n, [`args`], i);
  let a = h(e, [`name`]);
  if ((a != null && m(n, [`name`], a), h(e, [`partialArgs`]) !== void 0))
    throw Error(`partialArgs parameter is not supported in Gemini API.`);
  if (h(e, [`willContinue`]) !== void 0)
    throw Error(`willContinue parameter is not supported in Gemini API.`);
  return n;
}
function Sa(e, t) {
  let n = {},
    r = h(e, [`allowedFunctionNames`]);
  r != null && m(n, [`allowedFunctionNames`], r);
  let i = h(e, [`mode`]);
  if ((i != null && m(n, [`mode`], i), h(e, [`streamFunctionCallArguments`]) !== void 0))
    throw Error(`streamFunctionCallArguments parameter is not supported in Gemini API.`);
  return n;
}
function Ca(e, t) {
  let n = {},
    r = h(e, [`description`]);
  r != null && m(n, [`description`], r);
  let i = h(e, [`name`]);
  i != null && m(n, [`name`], i);
  let a = h(e, [`parameters`]);
  a != null && m(n, [`parameters`], a);
  let o = h(e, [`parametersJsonSchema`]);
  o != null && m(n, [`parametersJsonSchema`], o);
  let s = h(e, [`response`]);
  s != null && m(n, [`response`], s);
  let c = h(e, [`responseJsonSchema`]);
  if ((c != null && m(n, [`responseJsonSchema`], c), h(e, [`behavior`]) !== void 0))
    throw Error(`behavior parameter is not supported in Vertex AI.`);
  return n;
}
function wa(e, t, n, r) {
  let i = {},
    a = h(t, [`systemInstruction`]);
  n !== void 0 && a != null && m(n, [`systemInstruction`], Qi(F(a)));
  let o = h(t, [`temperature`]);
  o != null && m(i, [`temperature`], o);
  let s = h(t, [`topP`]);
  s != null && m(i, [`topP`], s);
  let c = h(t, [`topK`]);
  c != null && m(i, [`topK`], c);
  let l = h(t, [`candidateCount`]);
  l != null && m(i, [`candidateCount`], l);
  let u = h(t, [`maxOutputTokens`]);
  u != null && m(i, [`maxOutputTokens`], u);
  let d = h(t, [`stopSequences`]);
  d != null && m(i, [`stopSequences`], d);
  let f = h(t, [`responseLogprobs`]);
  f != null && m(i, [`responseLogprobs`], f);
  let p = h(t, [`logprobs`]);
  p != null && m(i, [`logprobs`], p);
  let g = h(t, [`presencePenalty`]);
  g != null && m(i, [`presencePenalty`], g);
  let _ = h(t, [`frequencyPenalty`]);
  _ != null && m(i, [`frequencyPenalty`], _);
  let v = h(t, [`seed`]);
  v != null && m(i, [`seed`], v);
  let y = h(t, [`responseMimeType`]);
  y != null && m(i, [`responseMimeType`], y);
  let b = h(t, [`responseSchema`]);
  b != null && m(i, [`responseSchema`], Xt(b));
  let x = h(t, [`responseJsonSchema`]);
  if ((x != null && m(i, [`responseJsonSchema`], x), h(t, [`routingConfig`]) !== void 0))
    throw Error(`routingConfig parameter is not supported in Gemini API.`);
  if (h(t, [`modelSelectionConfig`]) !== void 0)
    throw Error(`modelSelectionConfig parameter is not supported in Gemini API.`);
  let S = h(t, [`safetySettings`]);
  if (n !== void 0 && S != null) {
    let e = S;
    (Array.isArray(e) && (e = e.map((e) => Do(e))), m(n, [`safetySettings`], e));
  }
  let C = h(t, [`tools`]);
  if (n !== void 0 && C != null) {
    let e = z(C);
    (Array.isArray(e) && (e = e.map((e) => Fo(R(e)))), m(n, [`tools`], e));
  }
  let w = h(t, [`toolConfig`]);
  if ((n !== void 0 && w != null && m(n, [`toolConfig`], No(w)), h(t, [`labels`]) !== void 0))
    throw Error(`labels parameter is not supported in Gemini API.`);
  let T = h(t, [`cachedContent`]);
  n !== void 0 && T != null && m(n, [`cachedContent`], B(e, T));
  let E = h(t, [`responseModalities`]);
  E != null && m(i, [`responseModalities`], E);
  let D = h(t, [`mediaResolution`]);
  D != null && m(i, [`mediaResolution`], D);
  let O = h(t, [`speechConfig`]);
  if ((O != null && m(i, [`speechConfig`], Zt(O)), h(t, [`audioTimestamp`]) !== void 0))
    throw Error(`audioTimestamp parameter is not supported in Gemini API.`);
  let k = h(t, [`thinkingConfig`]);
  k != null && m(i, [`thinkingConfig`], k);
  let A = h(t, [`imageConfig`]);
  A != null && m(i, [`imageConfig`], no(A));
  let j = h(t, [`enableEnhancedCivicAnswers`]);
  if ((j != null && m(i, [`enableEnhancedCivicAnswers`], j), h(t, [`modelArmorConfig`]) !== void 0))
    throw Error(`modelArmorConfig parameter is not supported in Gemini API.`);
  return i;
}
function Ta(e, t, n, r) {
  let i = {},
    a = h(t, [`systemInstruction`]);
  n !== void 0 && a != null && m(n, [`systemInstruction`], q(F(a)));
  let o = h(t, [`temperature`]);
  o != null && m(i, [`temperature`], o);
  let s = h(t, [`topP`]);
  s != null && m(i, [`topP`], s);
  let c = h(t, [`topK`]);
  c != null && m(i, [`topK`], c);
  let l = h(t, [`candidateCount`]);
  l != null && m(i, [`candidateCount`], l);
  let u = h(t, [`maxOutputTokens`]);
  u != null && m(i, [`maxOutputTokens`], u);
  let d = h(t, [`stopSequences`]);
  d != null && m(i, [`stopSequences`], d);
  let f = h(t, [`responseLogprobs`]);
  f != null && m(i, [`responseLogprobs`], f);
  let p = h(t, [`logprobs`]);
  p != null && m(i, [`logprobs`], p);
  let g = h(t, [`presencePenalty`]);
  g != null && m(i, [`presencePenalty`], g);
  let _ = h(t, [`frequencyPenalty`]);
  _ != null && m(i, [`frequencyPenalty`], _);
  let v = h(t, [`seed`]);
  v != null && m(i, [`seed`], v);
  let y = h(t, [`responseMimeType`]);
  y != null && m(i, [`responseMimeType`], y);
  let b = h(t, [`responseSchema`]);
  b != null && m(i, [`responseSchema`], Xt(b));
  let x = h(t, [`responseJsonSchema`]);
  x != null && m(i, [`responseJsonSchema`], x);
  let S = h(t, [`routingConfig`]);
  S != null && m(i, [`routingConfig`], S);
  let C = h(t, [`modelSelectionConfig`]);
  C != null && m(i, [`modelConfig`], C);
  let w = h(t, [`safetySettings`]);
  if (n !== void 0 && w != null) {
    let e = w;
    (Array.isArray(e) && (e = e.map((e) => e)), m(n, [`safetySettings`], e));
  }
  let T = h(t, [`tools`]);
  if (n !== void 0 && T != null) {
    let e = z(T);
    (Array.isArray(e) && (e = e.map((e) => Io(R(e)))), m(n, [`tools`], e));
  }
  let E = h(t, [`toolConfig`]);
  n !== void 0 && E != null && m(n, [`toolConfig`], Po(E));
  let D = h(t, [`labels`]);
  n !== void 0 && D != null && m(n, [`labels`], D);
  let O = h(t, [`cachedContent`]);
  n !== void 0 && O != null && m(n, [`cachedContent`], B(e, O));
  let k = h(t, [`responseModalities`]);
  k != null && m(i, [`responseModalities`], k);
  let A = h(t, [`mediaResolution`]);
  A != null && m(i, [`mediaResolution`], A);
  let j = h(t, [`speechConfig`]);
  j != null && m(i, [`speechConfig`], Zt(j));
  let ee = h(t, [`audioTimestamp`]);
  ee != null && m(i, [`audioTimestamp`], ee);
  let te = h(t, [`thinkingConfig`]);
  te != null && m(i, [`thinkingConfig`], te);
  let ne = h(t, [`imageConfig`]);
  if (
    (ne != null && m(i, [`imageConfig`], ro(ne)), h(t, [`enableEnhancedCivicAnswers`]) !== void 0)
  )
    throw Error(`enableEnhancedCivicAnswers parameter is not supported in Vertex AI.`);
  let re = h(t, [`modelArmorConfig`]);
  return (n !== void 0 && re != null && m(n, [`modelArmorConfig`], re), i);
}
function Ea(e, t, n) {
  let r = {},
    i = h(t, [`model`]);
  i != null && m(r, [`_url`, `model`], P(e, i));
  let a = h(t, [`contents`]);
  if (a != null) {
    let e = I(a);
    (Array.isArray(e) && (e = e.map((e) => Qi(e))), m(r, [`contents`], e));
  }
  let o = h(t, [`config`]);
  return (o != null && m(r, [`generationConfig`], wa(e, o, r)), r);
}
function Da(e, t, n) {
  let r = {},
    i = h(t, [`model`]);
  i != null && m(r, [`_url`, `model`], P(e, i));
  let a = h(t, [`contents`]);
  if (a != null) {
    let e = I(a);
    (Array.isArray(e) && (e = e.map((e) => q(e))), m(r, [`contents`], e));
  }
  let o = h(t, [`config`]);
  return (o != null && m(r, [`generationConfig`], Ta(e, o, r)), r);
}
function Oa(e, t) {
  let n = {},
    r = h(e, [`sdkHttpResponse`]);
  r != null && m(n, [`sdkHttpResponse`], r);
  let i = h(e, [`candidates`]);
  if (i != null) {
    let e = i;
    (Array.isArray(e) && (e = e.map((e) => Ki(e))), m(n, [`candidates`], e));
  }
  let a = h(e, [`modelVersion`]);
  a != null && m(n, [`modelVersion`], a);
  let o = h(e, [`promptFeedback`]);
  o != null && m(n, [`promptFeedback`], o);
  let s = h(e, [`responseId`]);
  s != null && m(n, [`responseId`], s);
  let c = h(e, [`usageMetadata`]);
  return (c != null && m(n, [`usageMetadata`], c), n);
}
function ka(e, t) {
  let n = {},
    r = h(e, [`sdkHttpResponse`]);
  r != null && m(n, [`sdkHttpResponse`], r);
  let i = h(e, [`candidates`]);
  if (i != null) {
    let e = i;
    (Array.isArray(e) && (e = e.map((e) => e)), m(n, [`candidates`], e));
  }
  let a = h(e, [`createTime`]);
  a != null && m(n, [`createTime`], a);
  let o = h(e, [`modelVersion`]);
  o != null && m(n, [`modelVersion`], o);
  let s = h(e, [`promptFeedback`]);
  s != null && m(n, [`promptFeedback`], s);
  let c = h(e, [`responseId`]);
  c != null && m(n, [`responseId`], c);
  let l = h(e, [`usageMetadata`]);
  return (l != null && m(n, [`usageMetadata`], l), n);
}
function Aa(e, t, n) {
  let r = {};
  if (h(e, [`outputGcsUri`]) !== void 0)
    throw Error(`outputGcsUri parameter is not supported in Gemini API.`);
  if (h(e, [`negativePrompt`]) !== void 0)
    throw Error(`negativePrompt parameter is not supported in Gemini API.`);
  let i = h(e, [`numberOfImages`]);
  t !== void 0 && i != null && m(t, [`parameters`, `sampleCount`], i);
  let a = h(e, [`aspectRatio`]);
  t !== void 0 && a != null && m(t, [`parameters`, `aspectRatio`], a);
  let o = h(e, [`guidanceScale`]);
  if (
    (t !== void 0 && o != null && m(t, [`parameters`, `guidanceScale`], o),
    h(e, [`seed`]) !== void 0)
  )
    throw Error(`seed parameter is not supported in Gemini API.`);
  let s = h(e, [`safetyFilterLevel`]);
  t !== void 0 && s != null && m(t, [`parameters`, `safetySetting`], s);
  let c = h(e, [`personGeneration`]);
  t !== void 0 && c != null && m(t, [`parameters`, `personGeneration`], c);
  let l = h(e, [`includeSafetyAttributes`]);
  t !== void 0 && l != null && m(t, [`parameters`, `includeSafetyAttributes`], l);
  let u = h(e, [`includeRaiReason`]);
  t !== void 0 && u != null && m(t, [`parameters`, `includeRaiReason`], u);
  let d = h(e, [`language`]);
  t !== void 0 && d != null && m(t, [`parameters`, `language`], d);
  let f = h(e, [`outputMimeType`]);
  t !== void 0 && f != null && m(t, [`parameters`, `outputOptions`, `mimeType`], f);
  let p = h(e, [`outputCompressionQuality`]);
  if (
    (t !== void 0 && p != null && m(t, [`parameters`, `outputOptions`, `compressionQuality`], p),
    h(e, [`addWatermark`]) !== void 0)
  )
    throw Error(`addWatermark parameter is not supported in Gemini API.`);
  if (h(e, [`labels`]) !== void 0) throw Error(`labels parameter is not supported in Gemini API.`);
  let g = h(e, [`imageSize`]);
  if (
    (t !== void 0 && g != null && m(t, [`parameters`, `sampleImageSize`], g),
    h(e, [`enhancePrompt`]) !== void 0)
  )
    throw Error(`enhancePrompt parameter is not supported in Gemini API.`);
  return r;
}
function ja(e, t, n) {
  let r = {},
    i = h(e, [`outputGcsUri`]);
  t !== void 0 && i != null && m(t, [`parameters`, `storageUri`], i);
  let a = h(e, [`negativePrompt`]);
  t !== void 0 && a != null && m(t, [`parameters`, `negativePrompt`], a);
  let o = h(e, [`numberOfImages`]);
  t !== void 0 && o != null && m(t, [`parameters`, `sampleCount`], o);
  let s = h(e, [`aspectRatio`]);
  t !== void 0 && s != null && m(t, [`parameters`, `aspectRatio`], s);
  let c = h(e, [`guidanceScale`]);
  t !== void 0 && c != null && m(t, [`parameters`, `guidanceScale`], c);
  let l = h(e, [`seed`]);
  t !== void 0 && l != null && m(t, [`parameters`, `seed`], l);
  let u = h(e, [`safetyFilterLevel`]);
  t !== void 0 && u != null && m(t, [`parameters`, `safetySetting`], u);
  let d = h(e, [`personGeneration`]);
  t !== void 0 && d != null && m(t, [`parameters`, `personGeneration`], d);
  let f = h(e, [`includeSafetyAttributes`]);
  t !== void 0 && f != null && m(t, [`parameters`, `includeSafetyAttributes`], f);
  let p = h(e, [`includeRaiReason`]);
  t !== void 0 && p != null && m(t, [`parameters`, `includeRaiReason`], p);
  let g = h(e, [`language`]);
  t !== void 0 && g != null && m(t, [`parameters`, `language`], g);
  let _ = h(e, [`outputMimeType`]);
  t !== void 0 && _ != null && m(t, [`parameters`, `outputOptions`, `mimeType`], _);
  let v = h(e, [`outputCompressionQuality`]);
  t !== void 0 && v != null && m(t, [`parameters`, `outputOptions`, `compressionQuality`], v);
  let y = h(e, [`addWatermark`]);
  t !== void 0 && y != null && m(t, [`parameters`, `addWatermark`], y);
  let b = h(e, [`labels`]);
  t !== void 0 && b != null && m(t, [`labels`], b);
  let x = h(e, [`imageSize`]);
  t !== void 0 && x != null && m(t, [`parameters`, `sampleImageSize`], x);
  let S = h(e, [`enhancePrompt`]);
  return (t !== void 0 && S != null && m(t, [`parameters`, `enhancePrompt`], S), r);
}
function Ma(e, t, n) {
  let r = {},
    i = h(t, [`model`]);
  i != null && m(r, [`_url`, `model`], P(e, i));
  let a = h(t, [`prompt`]);
  a != null && m(r, [`instances[0]`, `prompt`], a);
  let o = h(t, [`config`]);
  return (o != null && Aa(o, r), r);
}
function Na(e, t, n) {
  let r = {},
    i = h(t, [`model`]);
  i != null && m(r, [`_url`, `model`], P(e, i));
  let a = h(t, [`prompt`]);
  a != null && m(r, [`instances[0]`, `prompt`], a);
  let o = h(t, [`config`]);
  return (o != null && ja(o, r), r);
}
function Pa(e, t) {
  let n = {},
    r = h(e, [`sdkHttpResponse`]);
  r != null && m(n, [`sdkHttpResponse`], r);
  let i = h(e, [`predictions`]);
  if (i != null) {
    let e = i;
    (Array.isArray(e) && (e = e.map((e) => Ka(e))), m(n, [`generatedImages`], e));
  }
  let a = h(e, [`positivePromptSafetyAttributes`]);
  return (a != null && m(n, [`positivePromptSafetyAttributes`], To(a)), n);
}
function Fa(e, t) {
  let n = {},
    r = h(e, [`sdkHttpResponse`]);
  r != null && m(n, [`sdkHttpResponse`], r);
  let i = h(e, [`predictions`]);
  if (i != null) {
    let e = i;
    (Array.isArray(e) && (e = e.map((e) => qa(e))), m(n, [`generatedImages`], e));
  }
  let a = h(e, [`positivePromptSafetyAttributes`]);
  return (a != null && m(n, [`positivePromptSafetyAttributes`], Eo(a)), n);
}
function Ia(e, t, n) {
  let r = {},
    i = h(e, [`numberOfVideos`]);
  if (
    (t !== void 0 && i != null && m(t, [`parameters`, `sampleCount`], i),
    h(e, [`outputGcsUri`]) !== void 0)
  )
    throw Error(`outputGcsUri parameter is not supported in Gemini API.`);
  if (h(e, [`fps`]) !== void 0) throw Error(`fps parameter is not supported in Gemini API.`);
  let a = h(e, [`durationSeconds`]);
  if (
    (t !== void 0 && a != null && m(t, [`parameters`, `durationSeconds`], a),
    h(e, [`seed`]) !== void 0)
  )
    throw Error(`seed parameter is not supported in Gemini API.`);
  let o = h(e, [`aspectRatio`]);
  t !== void 0 && o != null && m(t, [`parameters`, `aspectRatio`], o);
  let s = h(e, [`resolution`]);
  t !== void 0 && s != null && m(t, [`parameters`, `resolution`], s);
  let c = h(e, [`personGeneration`]);
  if (
    (t !== void 0 && c != null && m(t, [`parameters`, `personGeneration`], c),
    h(e, [`pubsubTopic`]) !== void 0)
  )
    throw Error(`pubsubTopic parameter is not supported in Gemini API.`);
  let l = h(e, [`negativePrompt`]);
  t !== void 0 && l != null && m(t, [`parameters`, `negativePrompt`], l);
  let u = h(e, [`enhancePrompt`]);
  if (
    (t !== void 0 && u != null && m(t, [`parameters`, `enhancePrompt`], u),
    h(e, [`generateAudio`]) !== void 0)
  )
    throw Error(`generateAudio parameter is not supported in Gemini API.`);
  let d = h(e, [`lastFrame`]);
  t !== void 0 && d != null && m(t, [`instances[0]`, `lastFrame`], oo(d));
  let f = h(e, [`referenceImages`]);
  if (t !== void 0 && f != null) {
    let e = f;
    (Array.isArray(e) && (e = e.map((e) => Yo(e))), m(t, [`instances[0]`, `referenceImages`], e));
  }
  if (h(e, [`mask`]) !== void 0) throw Error(`mask parameter is not supported in Gemini API.`);
  if (h(e, [`compressionQuality`]) !== void 0)
    throw Error(`compressionQuality parameter is not supported in Gemini API.`);
  return r;
}
function La(e, t, n) {
  let r = {},
    i = h(e, [`numberOfVideos`]);
  t !== void 0 && i != null && m(t, [`parameters`, `sampleCount`], i);
  let a = h(e, [`outputGcsUri`]);
  t !== void 0 && a != null && m(t, [`parameters`, `storageUri`], a);
  let o = h(e, [`fps`]);
  t !== void 0 && o != null && m(t, [`parameters`, `fps`], o);
  let s = h(e, [`durationSeconds`]);
  t !== void 0 && s != null && m(t, [`parameters`, `durationSeconds`], s);
  let c = h(e, [`seed`]);
  t !== void 0 && c != null && m(t, [`parameters`, `seed`], c);
  let l = h(e, [`aspectRatio`]);
  t !== void 0 && l != null && m(t, [`parameters`, `aspectRatio`], l);
  let u = h(e, [`resolution`]);
  t !== void 0 && u != null && m(t, [`parameters`, `resolution`], u);
  let d = h(e, [`personGeneration`]);
  t !== void 0 && d != null && m(t, [`parameters`, `personGeneration`], d);
  let f = h(e, [`pubsubTopic`]);
  t !== void 0 && f != null && m(t, [`parameters`, `pubsubTopic`], f);
  let p = h(e, [`negativePrompt`]);
  t !== void 0 && p != null && m(t, [`parameters`, `negativePrompt`], p);
  let g = h(e, [`enhancePrompt`]);
  t !== void 0 && g != null && m(t, [`parameters`, `enhancePrompt`], g);
  let _ = h(e, [`generateAudio`]);
  t !== void 0 && _ != null && m(t, [`parameters`, `generateAudio`], _);
  let v = h(e, [`lastFrame`]);
  t !== void 0 && v != null && m(t, [`instances[0]`, `lastFrame`], J(v));
  let y = h(e, [`referenceImages`]);
  if (t !== void 0 && y != null) {
    let e = y;
    (Array.isArray(e) && (e = e.map((e) => Xo(e))), m(t, [`instances[0]`, `referenceImages`], e));
  }
  let b = h(e, [`mask`]);
  t !== void 0 && b != null && m(t, [`instances[0]`, `mask`], Jo(b));
  let x = h(e, [`compressionQuality`]);
  return (t !== void 0 && x != null && m(t, [`parameters`, `compressionQuality`], x), r);
}
function Ra(e, t) {
  let n = {},
    r = h(e, [`name`]);
  r != null && m(n, [`name`], r);
  let i = h(e, [`metadata`]);
  i != null && m(n, [`metadata`], i);
  let a = h(e, [`done`]);
  a != null && m(n, [`done`], a);
  let o = h(e, [`error`]);
  o != null && m(n, [`error`], o);
  let s = h(e, [`response`, `generateVideoResponse`]);
  return (s != null && m(n, [`response`], Ha(s)), n);
}
function za(e, t) {
  let n = {},
    r = h(e, [`name`]);
  r != null && m(n, [`name`], r);
  let i = h(e, [`metadata`]);
  i != null && m(n, [`metadata`], i);
  let a = h(e, [`done`]);
  a != null && m(n, [`done`], a);
  let o = h(e, [`error`]);
  o != null && m(n, [`error`], o);
  let s = h(e, [`response`]);
  return (s != null && m(n, [`response`], Ua(s)), n);
}
function Ba(e, t, n) {
  let r = {},
    i = h(t, [`model`]);
  i != null && m(r, [`_url`, `model`], P(e, i));
  let a = h(t, [`prompt`]);
  a != null && m(r, [`instances[0]`, `prompt`], a);
  let o = h(t, [`image`]);
  o != null && m(r, [`instances[0]`, `image`], oo(o));
  let s = h(t, [`video`]);
  s != null && m(r, [`instances[0]`, `video`], Zo(s));
  let c = h(t, [`source`]);
  c != null && Wa(c, r);
  let l = h(t, [`config`]);
  return (l != null && Ia(l, r), r);
}
function Va(e, t, n) {
  let r = {},
    i = h(t, [`model`]);
  i != null && m(r, [`_url`, `model`], P(e, i));
  let a = h(t, [`prompt`]);
  a != null && m(r, [`instances[0]`, `prompt`], a);
  let o = h(t, [`image`]);
  o != null && m(r, [`instances[0]`, `image`], J(o));
  let s = h(t, [`video`]);
  s != null && m(r, [`instances[0]`, `video`], Qo(s));
  let c = h(t, [`source`]);
  c != null && Ga(c, r);
  let l = h(t, [`config`]);
  return (l != null && La(l, r), r);
}
function Ha(e, t) {
  let n = {},
    r = h(e, [`generatedSamples`]);
  if (r != null) {
    let e = r;
    (Array.isArray(e) && (e = e.map((e) => Ya(e))), m(n, [`generatedVideos`], e));
  }
  let i = h(e, [`raiMediaFilteredCount`]);
  i != null && m(n, [`raiMediaFilteredCount`], i);
  let a = h(e, [`raiMediaFilteredReasons`]);
  return (a != null && m(n, [`raiMediaFilteredReasons`], a), n);
}
function Ua(e, t) {
  let n = {},
    r = h(e, [`videos`]);
  if (r != null) {
    let e = r;
    (Array.isArray(e) && (e = e.map((e) => Xa(e))), m(n, [`generatedVideos`], e));
  }
  let i = h(e, [`raiMediaFilteredCount`]);
  i != null && m(n, [`raiMediaFilteredCount`], i);
  let a = h(e, [`raiMediaFilteredReasons`]);
  return (a != null && m(n, [`raiMediaFilteredReasons`], a), n);
}
function Wa(e, t, n) {
  let r = {},
    i = h(e, [`prompt`]);
  t !== void 0 && i != null && m(t, [`instances[0]`, `prompt`], i);
  let a = h(e, [`image`]);
  t !== void 0 && a != null && m(t, [`instances[0]`, `image`], oo(a));
  let o = h(e, [`video`]);
  return (t !== void 0 && o != null && m(t, [`instances[0]`, `video`], Zo(o)), r);
}
function Ga(e, t, n) {
  let r = {},
    i = h(e, [`prompt`]);
  t !== void 0 && i != null && m(t, [`instances[0]`, `prompt`], i);
  let a = h(e, [`image`]);
  t !== void 0 && a != null && m(t, [`instances[0]`, `image`], J(a));
  let o = h(e, [`video`]);
  return (t !== void 0 && o != null && m(t, [`instances[0]`, `video`], Qo(o)), r);
}
function Ka(e, t) {
  let n = {},
    r = h(e, [`_self`]);
  r != null && m(n, [`image`], io(r));
  let i = h(e, [`raiFilteredReason`]);
  i != null && m(n, [`raiFilteredReason`], i);
  let a = h(e, [`_self`]);
  return (a != null && m(n, [`safetyAttributes`], To(a)), n);
}
function qa(e, t) {
  let n = {},
    r = h(e, [`_self`]);
  r != null && m(n, [`image`], ao(r));
  let i = h(e, [`raiFilteredReason`]);
  i != null && m(n, [`raiFilteredReason`], i);
  let a = h(e, [`_self`]);
  a != null && m(n, [`safetyAttributes`], Eo(a));
  let o = h(e, [`prompt`]);
  return (o != null && m(n, [`enhancedPrompt`], o), n);
}
function Ja(e, t) {
  let n = {},
    r = h(e, [`_self`]);
  r != null && m(n, [`mask`], ao(r));
  let i = h(e, [`labels`]);
  if (i != null) {
    let e = i;
    (Array.isArray(e) && (e = e.map((e) => e)), m(n, [`labels`], e));
  }
  return n;
}
function Ya(e, t) {
  let n = {},
    r = h(e, [`video`]);
  return (r != null && m(n, [`video`], Ko(r)), n);
}
function Xa(e, t) {
  let n = {},
    r = h(e, [`_self`]);
  return (r != null && m(n, [`video`], qo(r)), n);
}
function Za(e, t) {
  let n = {},
    r = h(e, [`modelSelectionConfig`]);
  r != null && m(n, [`modelConfig`], r);
  let i = h(e, [`responseJsonSchema`]);
  i != null && m(n, [`responseJsonSchema`], i);
  let a = h(e, [`audioTimestamp`]);
  a != null && m(n, [`audioTimestamp`], a);
  let o = h(e, [`candidateCount`]);
  o != null && m(n, [`candidateCount`], o);
  let s = h(e, [`enableAffectiveDialog`]);
  s != null && m(n, [`enableAffectiveDialog`], s);
  let c = h(e, [`frequencyPenalty`]);
  c != null && m(n, [`frequencyPenalty`], c);
  let l = h(e, [`logprobs`]);
  l != null && m(n, [`logprobs`], l);
  let u = h(e, [`maxOutputTokens`]);
  u != null && m(n, [`maxOutputTokens`], u);
  let d = h(e, [`mediaResolution`]);
  d != null && m(n, [`mediaResolution`], d);
  let f = h(e, [`presencePenalty`]);
  f != null && m(n, [`presencePenalty`], f);
  let p = h(e, [`responseLogprobs`]);
  p != null && m(n, [`responseLogprobs`], p);
  let g = h(e, [`responseMimeType`]);
  g != null && m(n, [`responseMimeType`], g);
  let _ = h(e, [`responseModalities`]);
  _ != null && m(n, [`responseModalities`], _);
  let v = h(e, [`responseSchema`]);
  v != null && m(n, [`responseSchema`], v);
  let y = h(e, [`routingConfig`]);
  y != null && m(n, [`routingConfig`], y);
  let b = h(e, [`seed`]);
  b != null && m(n, [`seed`], b);
  let x = h(e, [`speechConfig`]);
  x != null && m(n, [`speechConfig`], x);
  let S = h(e, [`stopSequences`]);
  S != null && m(n, [`stopSequences`], S);
  let C = h(e, [`temperature`]);
  C != null && m(n, [`temperature`], C);
  let w = h(e, [`thinkingConfig`]);
  w != null && m(n, [`thinkingConfig`], w);
  let T = h(e, [`topK`]);
  T != null && m(n, [`topK`], T);
  let E = h(e, [`topP`]);
  if ((E != null && m(n, [`topP`], E), h(e, [`enableEnhancedCivicAnswers`]) !== void 0))
    throw Error(`enableEnhancedCivicAnswers parameter is not supported in Vertex AI.`);
  return n;
}
function Qa(e, t, n) {
  let r = {},
    i = h(t, [`model`]);
  return (i != null && m(r, [`_url`, `name`], P(e, i)), r);
}
function $a(e, t, n) {
  let r = {},
    i = h(t, [`model`]);
  return (i != null && m(r, [`_url`, `name`], P(e, i)), r);
}
function eo(e, t) {
  let n = {},
    r = h(e, [`authConfig`]);
  r != null && m(n, [`authConfig`], Wi(r));
  let i = h(e, [`enableWidget`]);
  return (i != null && m(n, [`enableWidget`], i), n);
}
function to(e, t) {
  let n = {},
    r = h(e, [`searchTypes`]);
  if ((r != null && m(n, [`searchTypes`], r), h(e, [`blockingConfidence`]) !== void 0))
    throw Error(`blockingConfidence parameter is not supported in Gemini API.`);
  if (h(e, [`excludeDomains`]) !== void 0)
    throw Error(`excludeDomains parameter is not supported in Gemini API.`);
  let i = h(e, [`timeRangeFilter`]);
  return (i != null && m(n, [`timeRangeFilter`], i), n);
}
function no(e, t) {
  let n = {},
    r = h(e, [`aspectRatio`]);
  r != null && m(n, [`aspectRatio`], r);
  let i = h(e, [`imageSize`]);
  if ((i != null && m(n, [`imageSize`], i), h(e, [`personGeneration`]) !== void 0))
    throw Error(`personGeneration parameter is not supported in Gemini API.`);
  if (h(e, [`prominentPeople`]) !== void 0)
    throw Error(`prominentPeople parameter is not supported in Gemini API.`);
  if (h(e, [`outputMimeType`]) !== void 0)
    throw Error(`outputMimeType parameter is not supported in Gemini API.`);
  if (h(e, [`outputCompressionQuality`]) !== void 0)
    throw Error(`outputCompressionQuality parameter is not supported in Gemini API.`);
  if (h(e, [`imageOutputOptions`]) !== void 0)
    throw Error(`imageOutputOptions parameter is not supported in Gemini API.`);
  return n;
}
function ro(e, t) {
  let n = {},
    r = h(e, [`aspectRatio`]);
  r != null && m(n, [`aspectRatio`], r);
  let i = h(e, [`imageSize`]);
  i != null && m(n, [`imageSize`], i);
  let a = h(e, [`personGeneration`]);
  a != null && m(n, [`personGeneration`], a);
  let o = h(e, [`prominentPeople`]);
  o != null && m(n, [`prominentPeople`], o);
  let s = h(e, [`outputMimeType`]);
  s != null && m(n, [`imageOutputOptions`, `mimeType`], s);
  let c = h(e, [`outputCompressionQuality`]);
  c != null && m(n, [`imageOutputOptions`, `compressionQuality`], c);
  let l = h(e, [`imageOutputOptions`]);
  return (l != null && m(n, [`imageOutputOptions`], l), n);
}
function io(e, t) {
  let n = {},
    r = h(e, [`bytesBase64Encoded`]);
  r != null && m(n, [`imageBytes`], V(r));
  let i = h(e, [`mimeType`]);
  return (i != null && m(n, [`mimeType`], i), n);
}
function ao(e, t) {
  let n = {},
    r = h(e, [`gcsUri`]);
  r != null && m(n, [`gcsUri`], r);
  let i = h(e, [`bytesBase64Encoded`]);
  i != null && m(n, [`imageBytes`], V(i));
  let a = h(e, [`mimeType`]);
  return (a != null && m(n, [`mimeType`], a), n);
}
function oo(e, t) {
  let n = {};
  if (h(e, [`gcsUri`]) !== void 0) throw Error(`gcsUri parameter is not supported in Gemini API.`);
  let r = h(e, [`imageBytes`]);
  r != null && m(n, [`bytesBase64Encoded`], V(r));
  let i = h(e, [`mimeType`]);
  return (i != null && m(n, [`mimeType`], i), n);
}
function J(e, t) {
  let n = {},
    r = h(e, [`gcsUri`]);
  r != null && m(n, [`gcsUri`], r);
  let i = h(e, [`imageBytes`]);
  i != null && m(n, [`bytesBase64Encoded`], V(i));
  let a = h(e, [`mimeType`]);
  return (a != null && m(n, [`mimeType`], a), n);
}
function so(e, t, n, r) {
  let i = {},
    a = h(t, [`pageSize`]);
  n !== void 0 && a != null && m(n, [`_query`, `pageSize`], a);
  let o = h(t, [`pageToken`]);
  n !== void 0 && o != null && m(n, [`_query`, `pageToken`], o);
  let s = h(t, [`filter`]);
  n !== void 0 && s != null && m(n, [`_query`, `filter`], s);
  let c = h(t, [`queryBase`]);
  return (n !== void 0 && c != null && m(n, [`_url`, `models_url`], on(e, c)), i);
}
function co(e, t, n, r) {
  let i = {},
    a = h(t, [`pageSize`]);
  n !== void 0 && a != null && m(n, [`_query`, `pageSize`], a);
  let o = h(t, [`pageToken`]);
  n !== void 0 && o != null && m(n, [`_query`, `pageToken`], o);
  let s = h(t, [`filter`]);
  n !== void 0 && s != null && m(n, [`_query`, `filter`], s);
  let c = h(t, [`queryBase`]);
  return (n !== void 0 && c != null && m(n, [`_url`, `models_url`], on(e, c)), i);
}
function lo(e, t, n) {
  let r = {},
    i = h(t, [`config`]);
  return (i != null && so(e, i, r), r);
}
function uo(e, t, n) {
  let r = {},
    i = h(t, [`config`]);
  return (i != null && co(e, i, r), r);
}
function fo(e, t) {
  let n = {},
    r = h(e, [`sdkHttpResponse`]);
  r != null && m(n, [`sdkHttpResponse`], r);
  let i = h(e, [`nextPageToken`]);
  i != null && m(n, [`nextPageToken`], i);
  let a = h(e, [`_self`]);
  if (a != null) {
    let e = sn(a);
    (Array.isArray(e) && (e = e.map((e) => ho(e))), m(n, [`models`], e));
  }
  return n;
}
function po(e, t) {
  let n = {},
    r = h(e, [`sdkHttpResponse`]);
  r != null && m(n, [`sdkHttpResponse`], r);
  let i = h(e, [`nextPageToken`]);
  i != null && m(n, [`nextPageToken`], i);
  let a = h(e, [`_self`]);
  if (a != null) {
    let e = sn(a);
    (Array.isArray(e) && (e = e.map((e) => go(e))), m(n, [`models`], e));
  }
  return n;
}
function mo(e, t) {
  let n = {},
    r = h(e, [`maskMode`]);
  r != null && m(n, [`maskMode`], r);
  let i = h(e, [`segmentationClasses`]);
  i != null && m(n, [`maskClasses`], i);
  let a = h(e, [`maskDilation`]);
  return (a != null && m(n, [`dilation`], a), n);
}
function ho(e, t) {
  let n = {},
    r = h(e, [`name`]);
  r != null && m(n, [`name`], r);
  let i = h(e, [`displayName`]);
  i != null && m(n, [`displayName`], i);
  let a = h(e, [`description`]);
  a != null && m(n, [`description`], a);
  let o = h(e, [`version`]);
  o != null && m(n, [`version`], o);
  let s = h(e, [`_self`]);
  s != null && m(n, [`tunedModelInfo`], Lo(s));
  let c = h(e, [`inputTokenLimit`]);
  c != null && m(n, [`inputTokenLimit`], c);
  let l = h(e, [`outputTokenLimit`]);
  l != null && m(n, [`outputTokenLimit`], l);
  let u = h(e, [`supportedGenerationMethods`]);
  u != null && m(n, [`supportedActions`], u);
  let d = h(e, [`temperature`]);
  d != null && m(n, [`temperature`], d);
  let f = h(e, [`maxTemperature`]);
  f != null && m(n, [`maxTemperature`], f);
  let p = h(e, [`topP`]);
  p != null && m(n, [`topP`], p);
  let g = h(e, [`topK`]);
  g != null && m(n, [`topK`], g);
  let _ = h(e, [`thinking`]);
  return (_ != null && m(n, [`thinking`], _), n);
}
function go(e, t) {
  let n = {},
    r = h(e, [`name`]);
  r != null && m(n, [`name`], r);
  let i = h(e, [`displayName`]);
  i != null && m(n, [`displayName`], i);
  let a = h(e, [`description`]);
  a != null && m(n, [`description`], a);
  let o = h(e, [`versionId`]);
  o != null && m(n, [`version`], o);
  let s = h(e, [`deployedModels`]);
  if (s != null) {
    let e = s;
    (Array.isArray(e) && (e = e.map((e) => ya(e))), m(n, [`endpoints`], e));
  }
  let c = h(e, [`labels`]);
  c != null && m(n, [`labels`], c);
  let l = h(e, [`_self`]);
  l != null && m(n, [`tunedModelInfo`], Ro(l));
  let u = h(e, [`defaultCheckpointId`]);
  u != null && m(n, [`defaultCheckpointId`], u);
  let d = h(e, [`checkpoints`]);
  if (d != null) {
    let e = d;
    (Array.isArray(e) && (e = e.map((e) => e)), m(n, [`checkpoints`], e));
  }
  return n;
}
function _o(e, t) {
  let n = {},
    r = h(e, [`mediaResolution`]);
  r != null && m(n, [`mediaResolution`], r);
  let i = h(e, [`codeExecutionResult`]);
  i != null && m(n, [`codeExecutionResult`], i);
  let a = h(e, [`executableCode`]);
  a != null && m(n, [`executableCode`], a);
  let o = h(e, [`fileData`]);
  o != null && m(n, [`fileData`], ba(o));
  let s = h(e, [`functionCall`]);
  s != null && m(n, [`functionCall`], xa(s));
  let c = h(e, [`functionResponse`]);
  c != null && m(n, [`functionResponse`], c);
  let l = h(e, [`inlineData`]);
  l != null && m(n, [`inlineData`], Gi(l));
  let u = h(e, [`text`]);
  u != null && m(n, [`text`], u);
  let d = h(e, [`thought`]);
  d != null && m(n, [`thought`], d);
  let f = h(e, [`thoughtSignature`]);
  f != null && m(n, [`thoughtSignature`], f);
  let p = h(e, [`videoMetadata`]);
  p != null && m(n, [`videoMetadata`], p);
  let g = h(e, [`toolCall`]);
  g != null && m(n, [`toolCall`], g);
  let _ = h(e, [`toolResponse`]);
  return (_ != null && m(n, [`toolResponse`], _), n);
}
function vo(e, t) {
  let n = {},
    r = h(e, [`mediaResolution`]);
  r != null && m(n, [`mediaResolution`], r);
  let i = h(e, [`codeExecutionResult`]);
  i != null && m(n, [`codeExecutionResult`], i);
  let a = h(e, [`executableCode`]);
  a != null && m(n, [`executableCode`], a);
  let o = h(e, [`fileData`]);
  o != null && m(n, [`fileData`], o);
  let s = h(e, [`functionCall`]);
  s != null && m(n, [`functionCall`], s);
  let c = h(e, [`functionResponse`]);
  c != null && m(n, [`functionResponse`], c);
  let l = h(e, [`inlineData`]);
  l != null && m(n, [`inlineData`], l);
  let u = h(e, [`text`]);
  u != null && m(n, [`text`], u);
  let d = h(e, [`thought`]);
  d != null && m(n, [`thought`], d);
  let f = h(e, [`thoughtSignature`]);
  f != null && m(n, [`thoughtSignature`], f);
  let p = h(e, [`videoMetadata`]);
  if ((p != null && m(n, [`videoMetadata`], p), h(e, [`toolCall`]) !== void 0))
    throw Error(`toolCall parameter is not supported in Vertex AI.`);
  if (h(e, [`toolResponse`]) !== void 0)
    throw Error(`toolResponse parameter is not supported in Vertex AI.`);
  return n;
}
function yo(e, t) {
  let n = {},
    r = h(e, [`productImage`]);
  return (r != null && m(n, [`image`], J(r)), n);
}
function bo(e, t, n) {
  let r = {},
    i = h(e, [`numberOfImages`]);
  t !== void 0 && i != null && m(t, [`parameters`, `sampleCount`], i);
  let a = h(e, [`baseSteps`]);
  t !== void 0 && a != null && m(t, [`parameters`, `baseSteps`], a);
  let o = h(e, [`outputGcsUri`]);
  t !== void 0 && o != null && m(t, [`parameters`, `storageUri`], o);
  let s = h(e, [`seed`]);
  t !== void 0 && s != null && m(t, [`parameters`, `seed`], s);
  let c = h(e, [`safetyFilterLevel`]);
  t !== void 0 && c != null && m(t, [`parameters`, `safetySetting`], c);
  let l = h(e, [`personGeneration`]);
  t !== void 0 && l != null && m(t, [`parameters`, `personGeneration`], l);
  let u = h(e, [`addWatermark`]);
  t !== void 0 && u != null && m(t, [`parameters`, `addWatermark`], u);
  let d = h(e, [`outputMimeType`]);
  t !== void 0 && d != null && m(t, [`parameters`, `outputOptions`, `mimeType`], d);
  let f = h(e, [`outputCompressionQuality`]);
  t !== void 0 && f != null && m(t, [`parameters`, `outputOptions`, `compressionQuality`], f);
  let p = h(e, [`enhancePrompt`]);
  t !== void 0 && p != null && m(t, [`parameters`, `enhancePrompt`], p);
  let g = h(e, [`labels`]);
  return (t !== void 0 && g != null && m(t, [`labels`], g), r);
}
function xo(e, t, n) {
  let r = {},
    i = h(t, [`model`]);
  i != null && m(r, [`_url`, `model`], P(e, i));
  let a = h(t, [`source`]);
  a != null && Co(a, r);
  let o = h(t, [`config`]);
  return (o != null && bo(o, r), r);
}
function So(e, t) {
  let n = {},
    r = h(e, [`predictions`]);
  if (r != null) {
    let e = r;
    (Array.isArray(e) && (e = e.map((e) => qa(e))), m(n, [`generatedImages`], e));
  }
  return n;
}
function Co(e, t, n) {
  let r = {},
    i = h(e, [`prompt`]);
  t !== void 0 && i != null && m(t, [`instances[0]`, `prompt`], i);
  let a = h(e, [`personImage`]);
  t !== void 0 && a != null && m(t, [`instances[0]`, `personImage`, `image`], J(a));
  let o = h(e, [`productImages`]);
  if (t !== void 0 && o != null) {
    let e = o;
    (Array.isArray(e) && (e = e.map((e) => yo(e))), m(t, [`instances[0]`, `productImages`], e));
  }
  return r;
}
function wo(e, t) {
  let n = {},
    r = h(e, [`referenceImage`]);
  r != null && m(n, [`referenceImage`], J(r));
  let i = h(e, [`referenceId`]);
  i != null && m(n, [`referenceId`], i);
  let a = h(e, [`referenceType`]);
  a != null && m(n, [`referenceType`], a);
  let o = h(e, [`maskImageConfig`]);
  o != null && m(n, [`maskImageConfig`], mo(o));
  let s = h(e, [`controlImageConfig`]);
  s != null && m(n, [`controlImageConfig`], $i(s));
  let c = h(e, [`styleImageConfig`]);
  c != null && m(n, [`styleImageConfig`], c);
  let l = h(e, [`subjectImageConfig`]);
  return (l != null && m(n, [`subjectImageConfig`], l), n);
}
function To(e, t) {
  let n = {},
    r = h(e, [`safetyAttributes`, `categories`]);
  r != null && m(n, [`categories`], r);
  let i = h(e, [`safetyAttributes`, `scores`]);
  i != null && m(n, [`scores`], i);
  let a = h(e, [`contentType`]);
  return (a != null && m(n, [`contentType`], a), n);
}
function Eo(e, t) {
  let n = {},
    r = h(e, [`safetyAttributes`, `categories`]);
  r != null && m(n, [`categories`], r);
  let i = h(e, [`safetyAttributes`, `scores`]);
  i != null && m(n, [`scores`], i);
  let a = h(e, [`contentType`]);
  return (a != null && m(n, [`contentType`], a), n);
}
function Do(e, t) {
  let n = {},
    r = h(e, [`category`]);
  if ((r != null && m(n, [`category`], r), h(e, [`method`]) !== void 0))
    throw Error(`method parameter is not supported in Gemini API.`);
  let i = h(e, [`threshold`]);
  return (i != null && m(n, [`threshold`], i), n);
}
function Oo(e, t) {
  let n = {},
    r = h(e, [`image`]);
  return (r != null && m(n, [`image`], J(r)), n);
}
function ko(e, t, n) {
  let r = {},
    i = h(e, [`mode`]);
  t !== void 0 && i != null && m(t, [`parameters`, `mode`], i);
  let a = h(e, [`maxPredictions`]);
  t !== void 0 && a != null && m(t, [`parameters`, `maxPredictions`], a);
  let o = h(e, [`confidenceThreshold`]);
  t !== void 0 && o != null && m(t, [`parameters`, `confidenceThreshold`], o);
  let s = h(e, [`maskDilation`]);
  t !== void 0 && s != null && m(t, [`parameters`, `maskDilation`], s);
  let c = h(e, [`binaryColorThreshold`]);
  t !== void 0 && c != null && m(t, [`parameters`, `binaryColorThreshold`], c);
  let l = h(e, [`labels`]);
  return (t !== void 0 && l != null && m(t, [`labels`], l), r);
}
function Ao(e, t, n) {
  let r = {},
    i = h(t, [`model`]);
  i != null && m(r, [`_url`, `model`], P(e, i));
  let a = h(t, [`source`]);
  a != null && Mo(a, r);
  let o = h(t, [`config`]);
  return (o != null && ko(o, r), r);
}
function jo(e, t) {
  let n = {},
    r = h(e, [`predictions`]);
  if (r != null) {
    let e = r;
    (Array.isArray(e) && (e = e.map((e) => Ja(e))), m(n, [`generatedMasks`], e));
  }
  return n;
}
function Mo(e, t, n) {
  let r = {},
    i = h(e, [`prompt`]);
  t !== void 0 && i != null && m(t, [`instances[0]`, `prompt`], i);
  let a = h(e, [`image`]);
  t !== void 0 && a != null && m(t, [`instances[0]`, `image`], J(a));
  let o = h(e, [`scribbleImage`]);
  return (t !== void 0 && o != null && m(t, [`instances[0]`, `scribble`], Oo(o)), r);
}
function No(e, t) {
  let n = {},
    r = h(e, [`retrievalConfig`]);
  r != null && m(n, [`retrievalConfig`], r);
  let i = h(e, [`functionCallingConfig`]);
  i != null && m(n, [`functionCallingConfig`], Sa(i));
  let a = h(e, [`includeServerSideToolInvocations`]);
  return (a != null && m(n, [`includeServerSideToolInvocations`], a), n);
}
function Po(e, t) {
  let n = {},
    r = h(e, [`retrievalConfig`]);
  r != null && m(n, [`retrievalConfig`], r);
  let i = h(e, [`functionCallingConfig`]);
  if (
    (i != null && m(n, [`functionCallingConfig`], i),
    h(e, [`includeServerSideToolInvocations`]) !== void 0)
  )
    throw Error(`includeServerSideToolInvocations parameter is not supported in Vertex AI.`);
  return n;
}
function Fo(e, t) {
  let n = {};
  if (h(e, [`retrieval`]) !== void 0)
    throw Error(`retrieval parameter is not supported in Gemini API.`);
  let r = h(e, [`computerUse`]);
  r != null && m(n, [`computerUse`], r);
  let i = h(e, [`fileSearch`]);
  i != null && m(n, [`fileSearch`], i);
  let a = h(e, [`googleSearch`]);
  a != null && m(n, [`googleSearch`], to(a));
  let o = h(e, [`googleMaps`]);
  o != null && m(n, [`googleMaps`], eo(o));
  let s = h(e, [`codeExecution`]);
  if ((s != null && m(n, [`codeExecution`], s), h(e, [`enterpriseWebSearch`]) !== void 0))
    throw Error(`enterpriseWebSearch parameter is not supported in Gemini API.`);
  let c = h(e, [`functionDeclarations`]);
  if (c != null) {
    let e = c;
    (Array.isArray(e) && (e = e.map((e) => e)), m(n, [`functionDeclarations`], e));
  }
  let l = h(e, [`googleSearchRetrieval`]);
  if ((l != null && m(n, [`googleSearchRetrieval`], l), h(e, [`parallelAiSearch`]) !== void 0))
    throw Error(`parallelAiSearch parameter is not supported in Gemini API.`);
  let u = h(e, [`urlContext`]);
  u != null && m(n, [`urlContext`], u);
  let d = h(e, [`mcpServers`]);
  if (d != null) {
    let e = d;
    (Array.isArray(e) && (e = e.map((e) => e)), m(n, [`mcpServers`], e));
  }
  return n;
}
function Io(e, t) {
  let n = {},
    r = h(e, [`retrieval`]);
  r != null && m(n, [`retrieval`], r);
  let i = h(e, [`computerUse`]);
  if ((i != null && m(n, [`computerUse`], i), h(e, [`fileSearch`]) !== void 0))
    throw Error(`fileSearch parameter is not supported in Vertex AI.`);
  let a = h(e, [`googleSearch`]);
  a != null && m(n, [`googleSearch`], a);
  let o = h(e, [`googleMaps`]);
  o != null && m(n, [`googleMaps`], o);
  let s = h(e, [`codeExecution`]);
  s != null && m(n, [`codeExecution`], s);
  let c = h(e, [`enterpriseWebSearch`]);
  c != null && m(n, [`enterpriseWebSearch`], c);
  let l = h(e, [`functionDeclarations`]);
  if (l != null) {
    let e = l;
    (Array.isArray(e) && (e = e.map((e) => Ca(e))), m(n, [`functionDeclarations`], e));
  }
  let u = h(e, [`googleSearchRetrieval`]);
  u != null && m(n, [`googleSearchRetrieval`], u);
  let d = h(e, [`parallelAiSearch`]);
  d != null && m(n, [`parallelAiSearch`], d);
  let f = h(e, [`urlContext`]);
  if ((f != null && m(n, [`urlContext`], f), h(e, [`mcpServers`]) !== void 0))
    throw Error(`mcpServers parameter is not supported in Vertex AI.`);
  return n;
}
function Lo(e, t) {
  let n = {},
    r = h(e, [`baseModel`]);
  r != null && m(n, [`baseModel`], r);
  let i = h(e, [`createTime`]);
  i != null && m(n, [`createTime`], i);
  let a = h(e, [`updateTime`]);
  return (a != null && m(n, [`updateTime`], a), n);
}
function Ro(e, t) {
  let n = {},
    r = h(e, [`labels`, `google-vertex-llm-tuning-base-model-id`]);
  r != null && m(n, [`baseModel`], r);
  let i = h(e, [`createTime`]);
  i != null && m(n, [`createTime`], i);
  let a = h(e, [`updateTime`]);
  return (a != null && m(n, [`updateTime`], a), n);
}
function zo(e, t, n) {
  let r = {},
    i = h(e, [`displayName`]);
  t !== void 0 && i != null && m(t, [`displayName`], i);
  let a = h(e, [`description`]);
  t !== void 0 && a != null && m(t, [`description`], a);
  let o = h(e, [`defaultCheckpointId`]);
  return (t !== void 0 && o != null && m(t, [`defaultCheckpointId`], o), r);
}
function Bo(e, t, n) {
  let r = {},
    i = h(e, [`displayName`]);
  t !== void 0 && i != null && m(t, [`displayName`], i);
  let a = h(e, [`description`]);
  t !== void 0 && a != null && m(t, [`description`], a);
  let o = h(e, [`defaultCheckpointId`]);
  return (t !== void 0 && o != null && m(t, [`defaultCheckpointId`], o), r);
}
function Vo(e, t, n) {
  let r = {},
    i = h(t, [`model`]);
  i != null && m(r, [`_url`, `name`], P(e, i));
  let a = h(t, [`config`]);
  return (a != null && zo(a, r), r);
}
function Ho(e, t, n) {
  let r = {},
    i = h(t, [`model`]);
  i != null && m(r, [`_url`, `model`], P(e, i));
  let a = h(t, [`config`]);
  return (a != null && Bo(a, r), r);
}
function Uo(e, t, n) {
  let r = {},
    i = h(e, [`outputGcsUri`]);
  t !== void 0 && i != null && m(t, [`parameters`, `storageUri`], i);
  let a = h(e, [`safetyFilterLevel`]);
  t !== void 0 && a != null && m(t, [`parameters`, `safetySetting`], a);
  let o = h(e, [`personGeneration`]);
  t !== void 0 && o != null && m(t, [`parameters`, `personGeneration`], o);
  let s = h(e, [`includeRaiReason`]);
  t !== void 0 && s != null && m(t, [`parameters`, `includeRaiReason`], s);
  let c = h(e, [`outputMimeType`]);
  t !== void 0 && c != null && m(t, [`parameters`, `outputOptions`, `mimeType`], c);
  let l = h(e, [`outputCompressionQuality`]);
  t !== void 0 && l != null && m(t, [`parameters`, `outputOptions`, `compressionQuality`], l);
  let u = h(e, [`enhanceInputImage`]);
  t !== void 0 && u != null && m(t, [`parameters`, `upscaleConfig`, `enhanceInputImage`], u);
  let d = h(e, [`imagePreservationFactor`]);
  t !== void 0 && d != null && m(t, [`parameters`, `upscaleConfig`, `imagePreservationFactor`], d);
  let f = h(e, [`labels`]);
  t !== void 0 && f != null && m(t, [`labels`], f);
  let p = h(e, [`numberOfImages`]);
  t !== void 0 && p != null && m(t, [`parameters`, `sampleCount`], p);
  let g = h(e, [`mode`]);
  return (t !== void 0 && g != null && m(t, [`parameters`, `mode`], g), r);
}
function Wo(e, t, n) {
  let r = {},
    i = h(t, [`model`]);
  i != null && m(r, [`_url`, `model`], P(e, i));
  let a = h(t, [`image`]);
  a != null && m(r, [`instances[0]`, `image`], J(a));
  let o = h(t, [`upscaleFactor`]);
  o != null && m(r, [`parameters`, `upscaleConfig`, `upscaleFactor`], o);
  let s = h(t, [`config`]);
  return (s != null && Uo(s, r), r);
}
function Go(e, t) {
  let n = {},
    r = h(e, [`sdkHttpResponse`]);
  r != null && m(n, [`sdkHttpResponse`], r);
  let i = h(e, [`predictions`]);
  if (i != null) {
    let e = i;
    (Array.isArray(e) && (e = e.map((e) => qa(e))), m(n, [`generatedImages`], e));
  }
  return n;
}
function Ko(e, t) {
  let n = {},
    r = h(e, [`uri`]);
  r != null && m(n, [`uri`], r);
  let i = h(e, [`encodedVideo`]);
  i != null && m(n, [`videoBytes`], V(i));
  let a = h(e, [`encoding`]);
  return (a != null && m(n, [`mimeType`], a), n);
}
function qo(e, t) {
  let n = {},
    r = h(e, [`gcsUri`]);
  r != null && m(n, [`uri`], r);
  let i = h(e, [`bytesBase64Encoded`]);
  i != null && m(n, [`videoBytes`], V(i));
  let a = h(e, [`mimeType`]);
  return (a != null && m(n, [`mimeType`], a), n);
}
function Jo(e, t) {
  let n = {},
    r = h(e, [`image`]);
  r != null && m(n, [`_self`], J(r));
  let i = h(e, [`maskMode`]);
  return (i != null && m(n, [`maskMode`], i), n);
}
function Yo(e, t) {
  let n = {},
    r = h(e, [`image`]);
  r != null && m(n, [`image`], oo(r));
  let i = h(e, [`referenceType`]);
  return (i != null && m(n, [`referenceType`], i), n);
}
function Xo(e, t) {
  let n = {},
    r = h(e, [`image`]);
  r != null && m(n, [`image`], J(r));
  let i = h(e, [`referenceType`]);
  return (i != null && m(n, [`referenceType`], i), n);
}
function Zo(e, t) {
  let n = {},
    r = h(e, [`uri`]);
  r != null && m(n, [`uri`], r);
  let i = h(e, [`videoBytes`]);
  i != null && m(n, [`encodedVideo`], V(i));
  let a = h(e, [`mimeType`]);
  return (a != null && m(n, [`encoding`], a), n);
}
function Qo(e, t) {
  let n = {},
    r = h(e, [`uri`]);
  r != null && m(n, [`gcsUri`], r);
  let i = h(e, [`videoBytes`]);
  i != null && m(n, [`bytesBase64Encoded`], V(i));
  let a = h(e, [`mimeType`]);
  return (a != null && m(n, [`mimeType`], a), n);
}
function $o(e, t) {
  let n = {},
    r = h(e, [`displayName`]);
  return (t !== void 0 && r != null && m(t, [`displayName`], r), n);
}
function es(e) {
  let t = {},
    n = h(e, [`config`]);
  return (n != null && $o(n, t), t);
}
function ts(e, t) {
  let n = {},
    r = h(e, [`force`]);
  return (t !== void 0 && r != null && m(t, [`_query`, `force`], r), n);
}
function ns(e) {
  let t = {},
    n = h(e, [`name`]);
  n != null && m(t, [`_url`, `name`], n);
  let r = h(e, [`config`]);
  return (r != null && ts(r, t), t);
}
function rs(e) {
  let t = {},
    n = h(e, [`name`]);
  return (n != null && m(t, [`_url`, `name`], n), t);
}
function is(e, t) {
  let n = {},
    r = h(e, [`customMetadata`]);
  if (t !== void 0 && r != null) {
    let e = r;
    (Array.isArray(e) && (e = e.map((e) => e)), m(t, [`customMetadata`], e));
  }
  let i = h(e, [`chunkingConfig`]);
  return (t !== void 0 && i != null && m(t, [`chunkingConfig`], i), n);
}
function as(e) {
  let t = {},
    n = h(e, [`name`]);
  n != null && m(t, [`name`], n);
  let r = h(e, [`metadata`]);
  r != null && m(t, [`metadata`], r);
  let i = h(e, [`done`]);
  i != null && m(t, [`done`], i);
  let a = h(e, [`error`]);
  a != null && m(t, [`error`], a);
  let o = h(e, [`response`]);
  return (o != null && m(t, [`response`], ss(o)), t);
}
function os(e) {
  let t = {},
    n = h(e, [`fileSearchStoreName`]);
  n != null && m(t, [`_url`, `file_search_store_name`], n);
  let r = h(e, [`fileName`]);
  r != null && m(t, [`fileName`], r);
  let i = h(e, [`config`]);
  return (i != null && is(i, t), t);
}
function ss(e) {
  let t = {},
    n = h(e, [`sdkHttpResponse`]);
  n != null && m(t, [`sdkHttpResponse`], n);
  let r = h(e, [`parent`]);
  r != null && m(t, [`parent`], r);
  let i = h(e, [`documentName`]);
  return (i != null && m(t, [`documentName`], i), t);
}
function cs(e, t) {
  let n = {},
    r = h(e, [`pageSize`]);
  t !== void 0 && r != null && m(t, [`_query`, `pageSize`], r);
  let i = h(e, [`pageToken`]);
  return (t !== void 0 && i != null && m(t, [`_query`, `pageToken`], i), n);
}
function ls(e) {
  let t = {},
    n = h(e, [`config`]);
  return (n != null && cs(n, t), t);
}
function us(e) {
  let t = {},
    n = h(e, [`sdkHttpResponse`]);
  n != null && m(t, [`sdkHttpResponse`], n);
  let r = h(e, [`nextPageToken`]);
  r != null && m(t, [`nextPageToken`], r);
  let i = h(e, [`fileSearchStores`]);
  if (i != null) {
    let e = i;
    (Array.isArray(e) && (e = e.map((e) => e)), m(t, [`fileSearchStores`], e));
  }
  return t;
}
function ds(e, t) {
  let n = {},
    r = h(e, [`mimeType`]);
  t !== void 0 && r != null && m(t, [`mimeType`], r);
  let i = h(e, [`displayName`]);
  t !== void 0 && i != null && m(t, [`displayName`], i);
  let a = h(e, [`customMetadata`]);
  if (t !== void 0 && a != null) {
    let e = a;
    (Array.isArray(e) && (e = e.map((e) => e)), m(t, [`customMetadata`], e));
  }
  let o = h(e, [`chunkingConfig`]);
  return (t !== void 0 && o != null && m(t, [`chunkingConfig`], o), n);
}
function fs(e) {
  let t = {},
    n = h(e, [`fileSearchStoreName`]);
  n != null && m(t, [`_url`, `file_search_store_name`], n);
  let r = h(e, [`config`]);
  return (r != null && ds(r, t), t);
}
function ps(e) {
  let t = {},
    n = h(e, [`sdkHttpResponse`]);
  return (n != null && m(t, [`sdkHttpResponse`], n), t);
}
var ms = `Content-Type`,
  hs = `X-Server-Timeout`,
  gs = `User-Agent`,
  _s = `x-goog-api-client`,
  vs = `google-genai-sdk/1.46.0`,
  ys = `v1beta1`,
  bs = `v1beta`,
  xs = 5,
  Ss = [408, 429, 500, 502, 503, 504],
  Cs = class {
    constructor(e) {
      ((this.clientOptions = Object.assign({}, e)),
        (this.customBaseUrl = e.httpOptions?.baseUrl),
        this.clientOptions.vertexai &&
          (this.clientOptions.project && this.clientOptions.location
            ? (this.clientOptions.apiKey = void 0)
            : this.clientOptions.apiKey &&
              ((this.clientOptions.project = void 0), (this.clientOptions.location = void 0))));
      let t = {};
      if (this.clientOptions.vertexai) {
        if (
          (!this.clientOptions.location &&
            !this.clientOptions.apiKey &&
            !this.customBaseUrl &&
            (this.clientOptions.location = `global`),
          !(
            (this.clientOptions.project && this.clientOptions.location) ||
            this.clientOptions.apiKey
          ) && !this.customBaseUrl)
        )
          throw Error(
            `Authentication is not set up. Please provide either a project and location, or an API key, or a custom base URL.`,
          );
        let n = (e.project && e.location) || !!e.apiKey;
        (this.customBaseUrl && !n
          ? ((t.baseUrl = this.customBaseUrl),
            (this.clientOptions.project = void 0),
            (this.clientOptions.location = void 0))
          : this.clientOptions.apiKey || this.clientOptions.location === `global`
            ? (t.baseUrl = `https://aiplatform.googleapis.com/`)
            : this.clientOptions.project &&
              this.clientOptions.location &&
              (t.baseUrl = `https://${this.clientOptions.location}-aiplatform.googleapis.com/`),
          (t.apiVersion = this.clientOptions.apiVersion ?? ys));
      } else
        (this.clientOptions.apiKey ||
          console.warn(`API key should be set when using the Gemini API.`),
          (t.apiVersion = this.clientOptions.apiVersion ?? bs),
          (t.baseUrl = `https://generativelanguage.googleapis.com/`));
      ((t.headers = this.getDefaultHeaders()),
        (this.clientOptions.httpOptions = t),
        e.httpOptions &&
          (this.clientOptions.httpOptions = this.patchHttpOptions(t, e.httpOptions)));
    }
    isVertexAI() {
      return this.clientOptions.vertexai ?? !1;
    }
    getProject() {
      return this.clientOptions.project;
    }
    getLocation() {
      return this.clientOptions.location;
    }
    getCustomBaseUrl() {
      return this.customBaseUrl;
    }
    async getAuthHeaders() {
      let e = new Headers();
      return (await this.clientOptions.auth.addAuthHeaders(e), e);
    }
    getApiVersion() {
      if (this.clientOptions.httpOptions && this.clientOptions.httpOptions.apiVersion !== void 0)
        return this.clientOptions.httpOptions.apiVersion;
      throw Error(`API version is not set.`);
    }
    getBaseUrl() {
      if (this.clientOptions.httpOptions && this.clientOptions.httpOptions.baseUrl !== void 0)
        return this.clientOptions.httpOptions.baseUrl;
      throw Error(`Base URL is not set.`);
    }
    getRequestUrl() {
      return this.getRequestUrlInternal(this.clientOptions.httpOptions);
    }
    getHeaders() {
      if (this.clientOptions.httpOptions && this.clientOptions.httpOptions.headers !== void 0)
        return this.clientOptions.httpOptions.headers;
      throw Error(`Headers are not set.`);
    }
    getRequestUrlInternal(e) {
      if (!e || e.baseUrl === void 0 || e.apiVersion === void 0)
        throw Error(`HTTP options are not correctly set.`);
      let t = [e.baseUrl.endsWith(`/`) ? e.baseUrl.slice(0, -1) : e.baseUrl];
      return (e.apiVersion && e.apiVersion !== `` && t.push(e.apiVersion), t.join(`/`));
    }
    getBaseResourcePath() {
      return `projects/${this.clientOptions.project}/locations/${this.clientOptions.location}`;
    }
    getApiKey() {
      return this.clientOptions.apiKey;
    }
    getWebsocketBaseUrl() {
      let e = this.getBaseUrl(),
        t = new URL(e);
      return ((t.protocol = t.protocol == `http:` ? `ws` : `wss`), t.toString());
    }
    setBaseUrl(e) {
      if (this.clientOptions.httpOptions) this.clientOptions.httpOptions.baseUrl = e;
      else throw Error(`HTTP options are not correctly set.`);
    }
    constructUrl(e, t, n) {
      let r = [this.getRequestUrlInternal(t)];
      return (
        n && r.push(this.getBaseResourcePath()), e !== `` && r.push(e), new URL(`${r.join(`/`)}`)
      );
    }
    shouldPrependVertexProjectPath(e, t) {
      return !(
        (t.baseUrl && t.baseUrlResourceScope === Ne.COLLECTION) ||
        this.clientOptions.apiKey ||
        !this.clientOptions.vertexai ||
        e.path.startsWith(`projects/`) ||
        (e.httpMethod === `GET` && e.path.startsWith(`publishers/google/models`))
      );
    }
    async request(e) {
      let t = this.clientOptions.httpOptions;
      e.httpOptions && (t = this.patchHttpOptions(this.clientOptions.httpOptions, e.httpOptions));
      let n = this.shouldPrependVertexProjectPath(e, t),
        r = this.constructUrl(e.path, t, n);
      if (e.queryParams)
        for (let [t, n] of Object.entries(e.queryParams)) r.searchParams.append(t, String(n));
      let i = {};
      if (e.httpMethod === `GET`) {
        if (e.body && e.body !== `{}`)
          throw Error(
            `Request body should be empty for GET request, but got non empty request body`,
          );
      } else i.body = e.body;
      return (
        (i = await this.includeExtraHttpOptionsToRequestInit(i, t, r.toString(), e.abortSignal)),
        this.unaryApiCall(r, i, e.httpMethod)
      );
    }
    patchHttpOptions(e, t) {
      let n = JSON.parse(JSON.stringify(e));
      for (let [e, r] of Object.entries(t))
        typeof r == `object`
          ? (n[e] = Object.assign(Object.assign({}, n[e]), r))
          : r !== void 0 && (n[e] = r);
      return n;
    }
    async requestStream(e) {
      let t = this.clientOptions.httpOptions;
      e.httpOptions && (t = this.patchHttpOptions(this.clientOptions.httpOptions, e.httpOptions));
      let n = this.shouldPrependVertexProjectPath(e, t),
        r = this.constructUrl(e.path, t, n);
      (!r.searchParams.has(`alt`) || r.searchParams.get(`alt`) !== `sse`) &&
        r.searchParams.set(`alt`, `sse`);
      let i = {};
      return (
        (i.body = e.body),
        (i = await this.includeExtraHttpOptionsToRequestInit(i, t, r.toString(), e.abortSignal)),
        this.streamApiCall(r, i, e.httpMethod)
      );
    }
    async includeExtraHttpOptionsToRequestInit(e, t, n, r) {
      if ((t && t.timeout) || r) {
        let n = new AbortController(),
          i = n.signal;
        if (t.timeout && t?.timeout > 0) {
          let e = setTimeout(() => n.abort(), t.timeout);
          e && typeof e.unref == `function` && e.unref();
        }
        (r &&
          r.addEventListener(`abort`, () => {
            n.abort();
          }),
          (e.signal = i));
      }
      return (
        t && t.extraBody !== null && Ts(e, t.extraBody),
        (e.headers = await this.getHeadersInternal(t, n)),
        e
      );
    }
    async unaryApiCall(e, t, n) {
      return this.apiCall(e.toString(), Object.assign(Object.assign({}, t), { method: n }))
        .then(async (e) => (await ws(e), new lt(e)))
        .catch((e) => {
          throw e instanceof Error ? e : Error(JSON.stringify(e));
        });
    }
    async streamApiCall(e, t, n) {
      return this.apiCall(e.toString(), Object.assign(Object.assign({}, t), { method: n }))
        .then(async (e) => (await ws(e), this.processStreamResponse(e)))
        .catch((e) => {
          throw e instanceof Error ? e : Error(JSON.stringify(e));
        });
    }
    processStreamResponse(e) {
      return G(this, arguments, function* () {
        let t = e?.body?.getReader(),
          n = new TextDecoder(`utf-8`);
        if (!t) throw Error(`Response body is empty`);
        try {
          let r = ``,
            i = [
              `

`,
              `\r\r`,
              `\r
\r
`,
            ];
          for (;;) {
            let { done: a, value: o } = yield W(t.read());
            if (a) {
              if (r.trim().length > 0) throw Error(`Incomplete JSON segment at the end`);
              break;
            }
            let s = n.decode(o, { stream: !0 });
            try {
              let e = JSON.parse(s);
              if (`error` in e) {
                let t = JSON.parse(JSON.stringify(e.error)),
                  n = t.status,
                  r = t.code,
                  i = `got status: ${n}. ${JSON.stringify(e)}`;
                if (r >= 400 && r < 600) throw new ai({ message: i, status: r });
              }
            } catch (e) {
              if (e.name === `ApiError`) throw e;
            }
            r += s;
            let c = -1,
              l = 0;
            for (;;) {
              ((c = -1), (l = 0));
              for (let e of i) {
                let t = r.indexOf(e);
                t !== -1 && (c === -1 || t < c) && ((c = t), (l = e.length));
              }
              if (c === -1) break;
              let t = r.substring(0, c);
              r = r.substring(c + l);
              let n = t.trim();
              if (n.startsWith(`data:`)) {
                let t = n.substring(5).trim();
                try {
                  yield yield W(
                    new lt(
                      new Response(t, {
                        headers: e?.headers,
                        status: e?.status,
                        statusText: e?.statusText,
                      }),
                    ),
                  );
                } catch (e) {
                  throw Error(`exception parsing stream chunk ${t}. ${e}`);
                }
              }
            }
          }
        } finally {
          t.releaseLock();
        }
      });
    }
    async apiCall(e, t) {
      if (!this.clientOptions.httpOptions || !this.clientOptions.httpOptions.retryOptions)
        return fetch(e, t);
      let n = this.clientOptions.httpOptions.retryOptions;
      return (0, s.default)(
        async () => {
          let n = await fetch(e, t);
          if (n.ok) return n;
          throw Ss.includes(n.status)
            ? Error(`Retryable HTTP Error: ${n.statusText}`)
            : new s.AbortError(`Non-retryable exception ${n.statusText} sending request`);
        },
        { retries: (n.attempts ?? xs) - 1 },
      );
    }
    getDefaultHeaders() {
      let e = {},
        t = vs + ` ` + this.clientOptions.userAgentExtra;
      return ((e[gs] = t), (e[_s] = t), (e[ms] = `application/json`), e);
    }
    async getHeadersInternal(e, t) {
      let n = new Headers();
      if (e && e.headers) {
        for (let [t, r] of Object.entries(e.headers)) n.append(t, r);
        e.timeout && e.timeout > 0 && n.append(hs, String(Math.ceil(e.timeout / 1e3)));
      }
      return (await this.clientOptions.auth.addAuthHeaders(n, t), n);
    }
    getFileName(e) {
      let t = ``;
      return (
        typeof e == `string` &&
          ((t = e.replace(/[/\\]+$/, ``)), (t = t.split(/[/\\]/).pop() ?? ``)),
        t
      );
    }
    async uploadFile(e, t) {
      let n = {};
      (t != null && ((n.mimeType = t.mimeType), (n.name = t.name), (n.displayName = t.displayName)),
        n.name && !n.name.startsWith(`files/`) && (n.name = `files/${n.name}`));
      let r = this.clientOptions.uploader,
        i = await r.stat(e);
      n.sizeBytes = String(i.size);
      let a = t?.mimeType ?? i.type;
      if (a === void 0 || a === ``)
        throw Error(`Can not determine mimeType. Please provide mimeType in the config.`);
      n.mimeType = a;
      let o = { file: n },
        s = this.getFileName(e),
        c = p(`upload/v1beta/files`, o._url),
        l = await this.fetchUploadUrl(c, n.sizeBytes, n.mimeType, s, o, t?.httpOptions);
      return r.upload(e, l, this);
    }
    async uploadFileToFileSearchStore(e, t, n) {
      let r = this.clientOptions.uploader,
        i = await r.stat(t),
        a = String(i.size),
        o = n?.mimeType ?? i.type;
      if (o === void 0 || o === ``)
        throw Error(`Can not determine mimeType. Please provide mimeType in the config.`);
      let s = `upload/v1beta/${e}:uploadToFileSearchStore`,
        c = this.getFileName(t),
        l = {};
      n != null && ds(n, l);
      let u = await this.fetchUploadUrl(s, a, o, c, l, n?.httpOptions);
      return r.uploadToFileSearchStore(t, u, this);
    }
    async downloadFile(e) {
      await this.clientOptions.downloader.download(e, this);
    }
    async fetchUploadUrl(e, t, n, r, i, a) {
      let o = {};
      o = a || {
        apiVersion: ``,
        headers: Object.assign(
          {
            "Content-Type": `application/json`,
            "X-Goog-Upload-Protocol": `resumable`,
            "X-Goog-Upload-Command": `start`,
            "X-Goog-Upload-Header-Content-Length": `${t}`,
            "X-Goog-Upload-Header-Content-Type": `${n}`,
          },
          r ? { "X-Goog-Upload-File-Name": r } : {},
        ),
      };
      let s = await this.request({
        path: e,
        body: JSON.stringify(i),
        httpMethod: `POST`,
        httpOptions: o,
      });
      if (!s || !s?.headers)
        throw Error(
          `Server did not return an HttpResponse or the returned HttpResponse did not have headers.`,
        );
      let c = s?.headers?.[`x-goog-upload-url`];
      if (c === void 0)
        throw Error(
          `Failed to get upload url. Server did not return the x-google-upload-url in the headers`,
        );
      return c;
    }
  };
async function ws(e) {
  if (e === void 0) throw Error(`response is undefined`);
  if (!e.ok) {
    let t = e.status,
      n;
    n = e.headers.get(`content-type`)?.includes(`application/json`)
      ? await e.json()
      : { error: { message: await e.text(), code: e.status, status: e.statusText } };
    let r = JSON.stringify(n);
    throw t >= 400 && t < 600 ? new ai({ message: r, status: t }) : Error(r);
  }
}
function Ts(e, t) {
  if (!t || Object.keys(t).length === 0) return;
  if (e.body instanceof Blob) {
    console.warn(
      `includeExtraBodyToRequestInit: extraBody provided but current request body is a Blob. extraBody will be ignored as merging is not supported for Blob bodies.`,
    );
    return;
  }
  let n = {};
  if (typeof e.body == `string` && e.body.length > 0)
    try {
      let t = JSON.parse(e.body);
      if (typeof t == `object` && t && !Array.isArray(t)) n = t;
      else {
        console.warn(
          `includeExtraBodyToRequestInit: Original request body is valid JSON but not a non-array object. Skip applying extraBody to the request body.`,
        );
        return;
      }
    } catch {
      console.warn(
        `includeExtraBodyToRequestInit: Original request body is not valid JSON. Skip applying extraBody to the request body.`,
      );
      return;
    }
  function r(e, t) {
    let n = Object.assign({}, e);
    for (let e in t)
      if (Object.prototype.hasOwnProperty.call(t, e)) {
        let i = t[e],
          a = n[e];
        i &&
        typeof i == `object` &&
        !Array.isArray(i) &&
        a &&
        typeof a == `object` &&
        !Array.isArray(a)
          ? (n[e] = r(a, i))
          : (a &&
              i &&
              typeof a != typeof i &&
              console.warn(
                `includeExtraBodyToRequestInit:deepMerge: Type mismatch for key "${e}". Original type: ${typeof a}, New type: ${typeof i}. Overwriting.`,
              ),
            (n[e] = i));
      }
    return n;
  }
  let i = r(n, t);
  e.body = JSON.stringify(i);
}
var Es = `mcp_used/unknown`,
  Ds = !1;
function Os(e) {
  for (let t of e) if (As(t) || (typeof t == `object` && `inputSchema` in t)) return !0;
  return Ds;
}
function ks(e) {
  e[_s] = ((e[_s] ?? ``) + ` ${Es}`).trimStart();
}
function As(e) {
  return typeof e == `object` && !!e && e instanceof Ms;
}
function js(e) {
  return G(this, arguments, function* (e, t = 100) {
    let n,
      r = 0;
    for (; r < t; ) {
      let t = yield W(e.listTools({ cursor: n }));
      for (let e of t.tools) (yield yield W(e), r++);
      if (!t.nextCursor) break;
      n = t.nextCursor;
    }
  });
}
var Ms = class e {
  constructor(e = [], t) {
    ((this.mcpTools = []),
      (this.functionNameToMcpClient = {}),
      (this.mcpClients = e),
      (this.config = t));
  }
  static create(t, n) {
    return new e(t, n);
  }
  async initialize() {
    var e, t, n, r;
    if (this.mcpTools.length > 0) return;
    let i = {},
      a = [];
    for (let l of this.mcpClients)
      try {
        for (
          var o = !0, s = ((t = void 0), K(js(l))), c;
          (c = await s.next()), (e = c.done), !e;
          o = !0
        ) {
          ((r = c.value), (o = !1));
          let e = r;
          a.push(e);
          let t = e.name;
          if (i[t])
            throw Error(
              `Duplicate function name ${t} found in MCP tools. Please ensure function names are unique.`,
            );
          i[t] = l;
        }
      } catch (e) {
        t = { error: e };
      } finally {
        try {
          !o && !e && (n = s.return) && (await n.call(s));
        } finally {
          if (t) throw t.error;
        }
      }
    ((this.mcpTools = a), (this.functionNameToMcpClient = i));
  }
  async tool() {
    return (await this.initialize(), un(this.mcpTools, this.config));
  }
  async callTool(e) {
    await this.initialize();
    let t = [];
    for (let n of e)
      if (n.name in this.functionNameToMcpClient) {
        let e = this.functionNameToMcpClient[n.name],
          r;
        this.config.timeout && (r = { timeout: this.config.timeout });
        let i = await e.callTool({ name: n.name, arguments: n.args }, void 0, r);
        t.push({ functionResponse: { name: n.name, response: i.isError ? { error: i } : i } });
      }
    return t;
  }
};
async function Ns(e, t, n) {
  let r = new It(),
    i;
  ((i = n.data instanceof Blob ? JSON.parse(await n.data.text()) : JSON.parse(n.data)),
    Object.assign(r, i),
    t(r));
}
var Ps = class {
    constructor(e, t, n) {
      ((this.apiClient = e), (this.auth = t), (this.webSocketFactory = n));
    }
    async connect(e) {
      if (this.apiClient.isVertexAI()) throw Error(`Live music is not supported for Vertex AI.`);
      console.warn(`Live music generation is experimental and may change in future versions.`);
      let t = this.apiClient.getWebsocketBaseUrl(),
        n = this.apiClient.getApiVersion(),
        r = Ls(this.apiClient.getDefaultHeaders()),
        i = `${t}/ws/google.ai.generativelanguage.${n}.GenerativeService.BidiGenerateMusic?key=${this.apiClient.getApiKey()}`,
        a = () => {},
        o = new Promise((e) => {
          a = e;
        }),
        s = e.callbacks,
        c = function () {
          a({});
        },
        l = this.apiClient,
        u = {
          onopen: c,
          onmessage: (e) => {
            Ns(l, s.onmessage, e);
          },
          onerror: s?.onerror ?? function (e) {},
          onclose: s?.onclose ?? function (e) {},
        },
        d = this.webSocketFactory.create(i, Is(r), u);
      (d.connect(), await o);
      let f = { setup: { model: P(this.apiClient, e.model) } };
      return (d.send(JSON.stringify(f)), new Fs(d, this.apiClient));
    }
  },
  Fs = class {
    constructor(e, t) {
      ((this.conn = e), (this.apiClient = t));
    }
    async setWeightedPrompts(e) {
      if (!e.weightedPrompts || Object.keys(e.weightedPrompts).length === 0)
        throw Error(`Weighted prompts must be set and contain at least one entry.`);
      let t = Ni(e);
      this.conn.send(JSON.stringify({ clientContent: t }));
    }
    async setMusicGenerationConfig(e) {
      e.musicGenerationConfig ||= {};
      let t = Mi(e);
      this.conn.send(JSON.stringify(t));
    }
    sendPlaybackControl(e) {
      let t = { playbackControl: e };
      this.conn.send(JSON.stringify(t));
    }
    play() {
      this.sendPlaybackControl(ct.PLAY);
    }
    pause() {
      this.sendPlaybackControl(ct.PAUSE);
    }
    stop() {
      this.sendPlaybackControl(ct.STOP);
    }
    resetContext() {
      this.sendPlaybackControl(ct.RESET_CONTEXT);
    }
    close() {
      this.conn.close();
    }
  };
function Is(e) {
  let t = {};
  return (
    e.forEach((e, n) => {
      t[n] = e;
    }),
    t
  );
}
function Ls(e) {
  let t = new Headers();
  for (let [n, r] of Object.entries(e)) t.append(n, r);
  return t;
}
var Rs =
  "FunctionResponse request must have an `id` field from the response of a ToolCall.FunctionalCalls in Google AI.";
async function zs(e, t, n) {
  let r = new Ft(),
    i;
  i =
    n.data instanceof Blob
      ? await n.data.text()
      : n.data instanceof ArrayBuffer
        ? new TextDecoder().decode(n.data)
        : n.data;
  let a = JSON.parse(i);
  if (e.isVertexAI()) {
    let e = Ii(a);
    Object.assign(r, e);
  } else Object.assign(r, a);
  t(r);
}
var Bs = class {
    constructor(e, t, n) {
      ((this.apiClient = e),
        (this.auth = t),
        (this.webSocketFactory = n),
        (this.music = new Ps(this.apiClient, this.auth, this.webSocketFactory)));
    }
    async connect(e) {
      if (e.config && e.config.httpOptions)
        throw Error(
          `The Live module does not support httpOptions at request-level in LiveConnectConfig yet. Please use the client-level httpOptions configuration instead.`,
        );
      let t = this.apiClient.getWebsocketBaseUrl(),
        n = this.apiClient.getApiVersion(),
        r,
        i = this.apiClient.getHeaders();
      e.config && e.config.tools && Os(e.config.tools) && ks(i);
      let a = Ws(i);
      if (this.apiClient.isVertexAI()) {
        let e = this.apiClient.getProject(),
          i = this.apiClient.getLocation(),
          o = this.apiClient.getApiKey(),
          s = (!!e && !!i) || !!o;
        this.apiClient.getCustomBaseUrl() && !s
          ? (r = t)
          : ((r = `${t}/ws/google.cloud.aiplatform.${n}.LlmBidiService/BidiGenerateContent`),
            await this.auth.addAuthHeaders(a, r));
      } else {
        let e = this.apiClient.getApiKey(),
          i = `BidiGenerateContent`,
          a = `key`;
        (e?.startsWith(`auth_tokens/`) &&
          (console.warn(
            `Warning: Ephemeral token support is experimental and may change in future versions.`,
          ),
          n !== `v1alpha` &&
            console.warn(
              `Warning: The SDK's ephemeral token support is in v1alpha only. Please use const ai = new GoogleGenAI({apiKey: token.name, httpOptions: { apiVersion: 'v1alpha' }}); before session connection.`,
            ),
          (i = `BidiGenerateContentConstrained`),
          (a = `access_token`)),
          (r = `${t}/ws/google.ai.generativelanguage.${n}.GenerativeService.${i}?${a}=${e}`));
      }
      let o = () => {},
        s = new Promise((e) => {
          o = e;
        }),
        c = e.callbacks,
        l = function () {
          var e;
          ((e = c?.onopen) == null || e.call(c), o({}));
        },
        u = this.apiClient,
        d = {
          onopen: l,
          onmessage: (e) => {
            zs(u, c.onmessage, e);
          },
          onerror: c?.onerror ?? function (e) {},
          onclose: c?.onclose ?? function (e) {},
        },
        f = this.webSocketFactory.create(r, Us(a), d);
      (f.connect(), await s);
      let p = P(this.apiClient, e.model);
      if (this.apiClient.isVertexAI() && p.startsWith(`publishers/`)) {
        let e = this.apiClient.getProject(),
          t = this.apiClient.getLocation();
        e && t && (p = `projects/${e}/locations/${t}/` + p);
      }
      let m = {};
      (this.apiClient.isVertexAI() &&
        e.config?.responseModalities === void 0 &&
        (e.config === void 0
          ? (e.config = { responseModalities: [Se.AUDIO] })
          : (e.config.responseModalities = [Se.AUDIO])),
        e.config?.generationConfig &&
          console.warn(
            "Setting `LiveConnectConfig.generation_config` is deprecated, please set the fields on `LiveConnectConfig` directly. This will become an error in a future version (not before Q3 2025).",
          ));
      let h = e.config?.tools ?? [],
        g = [];
      for (let e of h)
        if (this.isCallableTool(e)) {
          let t = e;
          g.push(await t.tool());
        } else g.push(e);
      g.length > 0 && (e.config.tools = g);
      let _ = { model: p, config: e.config, callbacks: e.callbacks };
      return (
        (m = this.apiClient.isVertexAI() ? ji(this.apiClient, _) : Ai(this.apiClient, _)),
        delete m.config,
        f.send(JSON.stringify(m)),
        new Hs(f, this.apiClient)
      );
    }
    isCallableTool(e) {
      return `callTool` in e && typeof e.callTool == `function`;
    }
  },
  Vs = { turnComplete: !0 },
  Hs = class {
    constructor(e, t) {
      ((this.conn = e), (this.apiClient = t));
    }
    tLiveClientContent(e, t) {
      if (t.turns !== null && t.turns !== void 0) {
        let n = [];
        try {
          ((n = I(t.turns)), e.isVertexAI() || (n = n.map((e) => Qi(e))));
        } catch {
          throw Error(`Failed to parse client content "turns", type: '${typeof t.turns}'`);
        }
        return { clientContent: { turns: n, turnComplete: t.turnComplete } };
      }
      return { clientContent: { turnComplete: t.turnComplete } };
    }
    tLiveClienttToolResponse(e, t) {
      let n = [];
      if (
        t.functionResponses == null ||
        ((n = Array.isArray(t.functionResponses) ? t.functionResponses : [t.functionResponses]),
        n.length === 0)
      )
        throw Error(`functionResponses is required.`);
      for (let t of n) {
        if (typeof t != `object` || !t || !(`name` in t) || !(`response` in t))
          throw Error(`Could not parse function response, type '${typeof t}'.`);
        if (!e.isVertexAI() && !(`id` in t)) throw Error(Rs);
      }
      return { toolResponse: { functionResponses: n } };
    }
    sendClientContent(e) {
      e = Object.assign(Object.assign({}, Vs), e);
      let t = this.tLiveClientContent(this.apiClient, e);
      this.conn.send(JSON.stringify(t));
    }
    sendRealtimeInput(e) {
      let t = {};
      ((t = this.apiClient.isVertexAI() ? { realtimeInput: Fi(e) } : { realtimeInput: Pi(e) }),
        this.conn.send(JSON.stringify(t)));
    }
    sendToolResponse(e) {
      if (e.functionResponses == null) throw Error(`Tool response parameters are required.`);
      let t = this.tLiveClienttToolResponse(this.apiClient, e);
      this.conn.send(JSON.stringify(t));
    }
    close() {
      this.conn.close();
    }
  };
function Us(e) {
  let t = {};
  return (
    e.forEach((e, n) => {
      t[n] = e;
    }),
    t
  );
}
function Ws(e) {
  let t = new Headers();
  for (let [n, r] of Object.entries(e)) t.append(n, r);
  return t;
}
var Gs = 10;
function Ks(e) {
  if (e?.automaticFunctionCalling?.disable) return !0;
  let t = !1;
  for (let n of e?.tools ?? [])
    if (qs(n)) {
      t = !0;
      break;
    }
  if (!t) return !0;
  let n = e?.automaticFunctionCalling?.maximumRemoteCalls;
  return (n && (n < 0 || !Number.isInteger(n))) || n == 0
    ? (console.warn(
        `Invalid maximumRemoteCalls value provided for automatic function calling. Disabled automatic function calling. Please provide a valid integer value greater than 0. maximumRemoteCalls provided:`,
        n,
      ),
      !0)
    : !1;
}
function qs(e) {
  return `callTool` in e && typeof e.callTool == `function`;
}
function Js(e) {
  return e.config?.tools?.some((e) => qs(e)) ?? !1;
}
function Ys(e) {
  let t = [];
  return (
    e?.config?.tools &&
      e.config.tools.forEach((e, n) => {
        if (qs(e)) return;
        let r = e;
        r.functionDeclarations && r.functionDeclarations.length > 0 && t.push(n);
      }),
    t
  );
}
function Xs(e) {
  return !e?.automaticFunctionCalling?.ignoreCallHistory;
}
var Zs = class extends f {
    constructor(e) {
      (super(),
        (this.apiClient = e),
        (this.embedContent = async (e) => {
          if (!this.apiClient.isVertexAI()) return await this.embedContentInternal(e);
          if (
            (e.model.includes(`gemini`) && e.model !== `gemini-embedding-001`) ||
            e.model.includes(`maas`)
          ) {
            let t = I(e.contents);
            if (t.length > 1)
              throw Error(
                `The embedContent API for this model only supports one content at a time.`,
              );
            let n = Object.assign(Object.assign({}, e), {
              content: t[0],
              embeddingApiType: Le.EMBED_CONTENT,
            });
            return await this.embedContentInternal(n);
          } else {
            let t = Object.assign(Object.assign({}, e), { embeddingApiType: Le.PREDICT });
            return await this.embedContentInternal(t);
          }
        }),
        (this.generateContent = async (e) => {
          let t = await this.processParamsMaybeAddMcpUsage(e);
          if ((this.maybeMoveToResponseJsonSchem(e), !Js(e) || Ks(e.config)))
            return await this.generateContentInternal(t);
          let n = Ys(e);
          if (n.length > 0) {
            let e = n.map((e) => `tools[${e}]`).join(`, `);
            throw Error(
              `Automatic function calling with CallableTools (or MCP objects) and basic FunctionDeclarations is not yet supported. Incompatible tools found at ${e}.`,
            );
          }
          let r,
            i,
            a = I(t.contents),
            o = t.config?.automaticFunctionCalling?.maximumRemoteCalls ?? Gs,
            s = 0;
          for (
            ;
            s < o &&
            ((r = await this.generateContentInternal(t)),
            !(!r.functionCalls || r.functionCalls.length === 0));
          ) {
            let n = r.candidates[0].content,
              o = [];
            for (let t of e.config?.tools ?? [])
              if (qs(t)) {
                let e = await t.callTool(r.functionCalls);
                o.push(...e);
              }
            (s++,
              (i = { role: `user`, parts: o }),
              (t.contents = I(t.contents)),
              t.contents.push(n),
              t.contents.push(i),
              Xs(t.config) && (a.push(n), a.push(i)));
          }
          return (Xs(t.config) && (r.automaticFunctionCallingHistory = a), r);
        }),
        (this.generateContentStream = async (e) => {
          if ((this.maybeMoveToResponseJsonSchem(e), Ks(e.config))) {
            let t = await this.processParamsMaybeAddMcpUsage(e);
            return await this.generateContentStreamInternal(t);
          }
          let t = Ys(e);
          if (t.length > 0) {
            let e = t.map((e) => `tools[${e}]`).join(`, `);
            throw Error(
              `Incompatible tools found at ${e}. Automatic function calling with CallableTools (or MCP objects) and basic FunctionDeclarations" is not yet supported.`,
            );
          }
          let n = e?.config?.toolConfig?.functionCallingConfig?.streamFunctionCallArguments,
            r = e?.config?.automaticFunctionCalling?.disable;
          if (n && !r)
            throw Error(
              `Running in streaming mode with 'streamFunctionCallArguments' enabled, this feature is not compatible with automatic function calling (AFC). Please set 'config.automaticFunctionCalling.disable' to true to disable AFC or leave 'config.toolConfig.functionCallingConfig.streamFunctionCallArguments' to be undefined or set to false to disable streaming function call arguments feature.`,
            );
          return await this.processAfcStream(e);
        }),
        (this.generateImages = async (e) =>
          await this.generateImagesInternal(e).then((e) => {
            let t,
              n = [];
            if (e?.generatedImages)
              for (let r of e.generatedImages)
                r && r?.safetyAttributes && r?.safetyAttributes?.contentType === `Positive Prompt`
                  ? (t = r?.safetyAttributes)
                  : n.push(r);
            let r;
            return (
              (r = t
                ? {
                    generatedImages: n,
                    positivePromptSafetyAttributes: t,
                    sdkHttpResponse: e.sdkHttpResponse,
                  }
                : { generatedImages: n, sdkHttpResponse: e.sdkHttpResponse }),
              r
            );
          })),
        (this.list = async (e) => {
          let t = { config: Object.assign(Object.assign({}, { queryBase: !0 }), e?.config) };
          if (this.apiClient.isVertexAI() && !t.config.queryBase) {
            if (t.config?.filter)
              throw Error(`Filtering tuned models list for Vertex AI is not currently supported`);
            t.config.filter = `labels.tune-type:*`;
          }
          return new U(
            H.PAGED_ITEM_MODELS,
            (e) => this.listInternal(e),
            await this.listInternal(t),
            t,
          );
        }),
        (this.editImage = async (e) => {
          let t = { model: e.model, prompt: e.prompt, referenceImages: [], config: e.config };
          return (
            e.referenceImages &&
              e.referenceImages &&
              (t.referenceImages = e.referenceImages.map((e) => e.toReferenceImageAPI())),
            await this.editImageInternal(t)
          );
        }),
        (this.upscaleImage = async (e) => {
          let t = { numberOfImages: 1, mode: `upscale` };
          e.config && (t = Object.assign(Object.assign({}, t), e.config));
          let n = { model: e.model, image: e.image, upscaleFactor: e.upscaleFactor, config: t };
          return await this.upscaleImageInternal(n);
        }),
        (this.generateVideos = async (e) => {
          if ((e.prompt || e.image || e.video) && e.source)
            throw Error(
              `Source and prompt/image/video are mutually exclusive. Please only use source.`,
            );
          return (
            this.apiClient.isVertexAI() ||
              (e.video?.uri && e.video?.videoBytes
                ? (e.video = { uri: e.video.uri, mimeType: e.video.mimeType })
                : e.source?.video?.uri &&
                  e.source?.video?.videoBytes &&
                  (e.source.video = {
                    uri: e.source.video.uri,
                    mimeType: e.source.video.mimeType,
                  })),
            await this.generateVideosInternal(e)
          );
        }));
    }
    maybeMoveToResponseJsonSchem(e) {
      e.config &&
        e.config.responseSchema &&
        (e.config.responseJsonSchema ||
          (Object.keys(e.config.responseSchema).includes(`$schema`) &&
            ((e.config.responseJsonSchema = e.config.responseSchema),
            delete e.config.responseSchema)));
    }
    async processParamsMaybeAddMcpUsage(e) {
      let t = e.config?.tools;
      if (!t) return e;
      let n = await Promise.all(t.map(async (e) => (qs(e) ? await e.tool() : e))),
        r = {
          model: e.model,
          contents: e.contents,
          config: Object.assign(Object.assign({}, e.config), { tools: n }),
        };
      if (((r.config.tools = n), e.config && e.config.tools && Os(e.config.tools))) {
        let t = e.config.httpOptions?.headers ?? {},
          n = Object.assign({}, t);
        (Object.keys(n).length === 0 && (n = this.apiClient.getDefaultHeaders()),
          ks(n),
          (r.config.httpOptions = Object.assign(Object.assign({}, e.config.httpOptions), {
            headers: n,
          })));
      }
      return r;
    }
    async initAfcToolsMap(e) {
      let t = new Map();
      for (let n of e.config?.tools ?? [])
        if (qs(n)) {
          let e = n,
            r = await e.tool();
          for (let n of r.functionDeclarations ?? []) {
            if (!n.name) throw Error(`Function declaration name is required.`);
            if (t.has(n.name)) throw Error(`Duplicate tool declaration name: ${n.name}`);
            t.set(n.name, e);
          }
        }
      return t;
    }
    async processAfcStream(e) {
      let t = e.config?.automaticFunctionCalling?.maximumRemoteCalls ?? Gs,
        n = !1,
        r = 0,
        i = await this.initAfcToolsMap(e);
      return (function (e, i, a) {
        return G(this, arguments, function* () {
          for (var o, s, c, l; r < t; ) {
            n &&= (r++, !1);
            let p = yield W(e.processParamsMaybeAddMcpUsage(a)),
              m = yield W(e.generateContentStreamInternal(p)),
              h = [],
              g = [];
            try {
              for (
                var u = !0, d = ((s = void 0), K(m)), f;
                (f = yield W(d.next())), (o = f.done), !o;
                u = !0
              ) {
                ((l = f.value), (u = !1));
                let e = l;
                if ((yield yield W(e), e.candidates && e.candidates[0]?.content)) {
                  g.push(e.candidates[0].content);
                  for (let n of e.candidates[0].content.parts ?? [])
                    if (r < t && n.functionCall) {
                      if (!n.functionCall.name)
                        throw Error(`Function call name was not returned by the model.`);
                      if (i.has(n.functionCall.name)) {
                        let e = yield W(i.get(n.functionCall.name).callTool([n.functionCall]));
                        h.push(...e);
                      } else
                        throw Error(
                          `Automatic function calling was requested, but not all the tools the model used implement the CallableTool interface. Available tools: ${i.keys()}, mising tool: ${n.functionCall.name}`,
                        );
                    }
                }
              }
            } catch (e) {
              s = { error: e };
            } finally {
              try {
                !u && !o && (c = d.return) && (yield W(c.call(d)));
              } finally {
                if (s) throw s.error;
              }
            }
            if (h.length > 0) {
              n = !0;
              let e = new ut();
              ((e.candidates = [{ content: { role: `user`, parts: h } }]), yield yield W(e));
              let t = [];
              (t.push(...g),
                t.push({ role: `user`, parts: h }),
                (a.contents = I(a.contents).concat(t)));
            } else break;
          }
        });
      })(this, i, e);
    }
    async generateContentInternal(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI()) {
        let i = Da(this.apiClient, e);
        return (
          (n = p(`{model}:generateContent`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `POST`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) =>
              e.json().then((t) => {
                let n = t;
                return ((n.sdkHttpResponse = { headers: e.headers }), n);
              }),
            )),
          t.then((e) => {
            let t = ka(e),
              n = new ut();
            return (Object.assign(n, t), n);
          })
        );
      } else {
        let i = Ea(this.apiClient, e);
        return (
          (n = p(`{model}:generateContent`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `POST`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) =>
              e.json().then((t) => {
                let n = t;
                return ((n.sdkHttpResponse = { headers: e.headers }), n);
              }),
            )),
          t.then((e) => {
            let t = Oa(e),
              n = new ut();
            return (Object.assign(n, t), n);
          })
        );
      }
    }
    async generateContentStreamInternal(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI()) {
        let i = Da(this.apiClient, e);
        return (
          (n = p(`{model}:streamGenerateContent?alt=sse`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient.requestStream({
            path: n,
            queryParams: r,
            body: JSON.stringify(i),
            httpMethod: `POST`,
            httpOptions: e.config?.httpOptions,
            abortSignal: e.config?.abortSignal,
          })),
          t.then(function (t) {
            return G(this, arguments, function* () {
              var n, r, i, a;
              try {
                for (var o = !0, s = K(t), c; (c = yield W(s.next())), (n = c.done), !n; o = !0) {
                  ((a = c.value), (o = !1));
                  let t = a,
                    n = ka(yield W(t.json()), e);
                  n.sdkHttpResponse = { headers: t.headers };
                  let r = new ut();
                  (Object.assign(r, n), yield yield W(r));
                }
              } catch (e) {
                r = { error: e };
              } finally {
                try {
                  !o && !n && (i = s.return) && (yield W(i.call(s)));
                } finally {
                  if (r) throw r.error;
                }
              }
            });
          })
        );
      } else {
        let i = Ea(this.apiClient, e);
        return (
          (n = p(`{model}:streamGenerateContent?alt=sse`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient.requestStream({
            path: n,
            queryParams: r,
            body: JSON.stringify(i),
            httpMethod: `POST`,
            httpOptions: e.config?.httpOptions,
            abortSignal: e.config?.abortSignal,
          })),
          t.then(function (t) {
            return G(this, arguments, function* () {
              var n, r, i, a;
              try {
                for (var o = !0, s = K(t), c; (c = yield W(s.next())), (n = c.done), !n; o = !0) {
                  ((a = c.value), (o = !1));
                  let t = a,
                    n = Oa(yield W(t.json()), e);
                  n.sdkHttpResponse = { headers: t.headers };
                  let r = new ut();
                  (Object.assign(r, n), yield yield W(r));
                }
              } catch (e) {
                r = { error: e };
              } finally {
                try {
                  !o && !n && (i = s.return) && (yield W(i.call(s)));
                } finally {
                  if (r) throw r.error;
                }
              }
            });
          })
        );
      }
    }
    async embedContentInternal(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI()) {
        let i = ga(this.apiClient, e, e);
        return (
          (n = p(gn(e.model) ? `{model}:embedContent` : `{model}:predict`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `POST`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) =>
              e.json().then((t) => {
                let n = t;
                return ((n.sdkHttpResponse = { headers: e.headers }), n);
              }),
            )),
          t.then((t) => {
            let n = va(t, e),
              r = new dt();
            return (Object.assign(r, n), r);
          })
        );
      } else {
        let i = ha(this.apiClient, e);
        return (
          (n = p(`{model}:batchEmbedContents`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `POST`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) =>
              e.json().then((t) => {
                let n = t;
                return ((n.sdkHttpResponse = { headers: e.headers }), n);
              }),
            )),
          t.then((e) => {
            let t = _a(e),
              n = new dt();
            return (Object.assign(n, t), n);
          })
        );
      }
    }
    async generateImagesInternal(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI()) {
        let i = Na(this.apiClient, e);
        return (
          (n = p(`{model}:predict`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `POST`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) =>
              e.json().then((t) => {
                let n = t;
                return ((n.sdkHttpResponse = { headers: e.headers }), n);
              }),
            )),
          t.then((e) => {
            let t = Fa(e),
              n = new ft();
            return (Object.assign(n, t), n);
          })
        );
      } else {
        let i = Ma(this.apiClient, e);
        return (
          (n = p(`{model}:predict`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `POST`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) =>
              e.json().then((t) => {
                let n = t;
                return ((n.sdkHttpResponse = { headers: e.headers }), n);
              }),
            )),
          t.then((e) => {
            let t = Pa(e),
              n = new ft();
            return (Object.assign(n, t), n);
          })
        );
      }
    }
    async editImageInternal(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI()) {
        let i = da(this.apiClient, e);
        return (
          (n = p(`{model}:predict`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `POST`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) =>
              e.json().then((t) => {
                let n = t;
                return ((n.sdkHttpResponse = { headers: e.headers }), n);
              }),
            )),
          t.then((e) => {
            let t = fa(e),
              n = new pt();
            return (Object.assign(n, t), n);
          })
        );
      } else throw Error(`This method is only supported by the Vertex AI.`);
    }
    async upscaleImageInternal(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI()) {
        let i = Wo(this.apiClient, e);
        return (
          (n = p(`{model}:predict`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `POST`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) =>
              e.json().then((t) => {
                let n = t;
                return ((n.sdkHttpResponse = { headers: e.headers }), n);
              }),
            )),
          t.then((e) => {
            let t = Go(e),
              n = new mt();
            return (Object.assign(n, t), n);
          })
        );
      } else throw Error(`This method is only supported by the Vertex AI.`);
    }
    async recontextImage(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI()) {
        let i = xo(this.apiClient, e);
        return (
          (n = p(`{model}:predict`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `POST`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) => e.json())),
          t.then((e) => {
            let t = So(e),
              n = new ht();
            return (Object.assign(n, t), n);
          })
        );
      } else throw Error(`This method is only supported by the Vertex AI.`);
    }
    async segmentImage(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI()) {
        let i = Ao(this.apiClient, e);
        return (
          (n = p(`{model}:predict`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `POST`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) => e.json())),
          t.then((e) => {
            let t = jo(e),
              n = new gt();
            return (Object.assign(n, t), n);
          })
        );
      } else throw Error(`This method is only supported by the Vertex AI.`);
    }
    async get(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI()) {
        let i = $a(this.apiClient, e);
        return (
          (n = p(`{name}`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `GET`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) => e.json())),
          t.then((e) => go(e))
        );
      } else {
        let i = Qa(this.apiClient, e);
        return (
          (n = p(`{name}`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `GET`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) => e.json())),
          t.then((e) => ho(e))
        );
      }
    }
    async listInternal(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI()) {
        let i = uo(this.apiClient, e);
        return (
          (n = p(`{models_url}`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `GET`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) =>
              e.json().then((t) => {
                let n = t;
                return ((n.sdkHttpResponse = { headers: e.headers }), n);
              }),
            )),
          t.then((e) => {
            let t = po(e),
              n = new _t();
            return (Object.assign(n, t), n);
          })
        );
      } else {
        let i = lo(this.apiClient, e);
        return (
          (n = p(`{models_url}`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `GET`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) =>
              e.json().then((t) => {
                let n = t;
                return ((n.sdkHttpResponse = { headers: e.headers }), n);
              }),
            )),
          t.then((e) => {
            let t = fo(e),
              n = new _t();
            return (Object.assign(n, t), n);
          })
        );
      }
    }
    async update(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI()) {
        let i = Ho(this.apiClient, e);
        return (
          (n = p(`{model}`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `PATCH`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) => e.json())),
          t.then((e) => go(e))
        );
      } else {
        let i = Vo(this.apiClient, e);
        return (
          (n = p(`{name}`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `PATCH`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) => e.json())),
          t.then((e) => ho(e))
        );
      }
    }
    async delete(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI()) {
        let i = sa(this.apiClient, e);
        return (
          (n = p(`{name}`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `DELETE`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) =>
              e.json().then((t) => {
                let n = t;
                return ((n.sdkHttpResponse = { headers: e.headers }), n);
              }),
            )),
          t.then((e) => {
            let t = la(e),
              n = new vt();
            return (Object.assign(n, t), n);
          })
        );
      } else {
        let i = oa(this.apiClient, e);
        return (
          (n = p(`{name}`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `DELETE`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) =>
              e.json().then((t) => {
                let n = t;
                return ((n.sdkHttpResponse = { headers: e.headers }), n);
              }),
            )),
          t.then((e) => {
            let t = ca(e),
              n = new vt();
            return (Object.assign(n, t), n);
          })
        );
      }
    }
    async countTokens(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI()) {
        let i = ra(this.apiClient, e);
        return (
          (n = p(`{model}:countTokens`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `POST`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) =>
              e.json().then((t) => {
                let n = t;
                return ((n.sdkHttpResponse = { headers: e.headers }), n);
              }),
            )),
          t.then((e) => {
            let t = aa(e),
              n = new yt();
            return (Object.assign(n, t), n);
          })
        );
      } else {
        let i = na(this.apiClient, e);
        return (
          (n = p(`{model}:countTokens`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `POST`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) =>
              e.json().then((t) => {
                let n = t;
                return ((n.sdkHttpResponse = { headers: e.headers }), n);
              }),
            )),
          t.then((e) => {
            let t = ia(e),
              n = new yt();
            return (Object.assign(n, t), n);
          })
        );
      }
    }
    async computeTokens(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI()) {
        let i = Ji(this.apiClient, e);
        return (
          (n = p(`{model}:computeTokens`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `POST`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) =>
              e.json().then((t) => {
                let n = t;
                return ((n.sdkHttpResponse = { headers: e.headers }), n);
              }),
            )),
          t.then((e) => {
            let t = Yi(e),
              n = new bt();
            return (Object.assign(n, t), n);
          })
        );
      } else throw Error(`This method is only supported by the Vertex AI.`);
    }
    async generateVideosInternal(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI()) {
        let i = Va(this.apiClient, e);
        return (
          (n = p(`{model}:predictLongRunning`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `POST`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) => e.json())),
          t.then((e) => {
            let t = za(e),
              n = new xt();
            return (Object.assign(n, t), n);
          })
        );
      } else {
        let i = Ba(this.apiClient, e);
        return (
          (n = p(`{model}:predictLongRunning`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `POST`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) => e.json())),
          t.then((e) => {
            let t = Ra(e),
              n = new xt();
            return (Object.assign(n, t), n);
          })
        );
      }
    }
  },
  Qs = class extends f {
    constructor(e) {
      (super(), (this.apiClient = e));
    }
    async getVideosOperation(e) {
      let t = e.operation,
        n = e.config;
      if (t.name === void 0 || t.name === ``) throw Error(`Operation name is required.`);
      if (this.apiClient.isVertexAI()) {
        let e = t.name.split(`/operations/`)[0],
          r;
        n && `httpOptions` in n && (r = n.httpOptions);
        let i = await this.fetchPredictVideosOperationInternal({
          operationName: t.name,
          resourceName: e,
          config: { httpOptions: r },
        });
        return t._fromAPIResponse({ apiResponse: i, _isVertexAI: !0 });
      } else {
        let e = await this.getVideosOperationInternal({ operationName: t.name, config: n });
        return t._fromAPIResponse({ apiResponse: e, _isVertexAI: !1 });
      }
    }
    async get(e) {
      let t = e.operation,
        n = e.config;
      if (t.name === void 0 || t.name === ``) throw Error(`Operation name is required.`);
      if (this.apiClient.isVertexAI()) {
        let e = t.name.split(`/operations/`)[0],
          r;
        n && `httpOptions` in n && (r = n.httpOptions);
        let i = await this.fetchPredictVideosOperationInternal({
          operationName: t.name,
          resourceName: e,
          config: { httpOptions: r },
        });
        return t._fromAPIResponse({ apiResponse: i, _isVertexAI: !0 });
      } else {
        let e = await this.getVideosOperationInternal({ operationName: t.name, config: n });
        return t._fromAPIResponse({ apiResponse: e, _isVertexAI: !1 });
      }
    }
    async getVideosOperationInternal(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI()) {
        let i = D(e);
        return (
          (n = p(`{operationName}`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `GET`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) => e.json())),
          t
        );
      } else {
        let i = E(e);
        return (
          (n = p(`{operationName}`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `GET`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) => e.json())),
          t
        );
      }
    }
    async fetchPredictVideosOperationInternal(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI()) {
        let i = y(e);
        return (
          (n = p(`{resourceName}:fetchPredictOperation`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `POST`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) => e.json())),
          t
        );
      } else throw Error(`This method is only supported by the Vertex AI.`);
    }
  };
function $s(e) {
  let t = {};
  if (h(e, [`languageCodes`]) !== void 0)
    throw Error(`languageCodes parameter is not supported in Gemini API.`);
  return t;
}
function ec(e) {
  let t = {},
    n = h(e, [`apiKey`]);
  if ((n != null && m(t, [`apiKey`], n), h(e, [`apiKeyConfig`]) !== void 0))
    throw Error(`apiKeyConfig parameter is not supported in Gemini API.`);
  if (h(e, [`authType`]) !== void 0)
    throw Error(`authType parameter is not supported in Gemini API.`);
  if (h(e, [`googleServiceAccountConfig`]) !== void 0)
    throw Error(`googleServiceAccountConfig parameter is not supported in Gemini API.`);
  if (h(e, [`httpBasicAuthConfig`]) !== void 0)
    throw Error(`httpBasicAuthConfig parameter is not supported in Gemini API.`);
  if (h(e, [`oauthConfig`]) !== void 0)
    throw Error(`oauthConfig parameter is not supported in Gemini API.`);
  if (h(e, [`oidcConfig`]) !== void 0)
    throw Error(`oidcConfig parameter is not supported in Gemini API.`);
  return t;
}
function tc(e) {
  let t = {},
    n = h(e, [`data`]);
  if ((n != null && m(t, [`data`], n), h(e, [`displayName`]) !== void 0))
    throw Error(`displayName parameter is not supported in Gemini API.`);
  let r = h(e, [`mimeType`]);
  return (r != null && m(t, [`mimeType`], r), t);
}
function nc(e) {
  let t = {},
    n = h(e, [`parts`]);
  if (n != null) {
    let e = n;
    (Array.isArray(e) && (e = e.map((e) => dc(e))), m(t, [`parts`], e));
  }
  let r = h(e, [`role`]);
  return (r != null && m(t, [`role`], r), t);
}
function rc(e, t, n) {
  let r = {},
    i = h(t, [`expireTime`]);
  n !== void 0 && i != null && m(n, [`expireTime`], i);
  let a = h(t, [`newSessionExpireTime`]);
  n !== void 0 && a != null && m(n, [`newSessionExpireTime`], a);
  let o = h(t, [`uses`]);
  n !== void 0 && o != null && m(n, [`uses`], o);
  let s = h(t, [`liveConnectConstraints`]);
  n !== void 0 && s != null && m(n, [`bidiGenerateContentSetup`], uc(e, s));
  let c = h(t, [`lockAdditionalFields`]);
  return (n !== void 0 && c != null && m(n, [`fieldMask`], c), r);
}
function ic(e, t) {
  let n = {},
    r = h(t, [`config`]);
  return (r != null && m(n, [`config`], rc(e, r, n)), n);
}
function ac(e) {
  let t = {};
  if (h(e, [`displayName`]) !== void 0)
    throw Error(`displayName parameter is not supported in Gemini API.`);
  let n = h(e, [`fileUri`]);
  n != null && m(t, [`fileUri`], n);
  let r = h(e, [`mimeType`]);
  return (r != null && m(t, [`mimeType`], r), t);
}
function oc(e) {
  let t = {},
    n = h(e, [`id`]);
  n != null && m(t, [`id`], n);
  let r = h(e, [`args`]);
  r != null && m(t, [`args`], r);
  let i = h(e, [`name`]);
  if ((i != null && m(t, [`name`], i), h(e, [`partialArgs`]) !== void 0))
    throw Error(`partialArgs parameter is not supported in Gemini API.`);
  if (h(e, [`willContinue`]) !== void 0)
    throw Error(`willContinue parameter is not supported in Gemini API.`);
  return t;
}
function sc(e) {
  let t = {},
    n = h(e, [`authConfig`]);
  n != null && m(t, [`authConfig`], ec(n));
  let r = h(e, [`enableWidget`]);
  return (r != null && m(t, [`enableWidget`], r), t);
}
function cc(e) {
  let t = {},
    n = h(e, [`searchTypes`]);
  if ((n != null && m(t, [`searchTypes`], n), h(e, [`blockingConfidence`]) !== void 0))
    throw Error(`blockingConfidence parameter is not supported in Gemini API.`);
  if (h(e, [`excludeDomains`]) !== void 0)
    throw Error(`excludeDomains parameter is not supported in Gemini API.`);
  let r = h(e, [`timeRangeFilter`]);
  return (r != null && m(t, [`timeRangeFilter`], r), t);
}
function lc(e, t) {
  let n = {},
    r = h(e, [`generationConfig`]);
  t !== void 0 && r != null && m(t, [`setup`, `generationConfig`], r);
  let i = h(e, [`responseModalities`]);
  t !== void 0 && i != null && m(t, [`setup`, `generationConfig`, `responseModalities`], i);
  let a = h(e, [`temperature`]);
  t !== void 0 && a != null && m(t, [`setup`, `generationConfig`, `temperature`], a);
  let o = h(e, [`topP`]);
  t !== void 0 && o != null && m(t, [`setup`, `generationConfig`, `topP`], o);
  let s = h(e, [`topK`]);
  t !== void 0 && s != null && m(t, [`setup`, `generationConfig`, `topK`], s);
  let c = h(e, [`maxOutputTokens`]);
  t !== void 0 && c != null && m(t, [`setup`, `generationConfig`, `maxOutputTokens`], c);
  let l = h(e, [`mediaResolution`]);
  t !== void 0 && l != null && m(t, [`setup`, `generationConfig`, `mediaResolution`], l);
  let u = h(e, [`seed`]);
  t !== void 0 && u != null && m(t, [`setup`, `generationConfig`, `seed`], u);
  let d = h(e, [`speechConfig`]);
  t !== void 0 && d != null && m(t, [`setup`, `generationConfig`, `speechConfig`], Qt(d));
  let f = h(e, [`thinkingConfig`]);
  t !== void 0 && f != null && m(t, [`setup`, `generationConfig`, `thinkingConfig`], f);
  let p = h(e, [`enableAffectiveDialog`]);
  t !== void 0 && p != null && m(t, [`setup`, `generationConfig`, `enableAffectiveDialog`], p);
  let g = h(e, [`systemInstruction`]);
  t !== void 0 && g != null && m(t, [`setup`, `systemInstruction`], nc(F(g)));
  let _ = h(e, [`tools`]);
  if (t !== void 0 && _ != null) {
    let e = z(_);
    (Array.isArray(e) && (e = e.map((e) => pc(R(e)))), m(t, [`setup`, `tools`], e));
  }
  let v = h(e, [`sessionResumption`]);
  t !== void 0 && v != null && m(t, [`setup`, `sessionResumption`], fc(v));
  let y = h(e, [`inputAudioTranscription`]);
  t !== void 0 && y != null && m(t, [`setup`, `inputAudioTranscription`], $s(y));
  let b = h(e, [`outputAudioTranscription`]);
  t !== void 0 && b != null && m(t, [`setup`, `outputAudioTranscription`], $s(b));
  let x = h(e, [`realtimeInputConfig`]);
  t !== void 0 && x != null && m(t, [`setup`, `realtimeInputConfig`], x);
  let S = h(e, [`contextWindowCompression`]);
  t !== void 0 && S != null && m(t, [`setup`, `contextWindowCompression`], S);
  let C = h(e, [`proactivity`]);
  if (
    (t !== void 0 && C != null && m(t, [`setup`, `proactivity`], C),
    h(e, [`explicitVadSignal`]) !== void 0)
  )
    throw Error(`explicitVadSignal parameter is not supported in Gemini API.`);
  return n;
}
function uc(e, t) {
  let n = {},
    r = h(t, [`model`]);
  r != null && m(n, [`setup`, `model`], P(e, r));
  let i = h(t, [`config`]);
  return (i != null && m(n, [`config`], lc(i, n)), n);
}
function dc(e) {
  let t = {},
    n = h(e, [`mediaResolution`]);
  n != null && m(t, [`mediaResolution`], n);
  let r = h(e, [`codeExecutionResult`]);
  r != null && m(t, [`codeExecutionResult`], r);
  let i = h(e, [`executableCode`]);
  i != null && m(t, [`executableCode`], i);
  let a = h(e, [`fileData`]);
  a != null && m(t, [`fileData`], ac(a));
  let o = h(e, [`functionCall`]);
  o != null && m(t, [`functionCall`], oc(o));
  let s = h(e, [`functionResponse`]);
  s != null && m(t, [`functionResponse`], s);
  let c = h(e, [`inlineData`]);
  c != null && m(t, [`inlineData`], tc(c));
  let l = h(e, [`text`]);
  l != null && m(t, [`text`], l);
  let u = h(e, [`thought`]);
  u != null && m(t, [`thought`], u);
  let d = h(e, [`thoughtSignature`]);
  d != null && m(t, [`thoughtSignature`], d);
  let f = h(e, [`videoMetadata`]);
  f != null && m(t, [`videoMetadata`], f);
  let p = h(e, [`toolCall`]);
  p != null && m(t, [`toolCall`], p);
  let g = h(e, [`toolResponse`]);
  return (g != null && m(t, [`toolResponse`], g), t);
}
function fc(e) {
  let t = {},
    n = h(e, [`handle`]);
  if ((n != null && m(t, [`handle`], n), h(e, [`transparent`]) !== void 0))
    throw Error(`transparent parameter is not supported in Gemini API.`);
  return t;
}
function pc(e) {
  let t = {};
  if (h(e, [`retrieval`]) !== void 0)
    throw Error(`retrieval parameter is not supported in Gemini API.`);
  let n = h(e, [`computerUse`]);
  n != null && m(t, [`computerUse`], n);
  let r = h(e, [`fileSearch`]);
  r != null && m(t, [`fileSearch`], r);
  let i = h(e, [`googleSearch`]);
  i != null && m(t, [`googleSearch`], cc(i));
  let a = h(e, [`googleMaps`]);
  a != null && m(t, [`googleMaps`], sc(a));
  let o = h(e, [`codeExecution`]);
  if ((o != null && m(t, [`codeExecution`], o), h(e, [`enterpriseWebSearch`]) !== void 0))
    throw Error(`enterpriseWebSearch parameter is not supported in Gemini API.`);
  let s = h(e, [`functionDeclarations`]);
  if (s != null) {
    let e = s;
    (Array.isArray(e) && (e = e.map((e) => e)), m(t, [`functionDeclarations`], e));
  }
  let c = h(e, [`googleSearchRetrieval`]);
  if ((c != null && m(t, [`googleSearchRetrieval`], c), h(e, [`parallelAiSearch`]) !== void 0))
    throw Error(`parallelAiSearch parameter is not supported in Gemini API.`);
  let l = h(e, [`urlContext`]);
  l != null && m(t, [`urlContext`], l);
  let u = h(e, [`mcpServers`]);
  if (u != null) {
    let e = u;
    (Array.isArray(e) && (e = e.map((e) => e)), m(t, [`mcpServers`], e));
  }
  return t;
}
function mc(e) {
  let t = [];
  for (let n in e)
    if (Object.prototype.hasOwnProperty.call(e, n)) {
      let r = e[n];
      if (typeof r == `object` && r && Object.keys(r).length > 0) {
        let e = Object.keys(r).map((e) => `${n}.${e}`);
        t.push(...e);
      } else t.push(n);
    }
  return t.join(`,`);
}
function hc(e, t) {
  let n = null,
    r = e.bidiGenerateContentSetup;
  if (typeof r == `object` && r && `setup` in r) {
    let t = r.setup;
    typeof t == `object` && t
      ? ((e.bidiGenerateContentSetup = t), (n = t))
      : delete e.bidiGenerateContentSetup;
  } else r !== void 0 && delete e.bidiGenerateContentSetup;
  let i = e.fieldMask;
  if (n) {
    let r = mc(n);
    if (Array.isArray(t?.lockAdditionalFields) && t?.lockAdditionalFields.length === 0)
      r ? (e.fieldMask = r) : delete e.fieldMask;
    else if (
      t?.lockAdditionalFields &&
      t.lockAdditionalFields.length > 0 &&
      i !== null &&
      Array.isArray(i) &&
      i.length > 0
    ) {
      let t = [
          `temperature`,
          `topK`,
          `topP`,
          `maxOutputTokens`,
          `responseModalities`,
          `seed`,
          `speechConfig`,
        ],
        n = [];
      i.length > 0 && (n = i.map((e) => (t.includes(e) ? `generationConfig.${e}` : e)));
      let a = [];
      (r && a.push(r),
        n.length > 0 && a.push(...n),
        a.length > 0 ? (e.fieldMask = a.join(`,`)) : delete e.fieldMask);
    } else delete e.fieldMask;
  } else
    i !== null && Array.isArray(i) && i.length > 0
      ? (e.fieldMask = i.join(`,`))
      : delete e.fieldMask;
  return e;
}
var gc = class extends f {
  constructor(e) {
    (super(), (this.apiClient = e));
  }
  async create(e) {
    let t,
      n = ``,
      r = {};
    if (this.apiClient.isVertexAI())
      throw Error(`The client.tokens.create method is only supported by the Gemini Developer API.`);
    {
      let i = ic(this.apiClient, e);
      ((n = p(`auth_tokens`, i._url)),
        (r = i._query),
        delete i.config,
        delete i._url,
        delete i._query);
      let a = hc(i, e.config);
      return (
        (t = this.apiClient
          .request({
            path: n,
            queryParams: r,
            body: JSON.stringify(a),
            httpMethod: `POST`,
            httpOptions: e.config?.httpOptions,
            abortSignal: e.config?.abortSignal,
          })
          .then((e) => e.json())),
        t.then((e) => e)
      );
    }
  }
};
function _c(e, t) {
  let n = {},
    r = h(e, [`force`]);
  return (t !== void 0 && r != null && m(t, [`_query`, `force`], r), n);
}
function vc(e) {
  let t = {},
    n = h(e, [`name`]);
  n != null && m(t, [`_url`, `name`], n);
  let r = h(e, [`config`]);
  return (r != null && _c(r, t), t);
}
function yc(e) {
  let t = {},
    n = h(e, [`name`]);
  return (n != null && m(t, [`_url`, `name`], n), t);
}
function bc(e, t) {
  let n = {},
    r = h(e, [`pageSize`]);
  t !== void 0 && r != null && m(t, [`_query`, `pageSize`], r);
  let i = h(e, [`pageToken`]);
  return (t !== void 0 && i != null && m(t, [`_query`, `pageToken`], i), n);
}
function xc(e) {
  let t = {},
    n = h(e, [`parent`]);
  n != null && m(t, [`_url`, `parent`], n);
  let r = h(e, [`config`]);
  return (r != null && bc(r, t), t);
}
function Sc(e) {
  let t = {},
    n = h(e, [`sdkHttpResponse`]);
  n != null && m(t, [`sdkHttpResponse`], n);
  let r = h(e, [`nextPageToken`]);
  r != null && m(t, [`nextPageToken`], r);
  let i = h(e, [`documents`]);
  if (i != null) {
    let e = i;
    (Array.isArray(e) && (e = e.map((e) => e)), m(t, [`documents`], e));
  }
  return t;
}
var Cc = class extends f {
    constructor(e) {
      (super(),
        (this.apiClient = e),
        (this.list = async (e) =>
          new U(
            H.PAGED_ITEM_DOCUMENTS,
            (t) => this.listInternal({ parent: e.parent, config: t.config }),
            await this.listInternal(e),
            e,
          )));
    }
    async get(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI())
        throw Error(`This method is only supported by the Gemini Developer API.`);
      {
        let i = yc(e);
        return (
          (n = p(`{name}`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `GET`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) => e.json())),
          t.then((e) => e)
        );
      }
    }
    async delete(e) {
      let t = ``,
        n = {};
      if (this.apiClient.isVertexAI())
        throw Error(`This method is only supported by the Gemini Developer API.`);
      {
        let r = vc(e);
        ((t = p(`{name}`, r._url)),
          (n = r._query),
          delete r._url,
          delete r._query,
          await this.apiClient.request({
            path: t,
            queryParams: n,
            body: JSON.stringify(r),
            httpMethod: `DELETE`,
            httpOptions: e.config?.httpOptions,
            abortSignal: e.config?.abortSignal,
          }));
      }
    }
    async listInternal(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI())
        throw Error(`This method is only supported by the Gemini Developer API.`);
      {
        let i = xc(e);
        return (
          (n = p(`{parent}/documents`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `GET`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) => e.json())),
          t.then((e) => {
            let t = Sc(e),
              n = new Et();
            return (Object.assign(n, t), n);
          })
        );
      }
    }
  },
  wc = class extends f {
    constructor(e, t = new Cc(e)) {
      (super(),
        (this.apiClient = e),
        (this.documents = t),
        (this.list = async (e = {}) =>
          new U(
            H.PAGED_ITEM_FILE_SEARCH_STORES,
            (e) => this.listInternal(e),
            await this.listInternal(e),
            e,
          )));
    }
    async uploadToFileSearchStore(e) {
      if (this.apiClient.isVertexAI())
        throw Error(`Vertex AI does not support uploading files to a file search store.`);
      return this.apiClient.uploadFileToFileSearchStore(e.fileSearchStoreName, e.file, e.config);
    }
    async create(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI())
        throw Error(`This method is only supported by the Gemini Developer API.`);
      {
        let i = es(e);
        return (
          (n = p(`fileSearchStores`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `POST`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) => e.json())),
          t.then((e) => e)
        );
      }
    }
    async get(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI())
        throw Error(`This method is only supported by the Gemini Developer API.`);
      {
        let i = rs(e);
        return (
          (n = p(`{name}`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `GET`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) => e.json())),
          t.then((e) => e)
        );
      }
    }
    async delete(e) {
      let t = ``,
        n = {};
      if (this.apiClient.isVertexAI())
        throw Error(`This method is only supported by the Gemini Developer API.`);
      {
        let r = ns(e);
        ((t = p(`{name}`, r._url)),
          (n = r._query),
          delete r._url,
          delete r._query,
          await this.apiClient.request({
            path: t,
            queryParams: n,
            body: JSON.stringify(r),
            httpMethod: `DELETE`,
            httpOptions: e.config?.httpOptions,
            abortSignal: e.config?.abortSignal,
          }));
      }
    }
    async listInternal(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI())
        throw Error(`This method is only supported by the Gemini Developer API.`);
      {
        let i = ls(e);
        return (
          (n = p(`fileSearchStores`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `GET`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) => e.json())),
          t.then((e) => {
            let t = us(e),
              n = new Dt();
            return (Object.assign(n, t), n);
          })
        );
      }
    }
    async uploadToFileSearchStoreInternal(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI())
        throw Error(`This method is only supported by the Gemini Developer API.`);
      {
        let i = fs(e);
        return (
          (n = p(`upload/v1beta/{file_search_store_name}:uploadToFileSearchStore`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `POST`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) => e.json())),
          t.then((e) => {
            let t = ps(e),
              n = new Ot();
            return (Object.assign(n, t), n);
          })
        );
      }
    }
    async importFile(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI())
        throw Error(`This method is only supported by the Gemini Developer API.`);
      {
        let i = os(e);
        return (
          (n = p(`{file_search_store_name}:importFile`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `POST`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) => e.json())),
          t.then((e) => {
            let t = as(e),
              n = new kt();
            return (Object.assign(n, t), n);
          })
        );
      }
    }
  },
  Tc = function () {
    let { crypto: e } = globalThis;
    if (e?.randomUUID) return ((Tc = e.randomUUID.bind(e)), e.randomUUID());
    let t = new Uint8Array(1),
      n = e ? () => e.getRandomValues(t)[0] : () => (Math.random() * 255) & 255;
    return `10000000-1000-4000-8000-100000000000`.replace(/[018]/g, (e) =>
      (e ^ (n() & (15 >> (e / 4)))).toString(16),
    );
  },
  Ec = () => Tc();
function Dc(e) {
  return (
    typeof e == `object` &&
    !!e &&
    ((`name` in e && e.name === `AbortError`) ||
      (`message` in e && String(e.message).includes(`FetchRequestCanceledException`)))
  );
}
var Oc = (e) => {
    if (e instanceof Error) return e;
    if (typeof e == `object` && e) {
      try {
        if (Object.prototype.toString.call(e) === `[object Error]`) {
          let t = Error(e.message, e.cause ? { cause: e.cause } : {});
          return (
            e.stack && (t.stack = e.stack),
            e.cause && !t.cause && (t.cause = e.cause),
            e.name && (t.name = e.name),
            t
          );
        }
      } catch {}
      try {
        return Error(JSON.stringify(e));
      } catch {}
    }
    return Error(e);
  },
  Y = class extends Error {},
  X = class e extends Y {
    constructor(t, n, r, i) {
      (super(`${e.makeMessage(t, n, r)}`), (this.status = t), (this.headers = i), (this.error = n));
    }
    static makeMessage(e, t, n) {
      let r = t?.message
        ? typeof t.message == `string`
          ? t.message
          : JSON.stringify(t.message)
        : t
          ? JSON.stringify(t)
          : n;
      return e && r
        ? `${e} ${r}`
        : e
          ? `${e} status code (no body)`
          : r || `(no status code or body)`;
    }
    static generate(t, n, r, i) {
      if (!t || !i) return new Ac({ message: r, cause: Oc(n) });
      let a = n;
      return t === 400
        ? new Mc(t, a, r, i)
        : t === 401
          ? new Nc(t, a, r, i)
          : t === 403
            ? new Pc(t, a, r, i)
            : t === 404
              ? new Fc(t, a, r, i)
              : t === 409
                ? new Ic(t, a, r, i)
                : t === 422
                  ? new Lc(t, a, r, i)
                  : t === 429
                    ? new Rc(t, a, r, i)
                    : t >= 500
                      ? new zc(t, a, r, i)
                      : new e(t, a, r, i);
    }
  },
  kc = class extends X {
    constructor({ message: e } = {}) {
      super(void 0, void 0, e || `Request was aborted.`, void 0);
    }
  },
  Ac = class extends X {
    constructor({ message: e, cause: t }) {
      (super(void 0, void 0, e || `Connection error.`, void 0), t && (this.cause = t));
    }
  },
  jc = class extends Ac {
    constructor({ message: e } = {}) {
      super({ message: e ?? `Request timed out.` });
    }
  },
  Mc = class extends X {},
  Nc = class extends X {},
  Pc = class extends X {},
  Fc = class extends X {},
  Ic = class extends X {},
  Lc = class extends X {},
  Rc = class extends X {},
  zc = class extends X {},
  Bc = /^[a-z][a-z0-9+.-]*:/i,
  Vc = (e) => Bc.test(e),
  Hc = (e) => ((Hc = Array.isArray), Hc(e)),
  Uc = Hc;
function Wc(e) {
  if (!e) return !0;
  for (let t in e) return !1;
  return !0;
}
function Gc(e, t) {
  return Object.prototype.hasOwnProperty.call(e, t);
}
var Kc = (e, t) => {
    if (typeof t != `number` || !Number.isInteger(t)) throw new Y(`${e} must be an integer`);
    if (t < 0) throw new Y(`${e} must be a positive integer`);
    return t;
  },
  qc = (e) => {
    try {
      return JSON.parse(e);
    } catch {
      return;
    }
  },
  Jc = (e) => new Promise((t) => setTimeout(t, e));
function Yc() {
  if (typeof fetch < `u`) return fetch;
  throw Error(
    "`fetch` is not defined as a global; Either pass `fetch` to the client, `new GeminiNextGenAPIClient({ fetch })` or polyfill the global, `globalThis.fetch = fetch`",
  );
}
function Xc(...e) {
  let t = globalThis.ReadableStream;
  if (t === void 0)
    throw Error(
      "`ReadableStream` is not defined as a global; You will need to polyfill it, `globalThis.ReadableStream = ReadableStream`",
    );
  return new t(...e);
}
function Zc(e) {
  let t = Symbol.asyncIterator in e ? e[Symbol.asyncIterator]() : e[Symbol.iterator]();
  return Xc({
    start() {},
    async pull(e) {
      let { done: n, value: r } = await t.next();
      n ? e.close() : e.enqueue(r);
    },
    async cancel() {
      await t.return?.call(t);
    },
  });
}
function Qc(e) {
  if (e[Symbol.asyncIterator]) return e;
  let t = e.getReader();
  return {
    async next() {
      try {
        let e = await t.read();
        return (e?.done && t.releaseLock(), e);
      } catch (e) {
        throw (t.releaseLock(), e);
      }
    },
    async return() {
      let e = t.cancel();
      return (t.releaseLock(), await e, { done: !0, value: void 0 });
    },
    [Symbol.asyncIterator]() {
      return this;
    },
  };
}
async function $c(e) {
  var t;
  if (typeof e != `object` || !e) return;
  if (e[Symbol.asyncIterator]) {
    await (t = e[Symbol.asyncIterator]()).return?.call(t);
    return;
  }
  let n = e.getReader(),
    r = n.cancel();
  (n.releaseLock(), await r);
}
var el = ({ headers: e, body: t }) => ({
  bodyHeaders: { "content-type": `application/json` },
  body: JSON.stringify(t),
});
function tl(e) {
  return Object.entries(e)
    .filter(([e, t]) => t !== void 0)
    .map(([e, t]) => {
      if (typeof t == `string` || typeof t == `number` || typeof t == `boolean`)
        return `${encodeURIComponent(e)}=${encodeURIComponent(t)}`;
      if (t === null) return `${encodeURIComponent(e)}=`;
      throw new Y(
        `Cannot stringify type ${typeof t}; Expected string, number, boolean, or null. If you need to pass nested query parameters, you can manually encode them, e.g. { query: { 'foo[key1]': value1, 'foo[key2]': value2 } }, and please open a GitHub issue requesting better support for your use case.`,
      );
    })
    .join(`&`);
}
var nl = `0.0.1`,
  rl = () => {
    if (typeof File > `u`) {
      let { process: e } = globalThis,
        t = typeof e?.versions?.node == `string` && parseInt(e.versions.node.split(`.`)) < 20;
      throw Error(
        "`File` is not defined as a global, which is required for file uploads." +
          (t
            ? " Update to Node 20 LTS or newer, or set `globalThis.File` to `import('node:buffer').File`."
            : ``),
      );
    }
  };
function il(e, t, n) {
  return (rl(), new File(e, t ?? `unknown_file`, n));
}
function al(e) {
  return (
    (
      (typeof e == `object` &&
        !!e &&
        ((`name` in e && e.name && String(e.name)) ||
          (`url` in e && e.url && String(e.url)) ||
          (`filename` in e && e.filename && String(e.filename)) ||
          (`path` in e && e.path && String(e.path)))) ||
      ``
    )
      .split(/[\\/]/)
      .pop() || void 0
  );
}
var ol = (e) => typeof e == `object` && !!e && typeof e[Symbol.asyncIterator] == `function`,
  sl = (e) =>
    typeof e == `object` &&
    !!e &&
    typeof e.size == `number` &&
    typeof e.type == `string` &&
    typeof e.text == `function` &&
    typeof e.slice == `function` &&
    typeof e.arrayBuffer == `function`,
  cl = (e) =>
    typeof e == `object` &&
    !!e &&
    typeof e.name == `string` &&
    typeof e.lastModified == `number` &&
    sl(e),
  ll = (e) =>
    typeof e == `object` && !!e && typeof e.url == `string` && typeof e.blob == `function`;
async function ul(e, t, n) {
  if ((rl(), (e = await e), cl(e)))
    return e instanceof File ? e : il([await e.arrayBuffer()], e.name);
  if (ll(e)) {
    let r = await e.blob();
    return ((t ||= new URL(e.url).pathname.split(/[\\/]/).pop()), il(await dl(r), t, n));
  }
  let r = await dl(e);
  if (((t ||= al(e)), !n?.type)) {
    let e = r.find((e) => typeof e == `object` && `type` in e && e.type);
    typeof e == `string` && (n = Object.assign(Object.assign({}, n), { type: e }));
  }
  return il(r, t, n);
}
async function dl(e) {
  var t, n, r, i;
  let a = [];
  if (typeof e == `string` || ArrayBuffer.isView(e) || e instanceof ArrayBuffer) a.push(e);
  else if (sl(e)) a.push(e instanceof Blob ? e : await e.arrayBuffer());
  else if (ol(e))
    try {
      for (var o = !0, s = K(e), c; (c = await s.next()), (t = c.done), !t; o = !0) {
        ((i = c.value), (o = !1));
        let e = i;
        a.push(...(await dl(e)));
      }
    } catch (e) {
      n = { error: e };
    } finally {
      try {
        !o && !t && (r = s.return) && (await r.call(s));
      } finally {
        if (n) throw n.error;
      }
    }
  else {
    let t = e?.constructor?.name;
    throw Error(`Unexpected data type: ${typeof e}${t ? `; constructor: ${t}` : ``}${fl(e)}`);
  }
  return a;
}
function fl(e) {
  return typeof e != `object` || !e
    ? ``
    : `; props: [${Object.getOwnPropertyNames(e)
        .map((e) => `"${e}"`)
        .join(`, `)}]`;
}
var pl = class {
  constructor(e) {
    this._client = e;
  }
};
pl._key = [];
function ml(e) {
  return e.replace(/[^A-Za-z0-9\-._~!$&'()*+,;=:@]+/g, encodeURIComponent);
}
var hl = Object.freeze(Object.create(null)),
  gl = ((e = ml) =>
    function (t, ...n) {
      if (t.length === 1) return t[0];
      let r = !1,
        i = [],
        a = t.reduce((t, a, o) => {
          /[?#]/.test(a) && (r = !0);
          let s = n[o],
            c = (r ? encodeURIComponent : e)(`` + s);
          return (
            o !== n.length &&
              (s == null ||
                (typeof s == `object` &&
                  s.toString ===
                    Object.getPrototypeOf(Object.getPrototypeOf(s.hasOwnProperty ?? hl) ?? hl)
                      ?.toString)) &&
              ((c = s + ``),
              i.push({
                start: t.length + a.length,
                length: c.length,
                error: `Value of type ${Object.prototype.toString.call(s).slice(8, -1)} is not a valid path parameter`,
              })),
            t + a + (o === n.length ? `` : c)
          );
        }, ``),
        o = a.split(/[?#]/, 1)[0],
        s = /(^|\/)(?:\.|%2e){1,2}(?=\/|$)/gi,
        c;
      for (; (c = s.exec(o)) !== null; ) {
        let e = c[0].startsWith(`/`),
          t = e ? 1 : 0,
          n = e ? c[0].slice(1) : c[0];
        i.push({
          start: c.index + t,
          length: n.length,
          error: `Value "${n}" can\'t be safely passed as a path parameter`,
        });
      }
      if ((i.sort((e, t) => e.start - t.start), i.length > 0)) {
        let e = 0,
          t = i.reduce((t, n) => {
            let r = ` `.repeat(n.start - e),
              i = `^`.repeat(n.length);
            return ((e = n.start + n.length), t + r + i);
          }, ``);
        throw new Y(
          `Path parameters result in path with invalid segments:\n${i.map((e) => e.error).join(`
`)}\n${a}\n${t}`,
        );
      }
      return a;
    })(ml),
  _l = class extends pl {
    create(e, t) {
      let { api_version: n = this._client.apiVersion } = e,
        r = Zr(e, [`api_version`]);
      if (`model` in r && `agent_config` in r)
        throw new Y(
          "Invalid request: specified `model` and `agent_config`. If specifying `model`, use `generation_config`.",
        );
      if (`agent` in r && `generation_config` in r)
        throw new Y(
          "Invalid request: specified `agent` and `generation_config`. If specifying `agent`, use `agent_config`.",
        );
      return this._client.post(
        gl`/${n}/interactions`,
        Object.assign(Object.assign({ body: r }, t), { stream: e.stream ?? !1 }),
      );
    }
    delete(e, t = {}, n) {
      let { api_version: r = this._client.apiVersion } = t ?? {};
      return this._client.delete(gl`/${r}/interactions/${e}`, n);
    }
    cancel(e, t = {}, n) {
      let { api_version: r = this._client.apiVersion } = t ?? {};
      return this._client.post(gl`/${r}/interactions/${e}/cancel`, n);
    }
    get(e, t = {}, n) {
      let r = t ?? {},
        { api_version: i = this._client.apiVersion } = r,
        a = Zr(r, [`api_version`]);
      return this._client.get(
        gl`/${i}/interactions/${e}`,
        Object.assign(Object.assign({ query: a }, n), { stream: t?.stream ?? !1 }),
      );
    }
  };
_l._key = Object.freeze([`interactions`]);
var vl = class extends _l {};
function yl(e) {
  let t = 0;
  for (let n of e) t += n.length;
  let n = new Uint8Array(t),
    r = 0;
  for (let t of e) (n.set(t, r), (r += t.length));
  return n;
}
var bl;
function xl(e) {
  let t;
  return (bl ??= ((t = new globalThis.TextEncoder()), t.encode.bind(t)))(e);
}
var Sl;
function Cl(e) {
  let t;
  return (Sl ??= ((t = new globalThis.TextDecoder()), t.decode.bind(t)))(e);
}
var wl = class {
  constructor() {
    ((this.buffer = new Uint8Array()), (this.carriageReturnIndex = null), (this.searchIndex = 0));
  }
  decode(e) {
    if (e == null) return [];
    let t = e instanceof ArrayBuffer ? new Uint8Array(e) : typeof e == `string` ? xl(e) : e;
    this.buffer = yl([this.buffer, t]);
    let n = [],
      r;
    for (; (r = Tl(this.buffer, this.carriageReturnIndex ?? this.searchIndex)) != null; ) {
      if (r.carriage && this.carriageReturnIndex == null) {
        this.carriageReturnIndex = r.index;
        continue;
      }
      if (
        this.carriageReturnIndex != null &&
        (r.index !== this.carriageReturnIndex + 1 || r.carriage)
      ) {
        (n.push(Cl(this.buffer.subarray(0, this.carriageReturnIndex - 1))),
          (this.buffer = this.buffer.subarray(this.carriageReturnIndex)),
          (this.carriageReturnIndex = null),
          (this.searchIndex = 0));
        continue;
      }
      let e = this.carriageReturnIndex === null ? r.preceding : r.preceding - 1,
        t = Cl(this.buffer.subarray(0, e));
      (n.push(t),
        (this.buffer = this.buffer.subarray(r.index)),
        (this.carriageReturnIndex = null),
        (this.searchIndex = 0));
    }
    return ((this.searchIndex = Math.max(0, this.buffer.length - 1)), n);
  }
  flush() {
    return this.buffer.length
      ? this.decode(`
`)
      : [];
  }
};
((wl.NEWLINE_CHARS = new Set([
  `
`,
  `\r`,
])),
  (wl.NEWLINE_REGEXP = /\r\n|[\n\r]/g));
function Tl(e, t) {
  let n = t ?? 0,
    r = e.indexOf(10, n),
    i = e.indexOf(13, n);
  if (r === -1 && i === -1) return null;
  let a;
  return (
    (a = r !== -1 && i !== -1 ? Math.min(r, i) : r === -1 ? i : r),
    e[a] === 10
      ? { preceding: a, index: a + 1, carriage: !1 }
      : { preceding: a, index: a + 1, carriage: !0 }
  );
}
var El = { off: 0, error: 200, warn: 300, info: 400, debug: 500 },
  Dl = (e, t, n) => {
    if (e) {
      if (Gc(El, e)) return e;
      Z(n).warn(
        `${t} was set to ${JSON.stringify(e)}, expected one of ${JSON.stringify(Object.keys(El))}`,
      );
    }
  };
function Ol() {}
function kl(e, t, n) {
  return !t || El[e] > El[n] ? Ol : t[e].bind(t);
}
var Al = { error: Ol, warn: Ol, info: Ol, debug: Ol },
  jl = new WeakMap();
function Z(e) {
  let t = e.logger,
    n = e.logLevel ?? `off`;
  if (!t) return Al;
  let r = jl.get(t);
  if (r && r[0] === n) return r[1];
  let i = {
    error: kl(`error`, t, n),
    warn: kl(`warn`, t, n),
    info: kl(`info`, t, n),
    debug: kl(`debug`, t, n),
  };
  return (jl.set(t, [n, i]), i);
}
var Q = (e) => (
    e.options && ((e.options = Object.assign({}, e.options)), delete e.options.headers),
    (e.headers &&= Object.fromEntries(
      (e.headers instanceof Headers ? [...e.headers] : Object.entries(e.headers)).map(([e, t]) => [
        e,
        e.toLowerCase() === `x-goog-api-key` ||
        e.toLowerCase() === `authorization` ||
        e.toLowerCase() === `cookie` ||
        e.toLowerCase() === `set-cookie`
          ? `***`
          : t,
      ]),
    )),
    `retryOfRequestLogID` in e &&
      (e.retryOfRequestLogID && (e.retryOf = e.retryOfRequestLogID), delete e.retryOfRequestLogID),
    e
  ),
  Ml = class e {
    constructor(e, t, n) {
      ((this.iterator = e), (this.controller = t), (this.client = n));
    }
    static fromSSEResponse(t, n, r) {
      let i = !1,
        a = r ? Z(r) : console;
      function o() {
        return G(this, arguments, function* () {
          var e, r, o, s;
          if (i)
            throw new Y("Cannot iterate over a consumed stream, use `.tee()` to split the stream.");
          i = !0;
          let c = !1;
          try {
            try {
              for (
                var l = !0, u = K(Nl(t, n)), d;
                (d = yield W(u.next())), (e = d.done), !e;
                l = !0
              ) {
                ((s = d.value), (l = !1));
                let e = s;
                if (!c)
                  if (e.data.startsWith(`[DONE]`)) {
                    c = !0;
                    continue;
                  } else
                    try {
                      yield yield W(JSON.parse(e.data));
                    } catch (t) {
                      throw (
                        a.error(`Could not parse message into JSON:`, e.data),
                        a.error(`From chunk:`, e.raw),
                        t
                      );
                    }
              }
            } catch (e) {
              r = { error: e };
            } finally {
              try {
                !l && !e && (o = u.return) && (yield W(o.call(u)));
              } finally {
                if (r) throw r.error;
              }
            }
            c = !0;
          } catch (e) {
            if (Dc(e)) return yield W(void 0);
            throw e;
          } finally {
            c || n.abort();
          }
        });
      }
      return new e(o, n, r);
    }
    static fromReadableStream(t, n, r) {
      let i = !1;
      function a() {
        return G(this, arguments, function* () {
          var e, n, r, i;
          let a = new wl(),
            o = Qc(t);
          try {
            for (var s = !0, c = K(o), l; (l = yield W(c.next())), (e = l.done), !e; s = !0) {
              ((i = l.value), (s = !1));
              let e = i;
              for (let t of a.decode(e)) yield yield W(t);
            }
          } catch (e) {
            n = { error: e };
          } finally {
            try {
              !s && !e && (r = c.return) && (yield W(r.call(c)));
            } finally {
              if (n) throw n.error;
            }
          }
          for (let e of a.flush()) yield yield W(e);
        });
      }
      function o() {
        return G(this, arguments, function* () {
          var e, t, r, o;
          if (i)
            throw new Y("Cannot iterate over a consumed stream, use `.tee()` to split the stream.");
          i = !0;
          let s = !1;
          try {
            try {
              for (var c = !0, l = K(a()), u; (u = yield W(l.next())), (e = u.done), !e; c = !0) {
                ((o = u.value), (c = !1));
                let e = o;
                s || (e && (yield yield W(JSON.parse(e))));
              }
            } catch (e) {
              t = { error: e };
            } finally {
              try {
                !c && !e && (r = l.return) && (yield W(r.call(l)));
              } finally {
                if (t) throw t.error;
              }
            }
            s = !0;
          } catch (e) {
            if (Dc(e)) return yield W(void 0);
            throw e;
          } finally {
            s || n.abort();
          }
        });
      }
      return new e(o, n, r);
    }
    [Symbol.asyncIterator]() {
      return this.iterator();
    }
    tee() {
      let t = [],
        n = [],
        r = this.iterator(),
        i = (e) => ({
          next: () => {
            if (e.length === 0) {
              let e = r.next();
              (t.push(e), n.push(e));
            }
            return e.shift();
          },
        });
      return [
        new e(() => i(t), this.controller, this.client),
        new e(() => i(n), this.controller, this.client),
      ];
    }
    toReadableStream() {
      let e = this,
        t;
      return Xc({
        async start() {
          t = e[Symbol.asyncIterator]();
        },
        async pull(e) {
          try {
            let { value: n, done: r } = await t.next();
            if (r) return e.close();
            let i = xl(
              JSON.stringify(n) +
                `
`,
            );
            e.enqueue(i);
          } catch (t) {
            e.error(t);
          }
        },
        async cancel() {
          await t.return?.call(t);
        },
      });
    }
  };
function Nl(e, t) {
  return G(this, arguments, function* () {
    var n, r, i, a;
    if (!e.body)
      throw (
        t.abort(),
        globalThis.navigator !== void 0 && globalThis.navigator.product === `ReactNative`
          ? new Y(
              `The default react-native fetch implementation does not support streaming. Please use expo/fetch: https://docs.expo.dev/versions/latest/sdk/expo/#expofetch-api`,
            )
          : new Y(`Attempted to iterate over a response with no body`)
      );
    let o = new Fl(),
      s = new wl(),
      c = Qc(e.body);
    try {
      for (var l = !0, u = K(Pl(c)), d; (d = yield W(u.next())), (n = d.done), !n; l = !0) {
        ((a = d.value), (l = !1));
        let e = a;
        for (let t of s.decode(e)) {
          let e = o.decode(t);
          e && (yield yield W(e));
        }
      }
    } catch (e) {
      r = { error: e };
    } finally {
      try {
        !l && !n && (i = u.return) && (yield W(i.call(u)));
      } finally {
        if (r) throw r.error;
      }
    }
    for (let e of s.flush()) {
      let t = o.decode(e);
      t && (yield yield W(t));
    }
  });
}
function Pl(e) {
  return G(this, arguments, function* () {
    var t, n, r, i;
    try {
      for (var a = !0, o = K(e), s; (s = yield W(o.next())), (t = s.done), !t; a = !0) {
        ((i = s.value), (a = !1));
        let e = i;
        e != null &&
          (yield yield W(
            e instanceof ArrayBuffer ? new Uint8Array(e) : typeof e == `string` ? xl(e) : e,
          ));
      }
    } catch (e) {
      n = { error: e };
    } finally {
      try {
        !a && !t && (r = o.return) && (yield W(r.call(o)));
      } finally {
        if (n) throw n.error;
      }
    }
  });
}
var Fl = class {
  constructor() {
    ((this.event = null), (this.data = []), (this.chunks = []));
  }
  decode(e) {
    if ((e.endsWith(`\r`) && (e = e.substring(0, e.length - 1)), !e)) {
      if (!this.event && !this.data.length) return null;
      let e = {
        event: this.event,
        data: this.data.join(`
`),
        raw: this.chunks,
      };
      return ((this.event = null), (this.data = []), (this.chunks = []), e);
    }
    if ((this.chunks.push(e), e.startsWith(`:`))) return null;
    let [t, n, r] = Il(e, `:`);
    return (
      r.startsWith(` `) && (r = r.substring(1)),
      t === `event` ? (this.event = r) : t === `data` && this.data.push(r),
      null
    );
  }
};
function Il(e, t) {
  let n = e.indexOf(t);
  return n === -1 ? [e, ``, ``] : [e.substring(0, n), t, e.substring(n + t.length)];
}
async function Ll(e, t) {
  let { response: n, requestLogID: r, retryOfRequestLogID: i, startTime: a } = t,
    o = await (async () => {
      if (t.options.stream)
        return (
          Z(e).debug(`response`, n.status, n.url, n.headers, n.body),
          t.options.__streamClass
            ? t.options.__streamClass.fromSSEResponse(n, t.controller, e)
            : Ml.fromSSEResponse(n, t.controller, e)
        );
      if (n.status === 204) return null;
      if (t.options.__binaryResponse) return n;
      let r = n.headers.get(`content-type`)?.split(`;`)[0]?.trim();
      return r?.includes(`application/json`) || r?.endsWith(`+json`)
        ? n.headers.get(`content-length`) === `0`
          ? void 0
          : await n.json()
        : await n.text();
    })();
  return (
    Z(e).debug(
      `[${r}] response parsed`,
      Q({
        retryOfRequestLogID: i,
        url: n.url,
        status: n.status,
        body: o,
        durationMs: Date.now() - a,
      }),
    ),
    o
  );
}
var Rl = class e extends Promise {
    constructor(e, t, n = Ll) {
      (super((e) => {
        e(null);
      }),
        (this.responsePromise = t),
        (this.parseResponse = n),
        (this.client = e));
    }
    _thenUnwrap(t) {
      return new e(this.client, this.responsePromise, async (e, n) =>
        t(await this.parseResponse(e, n), n),
      );
    }
    asResponse() {
      return this.responsePromise.then((e) => e.response);
    }
    async withResponse() {
      let [e, t] = await Promise.all([this.parse(), this.asResponse()]);
      return { data: e, response: t };
    }
    parse() {
      return (
        (this.parsedPromise ||= this.responsePromise.then((e) =>
          this.parseResponse(this.client, e),
        )),
        this.parsedPromise
      );
    }
    then(e, t) {
      return this.parse().then(e, t);
    }
    catch(e) {
      return this.parse().catch(e);
    }
    finally(e) {
      return this.parse().finally(e);
    }
  },
  zl = Symbol(`brand.privateNullableHeaders`);
function* Bl(e) {
  if (!e) return;
  if (zl in e) {
    let { values: t, nulls: n } = e;
    yield* t.entries();
    for (let e of n) yield [e, null];
    return;
  }
  let t = !1,
    n;
  e instanceof Headers
    ? (n = e.entries())
    : Uc(e)
      ? (n = e)
      : ((t = !0), (n = Object.entries(e ?? {})));
  for (let e of n) {
    let n = e[0];
    if (typeof n != `string`) throw TypeError(`expected header name to be a string`);
    let r = Uc(e[1]) ? e[1] : [e[1]],
      i = !1;
    for (let e of r) e !== void 0 && (t && !i && ((i = !0), yield [n, null]), yield [n, e]);
  }
}
var Vl = (e) => {
    let t = new Headers(),
      n = new Set();
    for (let r of e) {
      let e = new Set();
      for (let [i, a] of Bl(r)) {
        let r = i.toLowerCase();
        (e.has(r) || (t.delete(i), e.add(r)),
          a === null ? (t.delete(i), n.add(r)) : (t.append(i, a), n.delete(r)));
      }
    }
    return { [zl]: !0, values: t, nulls: n };
  },
  Hl = (e) => {
    var t;
    if (globalThis.process !== void 0) return {}?.[e]?.trim() ?? void 0;
    if (globalThis.Deno !== void 0) return (t = globalThis.Deno.env)?.get?.call(t, e)?.trim();
  },
  Ul,
  Wl = class e {
    constructor(t) {
      var {
          baseURL: n = Hl(`GEMINI_NEXT_GEN_API_BASE_URL`),
          apiKey: r = Hl(`GEMINI_API_KEY`) ?? null,
          apiVersion: i = `v1beta`,
        } = t,
        a = Zr(t, [`baseURL`, `apiKey`, `apiVersion`]);
      let o = Object.assign(Object.assign({ apiKey: r, apiVersion: i }, a), {
        baseURL: n || `https://generativelanguage.googleapis.com`,
      });
      ((this.baseURL = o.baseURL),
        (this.timeout = o.timeout ?? e.DEFAULT_TIMEOUT),
        (this.logger = o.logger ?? console));
      let s = `warn`;
      ((this.logLevel = s),
        (this.logLevel =
          Dl(o.logLevel, `ClientOptions.logLevel`, this) ??
          Dl(Hl(`GEMINI_NEXT_GEN_API_LOG`), `process.env['GEMINI_NEXT_GEN_API_LOG']`, this) ??
          s),
        (this.fetchOptions = o.fetchOptions),
        (this.maxRetries = o.maxRetries ?? 2),
        (this.fetch = o.fetch ?? Yc()),
        (this.encoder = el),
        (this._options = o),
        (this.apiKey = r),
        (this.apiVersion = i),
        (this.clientAdapter = o.clientAdapter));
    }
    withOptions(e) {
      return new this.constructor(
        Object.assign(
          Object.assign(Object.assign({}, this._options), {
            baseURL: this.baseURL,
            maxRetries: this.maxRetries,
            timeout: this.timeout,
            logger: this.logger,
            logLevel: this.logLevel,
            fetch: this.fetch,
            fetchOptions: this.fetchOptions,
            apiKey: this.apiKey,
            apiVersion: this.apiVersion,
          }),
          e,
        ),
      );
    }
    baseURLOverridden() {
      return this.baseURL !== `https://generativelanguage.googleapis.com`;
    }
    defaultQuery() {
      return this._options.defaultQuery;
    }
    validateHeaders({ values: e, nulls: t }) {
      if (
        !(e.has(`authorization`) || e.has(`x-goog-api-key`)) &&
        !(this.apiKey && e.get(`x-goog-api-key`)) &&
        !t.has(`x-goog-api-key`)
      )
        throw Error(
          `Could not resolve authentication method. Expected the apiKey to be set. Or for the "x-goog-api-key" headers to be explicitly omitted`,
        );
    }
    async authHeaders(e) {
      let t = Vl([e.headers]);
      if (!(t.values.has(`authorization`) || t.values.has(`x-goog-api-key`))) {
        if (this.apiKey) return Vl([{ "x-goog-api-key": this.apiKey }]);
        if (this.clientAdapter.isVertexAI()) return Vl([await this.clientAdapter.getAuthHeaders()]);
      }
    }
    stringifyQuery(e) {
      return tl(e);
    }
    getUserAgent() {
      return `${this.constructor.name}/JS ${nl}`;
    }
    defaultIdempotencyKey() {
      return `stainless-node-retry-${Ec()}`;
    }
    makeStatusError(e, t, n, r) {
      return X.generate(e, t, n, r);
    }
    buildURL(e, t, n) {
      let r = (!this.baseURLOverridden() && n) || this.baseURL,
        i = Vc(e)
          ? new URL(e)
          : new URL(r + (r.endsWith(`/`) && e.startsWith(`/`) ? e.slice(1) : e)),
        a = this.defaultQuery(),
        o = Object.fromEntries(i.searchParams);
      return (
        (!Wc(a) || !Wc(o)) && (t = Object.assign(Object.assign(Object.assign({}, o), a), t)),
        typeof t == `object` && t && !Array.isArray(t) && (i.search = this.stringifyQuery(t)),
        i.toString()
      );
    }
    async prepareOptions(e) {
      if (
        this.clientAdapter &&
        this.clientAdapter.isVertexAI() &&
        !e.path.startsWith(`/${this.apiVersion}/projects/`)
      ) {
        let t = e.path.slice(this.apiVersion.length + 1);
        e.path = `/${this.apiVersion}/projects/${this.clientAdapter.getProject()}/locations/${this.clientAdapter.getLocation()}${t}`;
      }
    }
    async prepareRequest(e, { url: t, options: n }) {}
    get(e, t) {
      return this.methodRequest(`get`, e, t);
    }
    post(e, t) {
      return this.methodRequest(`post`, e, t);
    }
    patch(e, t) {
      return this.methodRequest(`patch`, e, t);
    }
    put(e, t) {
      return this.methodRequest(`put`, e, t);
    }
    delete(e, t) {
      return this.methodRequest(`delete`, e, t);
    }
    methodRequest(e, t, n) {
      return this.request(Promise.resolve(n).then((n) => Object.assign({ method: e, path: t }, n)));
    }
    request(e, t = null) {
      return new Rl(this, this.makeRequest(e, t, void 0));
    }
    async makeRequest(e, t, n) {
      let r = await e,
        i = r.maxRetries ?? this.maxRetries;
      ((t ??= i), await this.prepareOptions(r));
      let { req: a, url: o, timeout: s } = await this.buildRequest(r, { retryCount: i - t });
      await this.prepareRequest(a, { url: o, options: r });
      let c = `log_` + ((Math.random() * (1 << 24)) | 0).toString(16).padStart(6, `0`),
        l = n === void 0 ? `` : `, retryOf: ${n}`,
        u = Date.now();
      if (
        (Z(this).debug(
          `[${c}] sending request`,
          Q({ retryOfRequestLogID: n, method: r.method, url: o, options: r, headers: a.headers }),
        ),
        r.signal?.aborted)
      )
        throw new kc();
      let d = new AbortController(),
        f = await this.fetchWithTimeout(o, a, s, d).catch(Oc),
        p = Date.now();
      if (f instanceof globalThis.Error) {
        let e = `retrying, ${t} attempts remaining`;
        if (r.signal?.aborted) throw new kc();
        let i = Dc(f) || /timed? ?out/i.test(String(f) + (`cause` in f ? String(f.cause) : ``));
        if (t)
          return (
            Z(this).info(`[${c}] connection ${i ? `timed out` : `failed`} - ${e}`),
            Z(this).debug(
              `[${c}] connection ${i ? `timed out` : `failed`} (${e})`,
              Q({ retryOfRequestLogID: n, url: o, durationMs: p - u, message: f.message }),
            ),
            this.retryRequest(r, t, n ?? c)
          );
        throw (
          Z(this).info(
            `[${c}] connection ${i ? `timed out` : `failed`} - error; no more retries left`,
          ),
          Z(this).debug(
            `[${c}] connection ${i ? `timed out` : `failed`} (error; no more retries left)`,
            Q({ retryOfRequestLogID: n, url: o, durationMs: p - u, message: f.message }),
          ),
          i ? new jc() : new Ac({ cause: f })
        );
      }
      let m = `[${c}${l}] ${a.method} ${o} ${f.ok ? `succeeded` : `failed`} with status ${f.status} in ${p - u}ms`;
      if (!f.ok) {
        let e = await this.shouldRetry(f);
        if (t && e) {
          let e = `retrying, ${t} attempts remaining`;
          return (
            await $c(f.body),
            Z(this).info(`${m} - ${e}`),
            Z(this).debug(
              `[${c}] response error (${e})`,
              Q({
                retryOfRequestLogID: n,
                url: f.url,
                status: f.status,
                headers: f.headers,
                durationMs: p - u,
              }),
            ),
            this.retryRequest(r, t, n ?? c, f.headers)
          );
        }
        let i = e ? `error; no more retries left` : `error; not retryable`;
        Z(this).info(`${m} - ${i}`);
        let a = await f.text().catch((e) => Oc(e).message),
          o = qc(a),
          s = o ? void 0 : a;
        throw (
          Z(this).debug(
            `[${c}] response error (${i})`,
            Q({
              retryOfRequestLogID: n,
              url: f.url,
              status: f.status,
              headers: f.headers,
              message: s,
              durationMs: Date.now() - u,
            }),
          ),
          this.makeStatusError(f.status, o, s, f.headers)
        );
      }
      return (
        Z(this).info(m),
        Z(this).debug(
          `[${c}] response start`,
          Q({
            retryOfRequestLogID: n,
            url: f.url,
            status: f.status,
            headers: f.headers,
            durationMs: p - u,
          }),
        ),
        {
          response: f,
          options: r,
          controller: d,
          requestLogID: c,
          retryOfRequestLogID: n,
          startTime: u,
        }
      );
    }
    async fetchWithTimeout(e, t, n, r) {
      let i = t || {},
        { signal: a, method: o } = i,
        s = Zr(i, [`signal`, `method`]),
        c = this._makeAbort(r);
      a && a.addEventListener(`abort`, c, { once: !0 });
      let l = setTimeout(c, n),
        u =
          (globalThis.ReadableStream && s.body instanceof globalThis.ReadableStream) ||
          (typeof s.body == `object` && s.body !== null && Symbol.asyncIterator in s.body),
        d = Object.assign(
          Object.assign(Object.assign({ signal: r.signal }, u ? { duplex: `half` } : {}), {
            method: `GET`,
          }),
          s,
        );
      o && (d.method = o.toUpperCase());
      try {
        return await this.fetch.call(void 0, e, d);
      } finally {
        clearTimeout(l);
      }
    }
    async shouldRetry(e) {
      let t = e.headers.get(`x-should-retry`);
      return t === `true`
        ? !0
        : t === `false`
          ? !1
          : e.status === 408 || e.status === 409 || e.status === 429 || e.status >= 500;
    }
    async retryRequest(e, t, n, r) {
      let i,
        a = r?.get(`retry-after-ms`);
      if (a) {
        let e = parseFloat(a);
        Number.isNaN(e) || (i = e);
      }
      let o = r?.get(`retry-after`);
      if (o && !i) {
        let e = parseFloat(o);
        i = Number.isNaN(e) ? Date.parse(o) - Date.now() : e * 1e3;
      }
      if (i === void 0) {
        let n = e.maxRetries ?? this.maxRetries;
        i = this.calculateDefaultRetryTimeoutMillis(t, n);
      }
      return (await Jc(i), this.makeRequest(e, t - 1, n));
    }
    calculateDefaultRetryTimeoutMillis(e, t) {
      let n = t - e;
      return Math.min(0.5 * 2 ** n, 8) * (1 - Math.random() * 0.25) * 1e3;
    }
    async buildRequest(e, { retryCount: t = 0 } = {}) {
      let n = Object.assign({}, e),
        { method: r, path: i, query: a, defaultBaseURL: o } = n,
        s = this.buildURL(i, a, o);
      (`timeout` in n && Kc(`timeout`, n.timeout), (n.timeout = n.timeout ?? this.timeout));
      let { bodyHeaders: c, body: l } = this.buildBody({ options: n }),
        u = await this.buildHeaders({ options: e, method: r, bodyHeaders: c, retryCount: t });
      return {
        req: Object.assign(
          Object.assign(
            Object.assign(
              Object.assign(
                Object.assign({ method: r, headers: u }, n.signal && { signal: n.signal }),
                globalThis.ReadableStream &&
                  l instanceof globalThis.ReadableStream && { duplex: `half` },
              ),
              l && { body: l },
            ),
            this.fetchOptions ?? {},
          ),
          n.fetchOptions ?? {},
        ),
        url: s,
        timeout: n.timeout,
      };
    }
    async buildHeaders({ options: e, method: t, bodyHeaders: n, retryCount: r }) {
      let i = {};
      this.idempotencyHeader &&
        t !== `get` &&
        ((e.idempotencyKey ||= this.defaultIdempotencyKey()),
        (i[this.idempotencyHeader] = e.idempotencyKey));
      let a = await this.authHeaders(e),
        o = Vl([
          i,
          { Accept: `application/json`, "User-Agent": this.getUserAgent() },
          this._options.defaultHeaders,
          n,
          e.headers,
          a,
        ]);
      return (this.validateHeaders(o), o.values);
    }
    _makeAbort(e) {
      return () => e.abort();
    }
    buildBody({ options: { body: e, headers: t } }) {
      if (!e) return { bodyHeaders: void 0, body: void 0 };
      let n = Vl([t]);
      return ArrayBuffer.isView(e) ||
        e instanceof ArrayBuffer ||
        e instanceof DataView ||
        (typeof e == `string` && n.values.has(`content-type`)) ||
        (globalThis.Blob && e instanceof globalThis.Blob) ||
        e instanceof FormData ||
        e instanceof URLSearchParams ||
        (globalThis.ReadableStream && e instanceof globalThis.ReadableStream)
        ? { bodyHeaders: void 0, body: e }
        : typeof e == `object` &&
            (Symbol.asyncIterator in e ||
              (Symbol.iterator in e && `next` in e && typeof e.next == `function`))
          ? { bodyHeaders: void 0, body: Zc(e) }
          : typeof e == `object` &&
              n.values.get(`content-type`) === `application/x-www-form-urlencoded`
            ? {
                bodyHeaders: { "content-type": `application/x-www-form-urlencoded` },
                body: this.stringifyQuery(e),
              }
            : this.encoder({ body: e, headers: n });
    }
  };
Wl.DEFAULT_TIMEOUT = 6e4;
var $ = class extends Wl {
  constructor() {
    (super(...arguments), (this.interactions = new vl(this)));
  }
};
((Ul = $),
  ($.GeminiNextGenAPIClient = Ul),
  ($.GeminiNextGenAPIClientError = Y),
  ($.APIError = X),
  ($.APIConnectionError = Ac),
  ($.APIConnectionTimeoutError = jc),
  ($.APIUserAbortError = kc),
  ($.NotFoundError = Fc),
  ($.ConflictError = Ic),
  ($.RateLimitError = Rc),
  ($.BadRequestError = Mc),
  ($.AuthenticationError = Nc),
  ($.InternalServerError = zc),
  ($.PermissionDeniedError = Pc),
  ($.UnprocessableEntityError = Lc),
  ($.toFile = ul),
  ($.Interactions = vl));
function Gl(e, t) {
  let n = {},
    r = h(e, [`name`]);
  return (r != null && m(n, [`_url`, `name`], r), n);
}
function Kl(e, t) {
  let n = {},
    r = h(e, [`name`]);
  return (r != null && m(n, [`_url`, `name`], r), n);
}
function ql(e, t) {
  let n = {},
    r = h(e, [`sdkHttpResponse`]);
  return (r != null && m(n, [`sdkHttpResponse`], r), n);
}
function Jl(e, t) {
  let n = {},
    r = h(e, [`sdkHttpResponse`]);
  return (r != null && m(n, [`sdkHttpResponse`], r), n);
}
function Yl(e, t, n) {
  let r = {};
  if (h(e, [`validationDataset`]) !== void 0)
    throw Error(`validationDataset parameter is not supported in Gemini API.`);
  let i = h(e, [`tunedModelDisplayName`]);
  if ((t !== void 0 && i != null && m(t, [`displayName`], i), h(e, [`description`]) !== void 0))
    throw Error(`description parameter is not supported in Gemini API.`);
  let a = h(e, [`epochCount`]);
  t !== void 0 && a != null && m(t, [`tuningTask`, `hyperparameters`, `epochCount`], a);
  let o = h(e, [`learningRateMultiplier`]);
  if (
    (o != null && m(r, [`tuningTask`, `hyperparameters`, `learningRateMultiplier`], o),
    h(e, [`exportLastCheckpointOnly`]) !== void 0)
  )
    throw Error(`exportLastCheckpointOnly parameter is not supported in Gemini API.`);
  if (h(e, [`preTunedModelCheckpointId`]) !== void 0)
    throw Error(`preTunedModelCheckpointId parameter is not supported in Gemini API.`);
  if (h(e, [`adapterSize`]) !== void 0)
    throw Error(`adapterSize parameter is not supported in Gemini API.`);
  if (h(e, [`tuningMode`]) !== void 0)
    throw Error(`tuningMode parameter is not supported in Gemini API.`);
  if (h(e, [`customBaseModel`]) !== void 0)
    throw Error(`customBaseModel parameter is not supported in Gemini API.`);
  let s = h(e, [`batchSize`]);
  t !== void 0 && s != null && m(t, [`tuningTask`, `hyperparameters`, `batchSize`], s);
  let c = h(e, [`learningRate`]);
  if (
    (t !== void 0 && c != null && m(t, [`tuningTask`, `hyperparameters`, `learningRate`], c),
    h(e, [`labels`]) !== void 0)
  )
    throw Error(`labels parameter is not supported in Gemini API.`);
  if (h(e, [`beta`]) !== void 0) throw Error(`beta parameter is not supported in Gemini API.`);
  if (h(e, [`baseTeacherModel`]) !== void 0)
    throw Error(`baseTeacherModel parameter is not supported in Gemini API.`);
  if (h(e, [`tunedTeacherModelSource`]) !== void 0)
    throw Error(`tunedTeacherModelSource parameter is not supported in Gemini API.`);
  if (h(e, [`sftLossWeightMultiplier`]) !== void 0)
    throw Error(`sftLossWeightMultiplier parameter is not supported in Gemini API.`);
  if (h(e, [`outputUri`]) !== void 0)
    throw Error(`outputUri parameter is not supported in Gemini API.`);
  if (h(e, [`encryptionSpec`]) !== void 0)
    throw Error(`encryptionSpec parameter is not supported in Gemini API.`);
  return r;
}
function Xl(e, t, n) {
  let r = {},
    i = h(n, [`config`, `method`]);
  if ((i === void 0 && (i = `SUPERVISED_FINE_TUNING`), i === `SUPERVISED_FINE_TUNING`)) {
    let n = h(e, [`validationDataset`]);
    t !== void 0 && n != null && m(t, [`supervisedTuningSpec`], pu(n));
  } else if (i === `PREFERENCE_TUNING`) {
    let n = h(e, [`validationDataset`]);
    t !== void 0 && n != null && m(t, [`preferenceOptimizationSpec`], pu(n));
  } else if (i === `DISTILLATION`) {
    let n = h(e, [`validationDataset`]);
    t !== void 0 && n != null && m(t, [`distillationSpec`], pu(n));
  }
  let a = h(e, [`tunedModelDisplayName`]);
  t !== void 0 && a != null && m(t, [`tunedModelDisplayName`], a);
  let o = h(e, [`description`]);
  t !== void 0 && o != null && m(t, [`description`], o);
  let s = h(n, [`config`, `method`]);
  if ((s === void 0 && (s = `SUPERVISED_FINE_TUNING`), s === `SUPERVISED_FINE_TUNING`)) {
    let n = h(e, [`epochCount`]);
    t !== void 0 && n != null && m(t, [`supervisedTuningSpec`, `hyperParameters`, `epochCount`], n);
  } else if (s === `PREFERENCE_TUNING`) {
    let n = h(e, [`epochCount`]);
    t !== void 0 &&
      n != null &&
      m(t, [`preferenceOptimizationSpec`, `hyperParameters`, `epochCount`], n);
  } else if (s === `DISTILLATION`) {
    let n = h(e, [`epochCount`]);
    t !== void 0 && n != null && m(t, [`distillationSpec`, `hyperParameters`, `epochCount`], n);
  }
  let c = h(n, [`config`, `method`]);
  if ((c === void 0 && (c = `SUPERVISED_FINE_TUNING`), c === `SUPERVISED_FINE_TUNING`)) {
    let n = h(e, [`learningRateMultiplier`]);
    t !== void 0 &&
      n != null &&
      m(t, [`supervisedTuningSpec`, `hyperParameters`, `learningRateMultiplier`], n);
  } else if (c === `PREFERENCE_TUNING`) {
    let n = h(e, [`learningRateMultiplier`]);
    t !== void 0 &&
      n != null &&
      m(t, [`preferenceOptimizationSpec`, `hyperParameters`, `learningRateMultiplier`], n);
  } else if (c === `DISTILLATION`) {
    let n = h(e, [`learningRateMultiplier`]);
    t !== void 0 &&
      n != null &&
      m(t, [`distillationSpec`, `hyperParameters`, `learningRateMultiplier`], n);
  }
  let l = h(n, [`config`, `method`]);
  if ((l === void 0 && (l = `SUPERVISED_FINE_TUNING`), l === `SUPERVISED_FINE_TUNING`)) {
    let n = h(e, [`exportLastCheckpointOnly`]);
    t !== void 0 && n != null && m(t, [`supervisedTuningSpec`, `exportLastCheckpointOnly`], n);
  } else if (l === `PREFERENCE_TUNING`) {
    let n = h(e, [`exportLastCheckpointOnly`]);
    t !== void 0 &&
      n != null &&
      m(t, [`preferenceOptimizationSpec`, `exportLastCheckpointOnly`], n);
  } else if (l === `DISTILLATION`) {
    let n = h(e, [`exportLastCheckpointOnly`]);
    t !== void 0 && n != null && m(t, [`distillationSpec`, `exportLastCheckpointOnly`], n);
  }
  let u = h(n, [`config`, `method`]);
  if ((u === void 0 && (u = `SUPERVISED_FINE_TUNING`), u === `SUPERVISED_FINE_TUNING`)) {
    let n = h(e, [`adapterSize`]);
    t !== void 0 &&
      n != null &&
      m(t, [`supervisedTuningSpec`, `hyperParameters`, `adapterSize`], n);
  } else if (u === `PREFERENCE_TUNING`) {
    let n = h(e, [`adapterSize`]);
    t !== void 0 &&
      n != null &&
      m(t, [`preferenceOptimizationSpec`, `hyperParameters`, `adapterSize`], n);
  } else if (u === `DISTILLATION`) {
    let n = h(e, [`adapterSize`]);
    t !== void 0 && n != null && m(t, [`distillationSpec`, `hyperParameters`, `adapterSize`], n);
  }
  let d = h(n, [`config`, `method`]);
  if ((d === void 0 && (d = `SUPERVISED_FINE_TUNING`), d === `SUPERVISED_FINE_TUNING`)) {
    let n = h(e, [`tuningMode`]);
    t !== void 0 && n != null && m(t, [`supervisedTuningSpec`, `tuningMode`], n);
  }
  let f = h(e, [`customBaseModel`]);
  t !== void 0 && f != null && m(t, [`customBaseModel`], f);
  let p = h(n, [`config`, `method`]);
  if ((p === void 0 && (p = `SUPERVISED_FINE_TUNING`), p === `SUPERVISED_FINE_TUNING`)) {
    let n = h(e, [`batchSize`]);
    t !== void 0 && n != null && m(t, [`supervisedTuningSpec`, `hyperParameters`, `batchSize`], n);
  }
  let g = h(n, [`config`, `method`]);
  if ((g === void 0 && (g = `SUPERVISED_FINE_TUNING`), g === `SUPERVISED_FINE_TUNING`)) {
    let n = h(e, [`learningRate`]);
    t !== void 0 &&
      n != null &&
      m(t, [`supervisedTuningSpec`, `hyperParameters`, `learningRate`], n);
  }
  let _ = h(e, [`labels`]);
  t !== void 0 && _ != null && m(t, [`labels`], _);
  let v = h(e, [`beta`]);
  t !== void 0 && v != null && m(t, [`preferenceOptimizationSpec`, `hyperParameters`, `beta`], v);
  let y = h(e, [`baseTeacherModel`]);
  t !== void 0 && y != null && m(t, [`distillationSpec`, `baseTeacherModel`], y);
  let b = h(e, [`tunedTeacherModelSource`]);
  t !== void 0 && b != null && m(t, [`distillationSpec`, `tunedTeacherModelSource`], b);
  let x = h(e, [`sftLossWeightMultiplier`]);
  t !== void 0 &&
    x != null &&
    m(t, [`distillationSpec`, `hyperParameters`, `sftLossWeightMultiplier`], x);
  let S = h(e, [`outputUri`]);
  t !== void 0 && S != null && m(t, [`outputUri`], S);
  let C = h(e, [`encryptionSpec`]);
  return (t !== void 0 && C != null && m(t, [`encryptionSpec`], C), r);
}
function Zl(e, t) {
  let n = {},
    r = h(e, [`baseModel`]);
  r != null && m(n, [`baseModel`], r);
  let i = h(e, [`preTunedModel`]);
  i != null && m(n, [`preTunedModel`], i);
  let a = h(e, [`trainingDataset`]);
  a != null && cu(a);
  let o = h(e, [`config`]);
  return (o != null && Yl(o, n), n);
}
function Ql(e, t) {
  let n = {},
    r = h(e, [`baseModel`]);
  r != null && m(n, [`baseModel`], r);
  let i = h(e, [`preTunedModel`]);
  i != null && m(n, [`preTunedModel`], i);
  let a = h(e, [`trainingDataset`]);
  a != null && lu(a, n, t);
  let o = h(e, [`config`]);
  return (o != null && Xl(o, n, t), n);
}
function $l(e, t) {
  let n = {},
    r = h(e, [`name`]);
  return (r != null && m(n, [`_url`, `name`], r), n);
}
function eu(e, t) {
  let n = {},
    r = h(e, [`name`]);
  return (r != null && m(n, [`_url`, `name`], r), n);
}
function tu(e, t, n) {
  let r = {},
    i = h(e, [`pageSize`]);
  t !== void 0 && i != null && m(t, [`_query`, `pageSize`], i);
  let a = h(e, [`pageToken`]);
  t !== void 0 && a != null && m(t, [`_query`, `pageToken`], a);
  let o = h(e, [`filter`]);
  return (t !== void 0 && o != null && m(t, [`_query`, `filter`], o), r);
}
function nu(e, t, n) {
  let r = {},
    i = h(e, [`pageSize`]);
  t !== void 0 && i != null && m(t, [`_query`, `pageSize`], i);
  let a = h(e, [`pageToken`]);
  t !== void 0 && a != null && m(t, [`_query`, `pageToken`], a);
  let o = h(e, [`filter`]);
  return (t !== void 0 && o != null && m(t, [`_query`, `filter`], o), r);
}
function ru(e, t) {
  let n = {},
    r = h(e, [`config`]);
  return (r != null && tu(r, n), n);
}
function iu(e, t) {
  let n = {},
    r = h(e, [`config`]);
  return (r != null && nu(r, n), n);
}
function au(e, t) {
  let n = {},
    r = h(e, [`sdkHttpResponse`]);
  r != null && m(n, [`sdkHttpResponse`], r);
  let i = h(e, [`nextPageToken`]);
  i != null && m(n, [`nextPageToken`], i);
  let a = h(e, [`tunedModels`]);
  if (a != null) {
    let e = a;
    (Array.isArray(e) && (e = e.map((e) => uu(e))), m(n, [`tuningJobs`], e));
  }
  return n;
}
function ou(e, t) {
  let n = {},
    r = h(e, [`sdkHttpResponse`]);
  r != null && m(n, [`sdkHttpResponse`], r);
  let i = h(e, [`nextPageToken`]);
  i != null && m(n, [`nextPageToken`], i);
  let a = h(e, [`tuningJobs`]);
  if (a != null) {
    let e = a;
    (Array.isArray(e) && (e = e.map((e) => du(e))), m(n, [`tuningJobs`], e));
  }
  return n;
}
function su(e, t) {
  let n = {},
    r = h(e, [`name`]);
  r != null && m(n, [`model`], r);
  let i = h(e, [`name`]);
  return (i != null && m(n, [`endpoint`], i), n);
}
function cu(e, t) {
  let n = {};
  if (h(e, [`gcsUri`]) !== void 0) throw Error(`gcsUri parameter is not supported in Gemini API.`);
  if (h(e, [`vertexDatasetResource`]) !== void 0)
    throw Error(`vertexDatasetResource parameter is not supported in Gemini API.`);
  let r = h(e, [`examples`]);
  if (r != null) {
    let e = r;
    (Array.isArray(e) && (e = e.map((e) => e)), m(n, [`examples`, `examples`], e));
  }
  return n;
}
function lu(e, t, n) {
  let r = {},
    i = h(n, [`config`, `method`]);
  if ((i === void 0 && (i = `SUPERVISED_FINE_TUNING`), i === `SUPERVISED_FINE_TUNING`)) {
    let n = h(e, [`gcsUri`]);
    t !== void 0 && n != null && m(t, [`supervisedTuningSpec`, `trainingDatasetUri`], n);
  } else if (i === `PREFERENCE_TUNING`) {
    let n = h(e, [`gcsUri`]);
    t !== void 0 && n != null && m(t, [`preferenceOptimizationSpec`, `trainingDatasetUri`], n);
  } else if (i === `DISTILLATION`) {
    let n = h(e, [`gcsUri`]);
    t !== void 0 && n != null && m(t, [`distillationSpec`, `promptDatasetUri`], n);
  }
  let a = h(n, [`config`, `method`]);
  if ((a === void 0 && (a = `SUPERVISED_FINE_TUNING`), a === `SUPERVISED_FINE_TUNING`)) {
    let n = h(e, [`vertexDatasetResource`]);
    t !== void 0 && n != null && m(t, [`supervisedTuningSpec`, `trainingDatasetUri`], n);
  } else if (a === `PREFERENCE_TUNING`) {
    let n = h(e, [`vertexDatasetResource`]);
    t !== void 0 && n != null && m(t, [`preferenceOptimizationSpec`, `trainingDatasetUri`], n);
  } else if (a === `DISTILLATION`) {
    let n = h(e, [`vertexDatasetResource`]);
    t !== void 0 && n != null && m(t, [`distillationSpec`, `promptDatasetUri`], n);
  }
  if (h(e, [`examples`]) !== void 0)
    throw Error(`examples parameter is not supported in Vertex AI.`);
  return r;
}
function uu(e, t) {
  let n = {},
    r = h(e, [`sdkHttpResponse`]);
  r != null && m(n, [`sdkHttpResponse`], r);
  let i = h(e, [`name`]);
  i != null && m(n, [`name`], i);
  let a = h(e, [`state`]);
  a != null && m(n, [`state`], en(a));
  let o = h(e, [`createTime`]);
  o != null && m(n, [`createTime`], o);
  let s = h(e, [`tuningTask`, `startTime`]);
  s != null && m(n, [`startTime`], s);
  let c = h(e, [`tuningTask`, `completeTime`]);
  c != null && m(n, [`endTime`], c);
  let l = h(e, [`updateTime`]);
  l != null && m(n, [`updateTime`], l);
  let u = h(e, [`description`]);
  u != null && m(n, [`description`], u);
  let d = h(e, [`baseModel`]);
  d != null && m(n, [`baseModel`], d);
  let f = h(e, [`_self`]);
  return (f != null && m(n, [`tunedModel`], su(f)), n);
}
function du(e, t) {
  let n = {},
    r = h(e, [`sdkHttpResponse`]);
  r != null && m(n, [`sdkHttpResponse`], r);
  let i = h(e, [`name`]);
  i != null && m(n, [`name`], i);
  let a = h(e, [`state`]);
  a != null && m(n, [`state`], en(a));
  let o = h(e, [`createTime`]);
  o != null && m(n, [`createTime`], o);
  let s = h(e, [`startTime`]);
  s != null && m(n, [`startTime`], s);
  let c = h(e, [`endTime`]);
  c != null && m(n, [`endTime`], c);
  let l = h(e, [`updateTime`]);
  l != null && m(n, [`updateTime`], l);
  let u = h(e, [`error`]);
  u != null && m(n, [`error`], u);
  let d = h(e, [`description`]);
  d != null && m(n, [`description`], d);
  let f = h(e, [`baseModel`]);
  f != null && m(n, [`baseModel`], f);
  let p = h(e, [`tunedModel`]);
  p != null && m(n, [`tunedModel`], p);
  let g = h(e, [`preTunedModel`]);
  g != null && m(n, [`preTunedModel`], g);
  let _ = h(e, [`supervisedTuningSpec`]);
  _ != null && m(n, [`supervisedTuningSpec`], _);
  let v = h(e, [`preferenceOptimizationSpec`]);
  v != null && m(n, [`preferenceOptimizationSpec`], v);
  let y = h(e, [`distillationSpec`]);
  y != null && m(n, [`distillationSpec`], y);
  let b = h(e, [`tuningDataStats`]);
  b != null && m(n, [`tuningDataStats`], b);
  let x = h(e, [`encryptionSpec`]);
  x != null && m(n, [`encryptionSpec`], x);
  let S = h(e, [`partnerModelTuningSpec`]);
  S != null && m(n, [`partnerModelTuningSpec`], S);
  let C = h(e, [`customBaseModel`]);
  C != null && m(n, [`customBaseModel`], C);
  let w = h(e, [`evaluateDatasetRuns`]);
  if (w != null) {
    let e = w;
    (Array.isArray(e) && (e = e.map((e) => e)), m(n, [`evaluateDatasetRuns`], e));
  }
  let T = h(e, [`experiment`]);
  T != null && m(n, [`experiment`], T);
  let E = h(e, [`fullFineTuningSpec`]);
  E != null && m(n, [`fullFineTuningSpec`], E);
  let D = h(e, [`labels`]);
  D != null && m(n, [`labels`], D);
  let O = h(e, [`outputUri`]);
  O != null && m(n, [`outputUri`], O);
  let k = h(e, [`pipelineJob`]);
  k != null && m(n, [`pipelineJob`], k);
  let A = h(e, [`serviceAccount`]);
  A != null && m(n, [`serviceAccount`], A);
  let j = h(e, [`tunedModelDisplayName`]);
  j != null && m(n, [`tunedModelDisplayName`], j);
  let ee = h(e, [`tuningJobState`]);
  ee != null && m(n, [`tuningJobState`], ee);
  let te = h(e, [`veoTuningSpec`]);
  return (te != null && m(n, [`veoTuningSpec`], te), n);
}
function fu(e, t) {
  let n = {},
    r = h(e, [`sdkHttpResponse`]);
  r != null && m(n, [`sdkHttpResponse`], r);
  let i = h(e, [`name`]);
  i != null && m(n, [`name`], i);
  let a = h(e, [`metadata`]);
  a != null && m(n, [`metadata`], a);
  let o = h(e, [`done`]);
  o != null && m(n, [`done`], o);
  let s = h(e, [`error`]);
  return (s != null && m(n, [`error`], s), n);
}
function pu(e, t) {
  let n = {},
    r = h(e, [`gcsUri`]);
  r != null && m(n, [`validationDatasetUri`], r);
  let i = h(e, [`vertexDatasetResource`]);
  return (i != null && m(n, [`validationDatasetUri`], i), n);
}
var mu = class extends f {
    constructor(e) {
      (super(),
        (this.apiClient = e),
        (this.list = async (e = {}) =>
          new U(
            H.PAGED_ITEM_TUNING_JOBS,
            (e) => this.listInternal(e),
            await this.listInternal(e),
            e,
          )),
        (this.get = async (e) => await this.getInternal(e)),
        (this.tune = async (e) => {
          if (this.apiClient.isVertexAI())
            if (e.baseModel.startsWith(`projects/`)) {
              let t = { tunedModelName: e.baseModel };
              e.config?.preTunedModelCheckpointId &&
                (t.checkpointId = e.config.preTunedModelCheckpointId);
              let n = Object.assign(Object.assign({}, e), { preTunedModel: t });
              return ((n.baseModel = void 0), await this.tuneInternal(n));
            } else {
              let t = Object.assign({}, e);
              return await this.tuneInternal(t);
            }
          else {
            let t = Object.assign({}, e),
              n = await this.tuneMldevInternal(t),
              r = ``;
            return (
              n.metadata !== void 0 && n.metadata.tunedModel !== void 0
                ? (r = n.metadata.tunedModel)
                : n.name !== void 0 &&
                  n.name.includes(`/operations/`) &&
                  (r = n.name.split(`/operations/`)[0]),
              { name: r, state: Ee.JOB_STATE_QUEUED }
            );
          }
        }));
    }
    async getInternal(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI()) {
        let i = eu(e);
        return (
          (n = p(`{name}`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `GET`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) =>
              e.json().then((t) => {
                let n = t;
                return ((n.sdkHttpResponse = { headers: e.headers }), n);
              }),
            )),
          t.then((e) => du(e))
        );
      } else {
        let i = $l(e);
        return (
          (n = p(`{name}`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `GET`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) =>
              e.json().then((t) => {
                let n = t;
                return ((n.sdkHttpResponse = { headers: e.headers }), n);
              }),
            )),
          t.then((e) => uu(e))
        );
      }
    }
    async listInternal(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI()) {
        let i = iu(e);
        return (
          (n = p(`tuningJobs`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `GET`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) =>
              e.json().then((t) => {
                let n = t;
                return ((n.sdkHttpResponse = { headers: e.headers }), n);
              }),
            )),
          t.then((e) => {
            let t = ou(e),
              n = new St();
            return (Object.assign(n, t), n);
          })
        );
      } else {
        let i = ru(e);
        return (
          (n = p(`tunedModels`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `GET`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) =>
              e.json().then((t) => {
                let n = t;
                return ((n.sdkHttpResponse = { headers: e.headers }), n);
              }),
            )),
          t.then((e) => {
            let t = au(e),
              n = new St();
            return (Object.assign(n, t), n);
          })
        );
      }
    }
    async cancel(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI()) {
        let i = Kl(e);
        return (
          (n = p(`{name}:cancel`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `POST`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) =>
              e.json().then((t) => {
                let n = t;
                return ((n.sdkHttpResponse = { headers: e.headers }), n);
              }),
            )),
          t.then((e) => {
            let t = Jl(e),
              n = new Ct();
            return (Object.assign(n, t), n);
          })
        );
      } else {
        let i = Gl(e);
        return (
          (n = p(`{name}:cancel`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `POST`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) =>
              e.json().then((t) => {
                let n = t;
                return ((n.sdkHttpResponse = { headers: e.headers }), n);
              }),
            )),
          t.then((e) => {
            let t = ql(e),
              n = new Ct();
            return (Object.assign(n, t), n);
          })
        );
      }
    }
    async tuneInternal(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI()) {
        let i = Ql(e, e);
        return (
          (n = p(`tuningJobs`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `POST`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) =>
              e.json().then((t) => {
                let n = t;
                return ((n.sdkHttpResponse = { headers: e.headers }), n);
              }),
            )),
          t.then((e) => du(e))
        );
      } else throw Error(`This method is only supported by the Vertex AI.`);
    }
    async tuneMldevInternal(e) {
      let t,
        n = ``,
        r = {};
      if (this.apiClient.isVertexAI())
        throw Error(`This method is only supported by the Gemini Developer API.`);
      {
        let i = Zl(e);
        return (
          (n = p(`tunedModels`, i._url)),
          (r = i._query),
          delete i._url,
          delete i._query,
          (t = this.apiClient
            .request({
              path: n,
              queryParams: r,
              body: JSON.stringify(i),
              httpMethod: `POST`,
              httpOptions: e.config?.httpOptions,
              abortSignal: e.config?.abortSignal,
            })
            .then((e) =>
              e.json().then((t) => {
                let n = t;
                return ((n.sdkHttpResponse = { headers: e.headers }), n);
              }),
            )),
          t.then((e) => fu(e))
        );
      }
    }
  },
  hu = class {
    async download(e, t) {
      throw Error(
        `Download to file is not supported in the browser, please use a browser compliant download like an <a> tag.`,
      );
    }
  },
  gu = 1024 * 1024 * 8,
  _u = 3,
  vu = 1e3,
  yu = 2,
  bu = `x-goog-upload-status`;
async function xu(e, t, n, r) {
  let i = await Cu(e, t, n, r),
    a = await i?.json();
  if (i?.headers?.[bu] !== `final`)
    throw Error(`Failed to upload file: Upload status is not finalized.`);
  return a.file;
}
async function Su(e, t, n, r) {
  let i = await Cu(e, t, n, r),
    a = await i?.json();
  if (i?.headers?.[bu] !== `final`)
    throw Error(`Failed to upload file: Upload status is not finalized.`);
  let o = A(a),
    s = new Lt();
  return (Object.assign(s, o), s);
}
async function Cu(e, t, n, r) {
  let i = t,
    a = r?.baseUrl || n.clientOptions.httpOptions?.baseUrl;
  if (a) {
    let e = new URL(a),
      n = new URL(t);
    ((n.protocol = e.protocol), (n.host = e.host), (n.port = e.port), (i = n.toString()));
  }
  let o = 0,
    s = 0,
    c = new lt(new Response()),
    l = `upload`;
  for (o = e.size; s < o; ) {
    let t = Math.min(gu, o - s),
      a = e.slice(s, s + t);
    s + t >= o && (l += `, finalize`);
    let u = 0,
      d = vu;
    for (; u < _u; ) {
      let e = Object.assign(Object.assign({}, r?.headers || {}), {
        "X-Goog-Upload-Command": l,
        "X-Goog-Upload-Offset": String(s),
        "Content-Length": String(t),
      });
      if (
        ((c = await n.request({
          path: ``,
          body: a,
          httpMethod: `POST`,
          httpOptions: Object.assign(Object.assign({}, r), {
            apiVersion: ``,
            baseUrl: i,
            headers: e,
          }),
        })),
        c?.headers?.[bu])
      )
        break;
      (u++, await Tu(d), (d *= yu));
    }
    if (((s += t), c?.headers?.[bu] !== `active`)) break;
    if (o <= s)
      throw Error(`All content has been uploaded, but the upload status is not finalized.`);
  }
  return c;
}
async function wu(e) {
  return { size: e.size, type: e.type };
}
function Tu(e) {
  return new Promise((t) => setTimeout(t, e));
}
var Eu = class {
    async upload(e, t, n, r) {
      if (typeof e == `string`) throw Error(`File path is not supported in browser uploader.`);
      return await xu(e, t, n, r);
    }
    async uploadToFileSearchStore(e, t, n, r) {
      if (typeof e == `string`) throw Error(`File path is not supported in browser uploader.`);
      return await Su(e, t, n, r);
    }
    async stat(e) {
      if (typeof e == `string`) throw Error(`File path is not supported in browser uploader.`);
      return await wu(e);
    }
  },
  Du = class {
    create(e, t, n) {
      return new Ou(e, t, n);
    }
  },
  Ou = class {
    constructor(e, t, n) {
      ((this.url = e), (this.headers = t), (this.callbacks = n));
    }
    connect() {
      ((this.ws = new WebSocket(this.url)),
        (this.ws.onopen = this.callbacks.onopen),
        (this.ws.onerror = this.callbacks.onerror),
        (this.ws.onclose = this.callbacks.onclose),
        (this.ws.onmessage = this.callbacks.onmessage));
    }
    send(e) {
      if (this.ws === void 0) throw Error(`WebSocket is not connected`);
      this.ws.send(e);
    }
    close() {
      if (this.ws === void 0) throw Error(`WebSocket is not connected`);
      this.ws.close();
    }
  },
  ku = `x-goog-api-key`,
  Au = class {
    constructor(e) {
      this.apiKey = e;
    }
    async addAuthHeaders(e, t) {
      if (e.get(ku) === null) {
        if (this.apiKey.startsWith(`auth_tokens/`))
          throw Error(`Ephemeral tokens are only supported by the live API.`);
        if (!this.apiKey) throw Error(`API key is missing. Please provide a valid API key.`);
        e.append(ku, this.apiKey);
      }
    }
  },
  ju = `gl-node/`,
  Mu = class {
    get interactions() {
      if (this._interactions !== void 0) return this._interactions;
      console.warn(
        `GoogleGenAI.interactions: Interactions usage is experimental and may change in future versions.`,
      );
      let e = this.httpOptions;
      return (
        e?.extraBody &&
          console.warn(
            `GoogleGenAI.interactions: Client level httpOptions.extraBody is not supported by the interactions client and will be ignored.`,
          ),
        (this._interactions = new $({
          baseURL: this.apiClient.getBaseUrl(),
          apiKey: this.apiKey,
          apiVersion: this.apiClient.getApiVersion(),
          clientAdapter: this.apiClient,
          defaultHeaders: this.apiClient.getDefaultHeaders(),
          timeout: e?.timeout,
          maxRetries: e?.retryOptions?.attempts,
        }).interactions),
        this._interactions
      );
    }
    constructor(e) {
      if (e.apiKey == null) throw Error(`An API Key must be set when running in a browser`);
      if (e.project || e.location)
        throw Error(
          `Vertex AI project based authentication is not supported on browser runtimes. Please do not provide a project or location.`,
        );
      ((this.vertexai = e.vertexai ?? !1), (this.apiKey = e.apiKey));
      let t = d(e.httpOptions, e.vertexai, void 0, void 0);
      (t && (e.httpOptions ? (e.httpOptions.baseUrl = t) : (e.httpOptions = { baseUrl: t })),
        (this.apiVersion = e.apiVersion),
        (this.httpOptions = e.httpOptions));
      let n = new Au(this.apiKey);
      ((this.apiClient = new Cs({
        auth: n,
        apiVersion: this.apiVersion,
        apiKey: this.apiKey,
        vertexai: this.vertexai,
        httpOptions: this.httpOptions,
        userAgentExtra: ju + `web`,
        uploader: new Eu(),
        downloader: new hu(),
      })),
        (this.models = new Zs(this.apiClient)),
        (this.live = new Bs(this.apiClient, n, new Du())),
        (this.batches = new pr(this.apiClient)),
        (this.chats = new ri(this.models, this.apiClient)),
        (this.caches = new Xr(this.apiClient)),
        (this.files = new gi(this.apiClient)),
        (this.operations = new Qs(this.apiClient)),
        (this.authTokens = new gc(this.apiClient)),
        (this.tunings = new mu(this.apiClient)),
        (this.fileSearchStores = new wc(this.apiClient)));
    }
  };
function Nu(e) {
  return e.thought === !0;
}
function Pu(e, t) {
  return typeof t == `string` && t.length > 0 ? t : e;
}
var Fu = /^[A-Za-z0-9+/]+={0,2}$/,
  Iu = `skip_thought_signature_validator`;
function Lu(e) {
  return !e || e.length % 4 != 0 ? !1 : Fu.test(e);
}
function Ru(e, t) {
  return e && Lu(t) ? t : void 0;
}
function zu(e) {
  return e.startsWith(`claude-`) || e.startsWith(`gpt-oss-`);
}
function Bu(e) {
  let t = e.toLowerCase().match(/^gemini(?:-live)?-(\d+)/);
  if (t) return Number.parseInt(t[1], 10);
}
function Vu(e) {
  let t = Bu(e);
  return t === void 0 ? !0 : t >= 3;
}
function Hu(e, t) {
  let i = [],
    a = r(t.messages, e, (t) => (zu(e.id) ? t.replace(/[^a-zA-Z0-9_-]/g, `_`).slice(0, 64) : t));
  for (let t of a)
    if (t.role === `user`)
      if (typeof t.content == `string`) i.push({ role: `user`, parts: [{ text: n(t.content) }] });
      else {
        let r = t.content.map((e) =>
            e.type === `text`
              ? { text: n(e.text) }
              : { inlineData: { mimeType: e.mimeType, data: e.data } },
          ),
          a = e.input.includes(`image`) ? r : r.filter((e) => e.text !== void 0);
        if (a.length === 0) continue;
        i.push({ role: `user`, parts: a });
      }
    else if (t.role === `assistant`) {
      let r = [],
        a = t.provider === e.provider && t.model === e.id;
      for (let i of t.content)
        if (i.type === `text`) {
          if (!i.text || i.text.trim() === ``) continue;
          let e = Ru(a, i.textSignature);
          r.push({ text: n(i.text), ...(e && { thoughtSignature: e }) });
        } else if (i.type === `thinking`) {
          if (!i.thinking || i.thinking.trim() === ``) continue;
          if (a) {
            let e = Ru(a, i.thinkingSignature);
            r.push({ thought: !0, text: n(i.thinking), ...(e && { thoughtSignature: e }) });
          } else r.push({ text: n(i.thinking) });
        } else if (i.type === `toolCall`) {
          let t = Ru(a, i.thoughtSignature),
            n = e.id.toLowerCase().includes(`gemini-3`),
            o = t || (n ? Iu : void 0),
            s = {
              functionCall: {
                name: i.name,
                args: i.arguments ?? {},
                ...(zu(e.id) ? { id: i.id } : {}),
              },
              ...(o && { thoughtSignature: o }),
            };
          r.push(s);
        }
      if (r.length === 0) continue;
      i.push({ role: `model`, parts: r });
    } else if (t.role === `toolResult`) {
      let r = t.content.filter((e) => e.type === `text`).map((e) => e.text).join(`
`),
        a = e.input.includes(`image`) ? t.content.filter((e) => e.type === `image`) : [],
        o = r.length > 0,
        s = a.length > 0,
        c = Vu(e.id),
        l = o ? n(r) : s ? `(see attached image)` : ``,
        u = a.map((e) => ({ inlineData: { mimeType: e.mimeType, data: e.data } })),
        d = zu(e.id),
        f = {
          functionResponse: {
            name: t.toolName,
            response: t.isError ? { error: l } : { output: l },
            ...(s && c && { parts: u }),
            ...(d ? { id: t.toolCallId } : {}),
          },
        },
        p = i[i.length - 1];
      (p?.role === `user` && p.parts?.some((e) => e.functionResponse)
        ? p.parts.push(f)
        : i.push({ role: `user`, parts: [f] }),
        s && !c && i.push({ role: `user`, parts: [{ text: `Tool result image:` }, ...u] }));
    }
  return i;
}
function Uu(e, t = !1) {
  if (e.length !== 0)
    return [
      {
        functionDeclarations: e.map((e) => ({
          name: e.name,
          description: e.description,
          ...(t ? { parameters: e.parameters } : { parametersJsonSchema: e.parameters }),
        })),
      },
    ];
}
function Wu(e) {
  switch (e) {
    case `auto`:
      return de.AUTO;
    case `none`:
      return de.NONE;
    case `any`:
      return de.ANY;
    default:
      return de.AUTO;
  }
}
function Gu(e) {
  switch (e) {
    case N.STOP:
      return `stop`;
    case N.MAX_TOKENS:
      return `length`;
    case N.BLOCKLIST:
    case N.PROHIBITED_CONTENT:
    case N.SPII:
    case N.SAFETY:
    case N.IMAGE_SAFETY:
    case N.IMAGE_PROHIBITED_CONTENT:
    case N.IMAGE_RECITATION:
    case N.IMAGE_OTHER:
    case N.RECITATION:
    case N.FINISH_REASON_UNSPECIFIED:
    case N.OTHER:
    case N.LANGUAGE:
    case N.MALFORMED_FUNCTION_CALL:
    case N.UNEXPECTED_TOOL_CALL:
    case N.NO_IMAGE:
      return `error`;
    default:
      throw Error(`Unhandled stop reason: ${e}`);
  }
}
function Ku(e) {
  switch (e) {
    case `STOP`:
      return `stop`;
    case `MAX_TOKENS`:
      return `length`;
    default:
      return `error`;
  }
}
export { Ku as a, Mu as c, Gu as i, fe as l, Uu as n, Wu as o, Nu as r, Pu as s, Hu as t };
