import type { BaseMessage } from "@langchain/core/messages";

/**
 * GUARDRAIL
 * =========
 * Runs AFTER the agent finishes a turn and decides whether its reply may be
 * returned to the user. It is independent of the LLM, so it still protects us
 * if the model ignores its system prompt.
 *
 *   agent turn ──► [1. policy rules] ──► [2. moderation classifier] ──► allow
 *                        │ violation            │ flagged / unavailable
 *                        └──────────────► block ◄┘
 *`
 * Layer 1 – Policy rules (deterministic, free, instant)
 *   Plain functions that inspect the turn's tool calls and the final reply.
 *   Use these for business rules ("refund cap", "must verify subscription")
 *   and exact-match leaks (API keys, card numbers).
 *
 * Layer 2 – Moderation classifier (ML, one HTTP call)
 *   OpenAI's moderation model scores the reply for harmful content (hate,
 *   violence, self-harm, ...). Use this for fuzzy content that rules can't
 *   express. It FAILS CLOSED: if the call errors or times out, we block.
 *
 * Rules run first; the classifier is only called if every rule passes.
 *
 * To add a policy: write a `Rule` and append it to `RULES`.
 */

/** Result of evaluating a turn. `reason` is only set when blocked. */
export type Verdict = { allowed: true } | { allowed: false; reason: string };

/** Largest refund (USD) the agent may issue without a human. */
const MAX_REFUND_USD = 100;

// ---------------------------------------------------------------------------
// Layer 1: policy rules
// ---------------------------------------------------------------------------

type ToolCall = { name: string; args: Record<string, any> };

/** What a rule gets to look at: this turn's tool activity and the reply. */
interface TurnContext {
  toolCalls: ToolCall[];
  /** Parsed JSON results of `fetch_user_subscription` calls this turn. */
  subscriptions: { status?: string }[];
  reply: string;
}

/** Returns a human-readable violation, or `null` if the turn is fine. */
type Rule = (ctx: TurnContext) => string | null;

const RULES: Rule[] = [
  // A refund requires an active subscription that was actually looked up.
  ({ toolCalls, subscriptions }) => {
    const refunded = toolCalls.some((c) => c.name === "process_refund");
    const active = subscriptions.some((s) => s.status === "active");
    return refunded && !active
      ? "refund issued without a verified active subscription"
      : null;
  },

  // Refunds above the cap need a human.
  ({ toolCalls }) => {
    const big = toolCalls.find(
      (c) => c.name === "process_refund" && c.args.amount > MAX_REFUND_USD,
    );
    return big
      ? `refund of $${big.args.amount} exceeds the $${MAX_REFUND_USD} limit`
      : null;
  },

  // The reply must not leak secrets.
  ({ reply }) => {
    const leaks: [RegExp, string][] = [
      [/\b(?:sk|pk)[-_][A-Za-z0-9]{16,}\b/, "an API key"],
      [/\b(?:\d[ -]?){13,16}\b/, "a card number"],
      [/-----BEGIN [A-Z ]*PRIVATE KEY-----/, "a private key"],
    ];
    const hit = leaks.find(([re]) => re.test(reply));
    return hit ? `reply contains ${hit[1]}` : null;
  },
];

/** Extract only the current turn (after the last human message) into a context. */
function buildContext(messages: BaseMessage[], reply: string): TurnContext {
  const lastHuman = messages.map((m) => m.type).lastIndexOf("human");
  const turn = messages.slice(lastHuman + 1);

  const toolCalls = turn.flatMap(
    (m) => ((m as { tool_calls?: ToolCall[] }).tool_calls ?? []),
  );

  const subscriptions = turn
    .filter((m) => m.type === "tool" && (m as { name?: string }).name === "fetch_user_subscription")
    .map((m) => {
      try {
        return JSON.parse(String(m.content));
      } catch {
        return {}; // unparseable result counts as "not verified"
      }
    });

  return { toolCalls, subscriptions, reply };
}

// ---------------------------------------------------------------------------
// Layer 2: moderation classifier
// ---------------------------------------------------------------------------

/** Returns the flagged categories as a string, or `null` if the text is clean. */
async function classify(text: string): Promise<string | null> {
  const res = await fetch("https://api.openai.com/v1/moderations", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
    },
    body: JSON.stringify({ model: "omni-moderation-latest", input: text }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);

  const { results } = (await res.json()) as {
    results: { flagged: boolean; categories: Record<string, boolean> }[];
  };
  const { flagged, categories } = results[0];
  return flagged
    ? Object.keys(categories).filter((k) => categories[k]).join(", ")
    : null;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Decide whether the agent's reply may be sent to the user.
 *
 * @param messages Full message history returned by the agent (the current
 *                 turn is picked out automatically).
 * @param reply    The agent's final text reply.
 */
export async function evaluateTurn(
  messages: BaseMessage[],
  reply: string,
): Promise<Verdict> {
  const ctx = buildContext(messages, reply);

  for (const rule of RULES) {
    const violation = rule(ctx);
    if (violation) return { allowed: false, reason: `policy: ${violation}` };
  }

  try {
    const categories = await classify(reply);
    if (categories) {
      return { allowed: false, reason: `classifier flagged: ${categories}` };
    }
  } catch (err) {
    return {
      allowed: false,
      reason: `classifier unavailable: ${(err as Error).message}`,
    };
  }

  return { allowed: true };
}
