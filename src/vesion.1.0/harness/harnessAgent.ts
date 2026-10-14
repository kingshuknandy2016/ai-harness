import { MemorySaver } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { createAgent } from "langchain";
import { fetchUserSubscriptionTool, processRefundTool } from "../tools";

// ============================================================================
// 2. HARNESS SETUP
// ============================================================================

// A. Brain: the LLM
export const model = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });

// B. Persistence: conversation state per thread
const checkpointer = new MemorySaver();

// C. System instructions / guardrails
const systemPrompt = `
You are a Customer Support AI Agent operating inside a secure enterprise harness.

RULES:
1. Always verify the user's subscription status with 'fetch_user_subscription'
   BEFORE performing any action.
2. Only issue refunds if the user's subscription status is 'active'.
3. Be professional and concise.
`;

// D. The agent itself
export const harnessAgent = createAgent({
  model,
  tools: [fetchUserSubscriptionTool, processRefundTool],
  checkpointer,
  systemPrompt,
});
