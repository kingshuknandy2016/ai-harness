// ============================================================================
// 3. EVALUATION / GUARDRAIL LAYER
// ============================================================================

import { HumanMessage } from "langchain";
import { evaluateTurn } from "./guardrail";
import { harnessAgent } from "./harnessAgent";

export async function  runHarnessTask(threadId: string, userInput: string) {
  console.log("\n======================================================");
  console.log(`[USER INPUT]: "${userInput}"`);
  console.log("======================================================");

  const config = { configurable: { thread_id: threadId } };

  // Step 1: run the agent loop
  const result = await harnessAgent.invoke(
    { messages: [new HumanMessage(userInput)] },
    config,
  );

  const finalMessage = result.messages[result.messages.length - 1];
  const finalContent = finalMessage.content.toString();

  // Step 2: post-execution safety check (demo only; use a real
  // classifier or policy engine in production)
  const passesGuardrail = !finalContent
    .toLowerCase()
    .includes("unauthorized_access_code");

  if (!passesGuardrail) {
    console.error(
      "[Harness Safety Evaluation]: BLOCKED response due to policy violation.",
    );
    return "Error: Output blocked by safety evaluation guardrail.";
  }
  // Step 2: post-execution safety check (policy engine + moderation classifier)
  //   const verdict = await evaluateTurn(result.messages, finalContent);

  //   if (!verdict.allowed) {
  //     console.error(
  //       `[Harness Safety Evaluation]: BLOCKED response (${verdict.reason}).`,
  //     );
  //     return "Error: Output blocked by safety evaluation guardrail.";
  //   }

  console.log(`\n[FINAL HARNESS OUTPUT]:\n${finalContent}`);
  return finalContent;
}
