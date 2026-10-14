// ============================================================================
// 4. EXECUTION DEMO
// ============================================================================

import "dotenv/config"; // must load before the OpenAI client is created
import { runHarnessTask } from "./harness/runHarnessTask";

async function main() {
  const sessionThreadId = "session_user_456_abc";

  // Turn 1: triggers both tools
  await runHarnessTask(
    sessionThreadId,
    "Hi, my user ID is user_456. Can you please refund $50 for my last invoice?",
  );

  // Turn 2: answered from memory in the same thread
  await runHarnessTask(
    sessionThreadId,
    "What was the refund status and ID you just generated?",
  );
}

main().catch(console.error);
