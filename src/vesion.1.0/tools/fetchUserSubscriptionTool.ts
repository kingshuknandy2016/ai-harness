import { tool } from "langchain";
import { z } from "zod";

// Tool 1: database lookup (simulated)
const fetchUserSubscriptionTool = tool(
  async ({ userId }: { userId: string }) => {
    console.log(
      `\n[Harness Executing Tool]: Fetching DB info for user ${userId}...`,
    );

    const mockDb: Record<string, { plan: string; status: string }> = {
      user_123: { plan: "Pro Tier", status: "active" },
      user_456: { plan: "Free Tier", status: "canceled" },
    };

    const user = mockDb[userId] ?? { plan: "Unknown", status: "not_found" };
    return JSON.stringify(user);
  },
  {
    name: "fetch_user_subscription",
    description:
      "Fetches user subscription details from the internal database given a user ID.",
    schema: z.object({
      userId: z.string().describe("The user ID to fetch details for"),
    }),
  },
);

export default fetchUserSubscriptionTool;
