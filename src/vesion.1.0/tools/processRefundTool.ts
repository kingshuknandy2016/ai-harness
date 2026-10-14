import { tool } from "langchain";
import { z } from "zod";

const processRefundTool = tool(
  async ({ userId, amount }: { userId: string; amount: number }) => {
    console.log(
      `\n[Harness Executing Tool]: Processing $${amount} refund for ${userId}...`,
    );

    return JSON.stringify({ status: "success", refundId: `ref_${Date.now()}` });
  },
  {
    name: "process_refund",
    description: "Processes a payment refund for a specific user ID.",
    schema: z.object({
      userId: z.string().describe("Target user ID"),
      amount: z.number().positive().describe("Amount to refund in USD"),
    }),
  },
);

export default processRefundTool;
