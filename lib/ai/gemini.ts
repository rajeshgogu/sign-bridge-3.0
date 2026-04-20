import { GoogleGenerativeAI } from "@google/generative-ai";

const apiKey = process.env.GOOGLE_GENERATIVE_AI_API_KEY;

if (!apiKey) {
  console.warn("GOOGLE_GENERATIVE_AI_API_KEY is missing. AI features will fail.");
}

const genAI = new GoogleGenerativeAI(apiKey || "PLACEHOLDER");

export const geminiModel = genAI.getGenerativeModel({
  model: "gemini-1.5-flash", // Updated to a stable, production-ready model
});

export async function generateContent(prompt: string): Promise<string> {
  if (!apiKey) throw new Error("AI functionality is disabled: API Key missing.");
  const result = await geminiModel.generateContent(prompt);
  return result.response.text();
}
