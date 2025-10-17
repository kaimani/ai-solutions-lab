/**
 * MLOps Tracking Utilities
 * Functions to calculate and send AI performance metrics
 */

// Types for our metrics data
export interface MetricsData {
  business_id: string;
  conversation_id?: string | undefined;
  session_id: string;
  response_time_ms: number;
  success_rate: number;
  user_satisfaction?: number;
  tokens_used: number;
  prompt_tokens?: number;
  completion_tokens?: number;
  api_cost_usd: number;
  model_name: string;
  intent_detected: string;
  appointment_requested: boolean;
  human_handoff_requested: boolean;
  appointment_booked?: boolean;
  user_message_length: number;
  ai_response_length: number;
  response_type: string;
}

/**
 * Track metrics by sending to MLOps service
 * This runs in the background and won't slow down user responses
 */
export async function trackMetrics(metricsData: MetricsData): Promise<void> {
  try {
    // Send metrics to Flask MLOps service (for Prometheus monitoring)
    const mlopsServiceUrl = process.env.MLOPS_SERVICE_URL || 'http://localhost:5001';
    
    const response = await fetch(`${mlopsServiceUrl}/track`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(metricsData),
    });

    if (!response.ok) {
      throw new Error(`MLOps service responded with status: ${response.status}`);
    }

    const result = await response.json();
    console.log('Metrics tracked successfully:', result.status);
  } catch (error) {
    // Log error but don't throw - we don't want metrics to break user experience
    console.error("Error tracking metrics:", error);
  }
}

/**
 * Calculate how many tokens our AI request used
 * This is an estimation based on text length
 */
export function calculateTokenUsage(
  systemPrompt: string,
  userMessage: string,
  aiResponse: string
) {
  // Rough estimation: 1 token ≈ 4 characters for English text
  const CHARS_PER_TOKEN = 4;

  const systemTokens = Math.ceil(systemPrompt.length / CHARS_PER_TOKEN);
  const userTokens = Math.ceil(userMessage.length / CHARS_PER_TOKEN);
  const responseTokens = Math.ceil(aiResponse.length / CHARS_PER_TOKEN);

  const totalTokens = systemTokens + userTokens + responseTokens;

  return {
    totalTokens,
    systemTokens,
    userTokens,
    responseTokens,
  };
}

/**
 * Estimate how much this AI request cost us
 * Based on Gemini API pricing
 */
export function estimateApiCost(totalTokens: number): number {
  // Gemini 1.5 Flash pricing: roughly $0.1875 per 1M tokens
  const COST_PER_MILLION_TOKENS = 0.1875;

  const costUsd = (totalTokens / 1_000_000) * COST_PER_MILLION_TOKENS;

  // Round to 6 decimal places
  return Math.round(costUsd * 1_000_000) / 1_000_000;
}