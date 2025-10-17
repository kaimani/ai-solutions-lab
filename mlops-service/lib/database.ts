// Add this interface to your existing database.ts file
export interface AIMetrics {
  id: string
  business_id: string
  conversation_id?: string
  session_id: string
  
  // Performance metrics
  response_time_ms: number
  success_rate: number
  
  // AI performance metrics
  tokens_used: number
  prompt_tokens?: number
  completion_tokens?: number
  api_cost_usd: number
  model_name: string
  
  // Business metrics
  intent_detected: string
  appointment_requested: boolean
  human_handoff_requested: boolean
  appointment_booked?: boolean
  
  // Message metrics
  user_message_length: number
  ai_response_length: number
  response_type: string
  
  created_at: string
}

// Add these functions to the end of your database.ts file
export async function createAIMetrics(metrics: Omit<AIMetrics, "id" | "created_at">) {
  const result = await sql`
    INSERT INTO ai_metrics (
      business_id, conversation_id, session_id,
      response_time_ms, success_rate,
      tokens_used, prompt_tokens, completion_tokens, api_cost_usd, model_name,
      intent_detected, appointment_requested, human_handoff_requested, appointment_booked,
      user_message_length, ai_response_length, response_type
    )
    VALUES (
      ${metrics.business_id}, ${metrics.conversation_id}, ${metrics.session_id},
      ${metrics.response_time_ms}, ${metrics.success_rate},
      ${metrics.tokens_used}, ${metrics.prompt_tokens}, ${metrics.completion_tokens}, ${metrics.api_cost_usd}, ${metrics.model_name},
      ${metrics.intent_detected}, ${metrics.appointment_requested}, ${metrics.human_handoff_requested}, ${metrics.appointment_booked || false},
      ${metrics.user_message_length}, ${metrics.ai_response_length}, ${metrics.response_type}
    )
    RETURNING *
  `
  return result[0] as AIMetrics
}