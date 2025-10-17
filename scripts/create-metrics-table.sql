-- Create AI Metrics table for MLOps tracking
-- Lab 2: AI Lifecycle & MLOps Integration
-- Run this in your Neon database console

CREATE TABLE IF NOT EXISTS ai_metrics (
    id SERIAL PRIMARY KEY,
    business_id VARCHAR(255) NOT NULL,
    conversation_id VARCHAR(255),
    session_id VARCHAR(255) NOT NULL,
    
    -- Performance metrics
    response_time_ms INTEGER NOT NULL,
    success_rate DECIMAL(5,4) NOT NULL,
    
    -- AI performance metrics
    tokens_used INTEGER NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    api_cost_usd DECIMAL(10,6) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    
    -- Business metrics
    intent_detected VARCHAR(50) NOT NULL,
    appointment_requested BOOLEAN DEFAULT FALSE,
    human_handoff_requested BOOLEAN DEFAULT FALSE,
    appointment_booked BOOLEAN DEFAULT FALSE,
    
    -- Message metrics
    user_message_length INTEGER NOT NULL,
    ai_response_length INTEGER NOT NULL,
    response_type VARCHAR(50) NOT NULL,
    
    -- Timestamp
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_ai_metrics_business_id ON ai_metrics(business_id);
CREATE INDEX IF NOT EXISTS idx_ai_metrics_created_at ON ai_metrics(created_at);
CREATE INDEX IF NOT EXISTS idx_ai_metrics_conversation_id ON ai_metrics(conversation_id);
CREATE INDEX IF NOT EXISTS idx_ai_metrics_business_created ON ai_metrics(business_id, created_at);

-- Verify table was created
SELECT 'ai_metrics table created successfully' as status;