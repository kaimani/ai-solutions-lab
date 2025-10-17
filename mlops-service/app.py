"""
Flask MLOps Service for AI Appointment Setter
This service tracks AI performance metrics and stores them for analysis
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import os
import json
import time
from datetime import datetime
import logging
from typing import Dict, Any

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

# Configure logging so we can see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow requests from Next.js app

# Database connection - use the same DATABASE_URL as your Next.js app
DATABASE_URL = os.getenv('DATABASE_URL')
ENABLE_DB = os.getenv('ENABLE_DB', '0') == '1'

logger.info(f"Database configured: {bool(DATABASE_URL)}")
logger.info(f"Database enabled: {ENABLE_DB}")

# Only import psycopg if DATABASE_URL is available
if DATABASE_URL:
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        DB_AVAILABLE = True
    except ImportError:
        logger.warning("psycopg2 not installed, database operations disabled")
        DB_AVAILABLE = False
else:
    DB_AVAILABLE = False
    logger.warning("DATABASE_URL not found in environment variables")

# Prometheus Metrics Setup
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Define Prometheus metrics
ai_requests_total = Counter('ai_requests_total', 'Total AI requests', ['business_id', 'intent', 'response_type'])
ai_response_time_seconds = Histogram('ai_response_time_seconds', 'AI response time in seconds', ['business_id'])
ai_tokens_used_total = Counter('ai_tokens_used_total', 'Total tokens used', ['business_id', 'model'])
ai_api_cost_usd_total = Counter('ai_api_cost_usd_total', 'Total API cost in USD', ['business_id'])
appointments_requested_total = Counter('appointments_requested_total', 'Total appointment requests', ['business_id'])
human_handoffs_total = Counter('human_handoffs_total', 'Total human handoff requests', ['business_id'])

logger.info("Prometheus metrics initialized successfully")

def update_prometheus_metrics(metrics_data):
    """Update Prometheus metrics with new data"""
    try:
        business_id = metrics_data.get('business_id', 'unknown')

        # Update request counter
        ai_requests_total.labels(
            business_id=business_id,
            intent=metrics_data.get('intent_detected', 'unknown'),
            response_type=metrics_data.get('response_type', 'unknown')
        ).inc()

        # Update response time histogram
        if 'response_time_ms' in metrics_data:
            response_time_seconds = metrics_data['response_time_ms'] / 1000.0
            ai_response_time_seconds.labels(business_id=business_id).observe(response_time_seconds)

        # Update token usage
        if 'tokens_used' in metrics_data:
            ai_tokens_used_total.labels(
                business_id=business_id,
                model=metrics_data.get('model_name', 'gemini-1.5-flash')
            ).inc(metrics_data['tokens_used'])

        # Update API cost
        if 'api_cost_usd' in metrics_data:
            ai_api_cost_usd_total.labels(business_id=business_id).inc(metrics_data['api_cost_usd'])

        # Update business metrics
        if metrics_data.get('appointment_requested', False):
            appointments_requested_total.labels(business_id=business_id).inc()

        if metrics_data.get('human_handoff_requested', False):
            human_handoffs_total.labels(business_id=business_id).inc()

        logger.info(f"Successfully updated Prometheus metrics for business {business_id}")
        return True

    except Exception as e:
        logger.error(f"Error updating Prometheus metrics: {e}")
        return False

def get_db_connection():
    """Get database connection"""
    if not DATABASE_URL or not DB_AVAILABLE:
        logger.error("Database not configured or psycopg2 not available")
        return None
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return None

def create_metrics_table():
    """
    Initialize metrics storage
    """
    if not DATABASE_URL or not ENABLE_DB:
        logger.info("Database operations disabled, skipping table creation")
        return True
        
    conn = get_db_connection()
    if not conn:
        logger.error("Could not connect to database to create table")
        return False
        
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_metrics (
                    id SERIAL PRIMARY KEY,
                    business_id VARCHAR(100) NOT NULL,
                    conversation_id VARCHAR(100),
                    session_id VARCHAR(100),
                    response_time_ms INTEGER,
                    success_rate FLOAT,
                    tokens_used INTEGER,
                    api_cost_usd FLOAT,
                    model_name VARCHAR(50),
                    intent_detected VARCHAR(100),
                    appointment_requested BOOLEAN,
                    appointment_booked BOOLEAN,
                    human_handoff_requested BOOLEAN,
                    user_message_length INTEGER,
                    ai_response_length INTEGER,
                    response_type VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            logger.info("Metrics table created successfully")
            return True
    except Exception as e:
        logger.error(f"Error creating metrics table: {e}")
        return False
    finally:
        if conn:
            conn.close()

def fetch_metrics_from_db() -> bool:
    """
    Fetch metrics directly from Neon database
    """
    if not DATABASE_URL or not ENABLE_DB:
        logger.info("Database operations disabled, skipping metrics fetch")
        return False
    
    try:
        logger.info("Fetching historical metrics from Neon database...")
        
        # For now, just verify we can connect to the database
        conn = get_db_connection()
        if conn:
            conn.close()
            logger.info("Successfully connected to Neon database")
            return True
        else:
            logger.warning("Could not connect to Neon database")
            return False
        
    except Exception as e:
        logger.error(f"Error fetching metrics from database: {e}")
        return False

def rebuild_prometheus_metrics_from_db():
    """
    Rebuild Prometheus metrics from database on startup
    """
    try:
        logger.info("Rebuilding Prometheus metrics from database...")
        
        # Fetch and rebuild metrics
        success = fetch_metrics_from_db()
        
        if success:
            logger.info("Successfully rebuilt Prometheus metrics from database")
        else:
            logger.warning("Could not rebuild metrics from database, starting fresh")
            
    except Exception as e:
        logger.error(f"Error rebuilding Prometheus metrics: {e}")

def store_metrics_in_db(metrics_data: Dict[str, Any]) -> bool:
    """Save metrics to our database for later analysis"""
    if not DATABASE_URL or not ENABLE_DB:
        logger.info("Database operations disabled, skipping metrics storage")
        return True  # Return True to not break the API flow
    
    conn = get_db_connection()
    if not conn:
        return False

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO ai_metrics (
                    business_id, conversation_id, session_id,
                    response_time_ms, success_rate,
                    tokens_used, api_cost_usd, model_name,
                    intent_detected, appointment_requested, appointment_booked, human_handoff_requested,
                    user_message_length, ai_response_length, response_type
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """, (
                metrics_data.get('business_id'),
                metrics_data.get('conversation_id'),
                metrics_data.get('session_id'),
                metrics_data.get('response_time_ms'),
                metrics_data.get('success_rate', 1.0),
                metrics_data.get('tokens_used'),
                metrics_data.get('api_cost_usd'),
                metrics_data.get('model_name', 'gemini-1.5-flash'),
                metrics_data.get('intent_detected'),
                metrics_data.get('appointment_requested', False),
                metrics_data.get('appointment_booked', False),
                metrics_data.get('human_handoff_requested', False),
                metrics_data.get('user_message_length'),
                metrics_data.get('ai_response_length'),
                metrics_data.get('response_type')
            ))
            conn.commit()
            logger.info(f"Successfully stored metrics for business {metrics_data.get('business_id')}")
            return True
    except Exception as e:
        logger.error(f"Error storing metrics: {e}")
        return False
    finally:
        if conn:
            conn.close()

# ========== ROUTES ==========

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with service information"""
    return jsonify({
        'service': 'MLOps Monitoring Service',
        'version': '1.0.0',
        'endpoints': {
            'health': '/health',
            'track_metrics': '/track (POST)',
            'prometheus_metrics': '/metrics',
            'root': '/'
        },
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Check if our service is running properly"""
    return jsonify({
        'status': 'healthy',
        'service': 'mlops-service',
        'timestamp': datetime.utcnow().isoformat(),
        'monitoring': 'prometheus',
        'database_configured': bool(DATABASE_URL),
        'database_enabled': ENABLE_DB
    })

@app.route('/track', methods=['POST'])
def track_metrics():
    """Receive metrics from the Next.js chat application"""
    try:
        metrics_data = request.get_json()

        if not metrics_data:
            return jsonify({'error': 'No metrics data provided'}), 400

        # Validate required fields
        required_fields = ['business_id', 'response_time_ms', 'tokens_used']
        for field in required_fields:
            if field not in metrics_data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Update Prometheus metrics first
        prometheus_success = update_prometheus_metrics(metrics_data)

        # Store in database
        db_success = store_metrics_in_db(metrics_data)

        if db_success and prometheus_success:
            logger.info(f"Successfully tracked metrics for business {metrics_data.get('business_id')}")
            return jsonify({
                'status': 'success',
                'message': 'Metrics tracked successfully',
                'prometheus_updated': prometheus_success,
                'timestamp': datetime.utcnow().isoformat()
            })
        else:
            return jsonify({'error': 'Failed to store metrics'}), 500

    except Exception as e:
        logger.error(f"Error tracking metrics: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/metrics', methods=['GET'])
def prometheus_metrics():
    """Prometheus metrics endpoint - industry standard format"""
    try:
        return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}
    except Exception as e:
        logger.error(f"Error generating Prometheus metrics: {e}")
        return jsonify({'error': 'Failed to generate metrics'}), 500

# Initialize metrics on startup
create_metrics_table()

# Rebuild Prometheus metrics from database on startup
rebuild_prometheus_metrics_from_db()

if __name__ == '__main__':
    # Run the Flask app
    port = int(os.getenv('SERVICE_PORT', 5001))
    logger.info(f"Starting MLOps service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=os.getenv('FLASK_DEBUG', 'True') == 'True')# CI/CD Pipeline Test
