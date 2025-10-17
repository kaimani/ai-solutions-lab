#!/bin/bash

echo "🚀 Starting MLOps Service for AI Appointment Setter"
echo "=================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Create metrics directory for Prometheus
mkdir -p metrics

echo "✅ Setup complete!"
echo "🌐 Service will be available at: http://localhost:5001"
echo "Starting Flask application..."

# Start Flask application
python app.py