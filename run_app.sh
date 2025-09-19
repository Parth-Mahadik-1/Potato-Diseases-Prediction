#!/bin/bash

# Potato Disease Detection Flask App - Run Script
echo "🥔 Starting Potato Disease Detection App..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3 and try again."
    exit 1
fi

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if model file exists
if [ ! -f "potato_disease_model.h5" ]; then
    echo "⚠  Model file 'potato_disease_model.h5' not found!"
    echo ""
    echo "Please do the following:"
    echo "1. Run your Jupyter notebook to train the model"
    echo "2. Add this code at the end of your notebook:"
    echo "   model.save('potato_disease_model.h5')"
    echo "3. Copy the generated .h5 file to this directory"
    echo "4. Run this script again"
    echo ""
    exit 1
fi

# Create uploads directory
mkdir -p uploads

# Test model loading
echo "🧪 Testing model loading..."
python model_setup.py

# Start Flask app
echo "🚀 Starting Flask application..."
echo "Access the app at: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

python app.py