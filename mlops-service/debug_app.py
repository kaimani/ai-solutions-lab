#!/usr/bin/env python3
import sys
import traceback

print("🔍 Starting debug...")

try:
    print("1. Attempting to import app...")
    from app import app
    print("✅ Successfully imported app")
    
    print("2. Checking if app is a Flask instance...")
    from flask import Flask
    if isinstance(app, Flask):
        print("✅ App is a valid Flask instance")
    else:
        print("❌ App is not a Flask instance")
    
    print("3. Starting server...")
    app.run(host='0.0.0.0', port=5001, debug=True)
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    print("Full traceback:")
    traceback.print_exc()
    sys.exit(1)