#!/usr/bin/env python3
"""
Setup script for AI Script
Run this script to set up the environment and test the installation
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 7:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("Please use Python 3.7 or higher")
        return False

def install_dependencies():
    """Install required dependencies"""
    return run_command("pip install -r requirements.txt", "Installing dependencies")

def test_imports():
    """Test if all required modules can be imported"""
    print("\n🧪 Testing imports...")
    try:
        import numpy
        import pandas
        import sklearn
        import nltk
        import textblob
        import matplotlib
        import seaborn
        import joblib
        print("✅ All required modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def run_demo():
    """Run the AI script demo"""
    return run_command("python ai_script.py --demo", "Running AI script demo")

def main():
    """Main setup function"""
    print("🚀 AI Script Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed at dependency installation")
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        print("\n❌ Setup failed at import testing")
        sys.exit(1)
    
    # Run demo
    if not run_demo():
        print("\n❌ Setup failed at demo execution")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nYou can now use the AI script with the following commands:")
    print("  python ai_script.py --demo                    # Run full demo")
    print("  python ai_script.py --sentiment 'Your text'   # Analyze sentiment")
    print("  python ai_script.py --train sample_data.json  # Train model")
    print("  python ai_script.py --predict 'Your text'     # Make prediction")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()
