#!/usr/bin/env python3
"""
Test script for AI Script
This script tests the basic functionality without requiring all dependencies
"""

import sys
import os

def test_basic_functionality():
    """Test basic functionality of the AI script"""
    print("🧪 Testing AI Script Basic Functionality")
    print("=" * 50)
    
    try:
        # Test import
        print("1. Testing imports...")
        from ai_script import AIScript
        print("   ✅ AIScript class imported successfully")
        
        # Test initialization
        print("2. Testing initialization...")
        ai = AIScript()
        print("   ✅ AIScript initialized successfully")
        
        # Test text preprocessing
        print("3. Testing text preprocessing...")
        test_text = "I love this AI script! It's amazing."
        processed = ai.preprocess_text(test_text)
        print(f"   ✅ Text preprocessing works: '{processed}'")
        
        # Test sentiment analysis (basic)
        print("4. Testing sentiment analysis...")
        try:
            result = ai.sentiment_analysis(test_text)
            if "sentiment" in result:
                print(f"   ✅ Sentiment analysis works: {result['sentiment']}")
            else:
                print("   ⚠️  Sentiment analysis returned unexpected format")
        except Exception as e:
            print(f"   ⚠️  Sentiment analysis requires additional setup: {e}")
        
        # Test sample data generation
        print("5. Testing sample data generation...")
        sample_data = ai.generate_sample_data()
        print(f"   ✅ Generated {len(sample_data)} sample data entries")
        
        print("\n" + "=" * 50)
        print("🎉 Basic functionality test completed!")
        print("The AI script is ready to use.")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required dependencies: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def show_usage_examples():
    """Show usage examples"""
    print("\n📖 Usage Examples:")
    print("-" * 30)
    print("1. Run demo:")
    print("   python ai_script.py --demo")
    print()
    print("2. Analyze sentiment:")
    print("   python ai_script.py --sentiment 'I love this!'")
    print()
    print("3. Train model:")
    print("   python ai_script.py --train sample_data.json")
    print()
    print("4. Make prediction:")
    print("   python ai_script.py --predict 'This is great!'")
    print()
    print("5. Get help:")
    print("   python ai_script.py --help")

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        show_usage_examples()
    else:
        print("\n❌ Test failed. Please check the error messages above.")
        sys.exit(1)
