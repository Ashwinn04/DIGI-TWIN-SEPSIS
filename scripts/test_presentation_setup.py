#!/usr/bin/env python3
"""
Quick test script to verify presentation setup
Run this before your presentation to ensure everything works
"""
import sys
import os

def test_imports():
    """Test that all required packages are installed"""
    print("🔍 Testing imports...")
    try:
        import streamlit
        print("  ✅ Streamlit")
    except ImportError:
        print("  ❌ Streamlit - Run: pip install streamlit")
        return False
    
    try:
        import pandas as pd
        print("  ✅ Pandas")
    except ImportError:
        print("  ❌ Pandas - Run: pip install pandas")
        return False
    
    try:
        import numpy as np
        print("  ✅ NumPy")
    except ImportError:
        print("  ❌ NumPy - Run: pip install numpy")
        return False
    
    try:
        import plotly
        print("  ✅ Plotly")
    except ImportError:
        print("  ❌ Plotly - Run: pip install plotly")
        return False
    
    try:
        import torch
        print("  ✅ PyTorch")
    except ImportError:
        print("  ⚠️  PyTorch (optional - needed for deep learning models)")
    
    return True

def test_files():
    """Test that required files exist"""
    print("\n🔍 Testing files...")
    
    files_to_check = [
        ('dashboard_real.py', 'Main dashboard file'),
        ('integration_real.py', 'Integration system'),
        ('README.md', 'Documentation'),
    ]
    
    all_exist = True
    for file, desc in files_to_check:
        if os.path.exists(file):
            print(f"  ✅ {file} - {desc}")
        else:
            print(f"  ❌ {file} - {desc} - MISSING!")
            all_exist = False
    
    # Check for data files
    data_files = ['Dataset.csv', 'Capstone/Dataset.csv', 'fully_cleaned_sepsis_data.csv']
    data_found = False
    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"  ✅ {data_file} - Data file found")
            data_found = True
            break
    
    if not data_found:
        print("  ⚠️  No data file found (Dataset.csv or fully_cleaned_sepsis_data.csv)")
        print("     Dashboard will work but may have limited functionality")
    
    return all_exist

def test_integration_system():
    """Test that integration system can be imported"""
    print("\n🔍 Testing integration system...")
    try:
        sys.path.insert(0, os.getcwd())
        from integration_real import get_integration_system, check_baseline_models_status
        
        print("  ✅ Integration system imports successfully")
        
        # Check model status
        status = check_baseline_models_status()
        print(f"\n  📊 Model Status:")
        print(f"     Initialized: {status.get('initialized', False)}")
        print(f"     Baseline models: {status.get('baseline_models_loaded', 0)}")
        print(f"     Deep learning models: {status.get('deep_learning_models_loaded', 0)}")
        
        if status.get('baseline_models'):
            print(f"     Baseline models: {', '.join(status['baseline_models'])}")
        if status.get('deep_learning_models'):
            print(f"     DL models: {', '.join(status['deep_learning_models'])}")
        
        return True
    except Exception as e:
        print(f"  ❌ Integration system error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dashboard_import():
    """Test that dashboard can be imported (without running)"""
    print("\n🔍 Testing dashboard import...")
    try:
        # Just check if file is syntactically correct
        with open('dashboard_real.py', 'r') as f:
            code = f.read()
        compile(code, 'dashboard_real.py', 'exec')
        print("  ✅ Dashboard file is valid Python")
        return True
    except SyntaxError as e:
        print(f"  ❌ Dashboard has syntax errors: {e}")
        return False
    except Exception as e:
        print(f"  ⚠️  Could not verify dashboard: {e}")
        return True  # Don't fail on this

def main():
    """Run all tests"""
    print("=" * 60)
    print("🎤 PRESENTATION SETUP TEST")
    print("=" * 60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Files", test_files()))
    results.append(("Integration System", test_integration_system()))
    results.append(("Dashboard", test_dashboard_import()))
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Ready for presentation!")
        print("\n🚀 To start dashboard:")
        print("   streamlit run dashboard_real.py")
    else:
        print("⚠️  SOME TESTS FAILED - Please fix issues above")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())


