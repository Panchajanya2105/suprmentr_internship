"""
Verify StockPro dependencies installation
"""

import sys

def check_imports():
    """Check if all required packages can be imported"""
    
    packages = {
        # Core
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'yfinance': 'yfinance',
        
        # Visualization
        'plotly': 'plotly',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        
        # Technical Analysis
        'ta': 'ta',
        
        # Web Framework
        'streamlit': 'streamlit',
        
        # ML/DL
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'tensorflow': 'tensorflow',
        
        # Time Series
        'prophet': 'prophet',
        'statsmodels': 'statsmodels',
        'pmdarima': 'pmdarima',
        
        # Utils
        'requests': 'requests',
        'bs4': 'beautifulsoup4',
        'dotenv': 'python-dotenv',
        'joblib': 'joblib',
        'tqdm': 'tqdm',
        'PIL': 'Pillow',
        
        # Export
        'fpdf': 'fpdf2',
        'openpyxl': 'openpyxl'
    }
    
    missing = []
    installed = []
    
    for import_name, package_name in packages.items():
        try:
            __import__(import_name)
            installed.append(package_name)
            print(f"✅ {package_name} - Successfully imported")
        except ImportError:
            missing.append(package_name)
            print(f"❌ {package_name} - Failed to import")
    
    print("\n" + "="*50)
    print(f"Installed: {len(installed)}/{len(packages)}")
    print(f"Missing: {len(missing)}/{len(packages)}")
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing)}")
    else:
        print("\n🎉 All packages installed successfully!")
    
    return len(missing) == 0

if __name__ == "__main__":
    print("Checking StockPro dependencies...\n")
    success = check_imports()
    sys.exit(0 if success else 1)