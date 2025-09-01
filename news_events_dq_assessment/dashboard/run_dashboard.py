"""
Dashboard Launcher Script
Simple way to start the data quality dashboard
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from dashboard.app import DataQualityDashboard

def main():
    print("🚀 Launching News Events Data Quality Dashboard...")
    print("-" * 50)
    
    # Check if database exists
    db_path = project_root / "news_events_dq.db"
    if not db_path.exists():
        print(f"⚠️  Database not found: {db_path}")
        print("💡 To create database, run:")
        print("   python main.py --mode database-only --data-path data/processed/cleaned_events.json")
        print("")
        print("📊 Dashboard will use mock data for demonstration")
    else:
        print(f"✅ Database found: {db_path}")
        print("📊 Dashboard will show your actual data")
    
    print("")
    print("🌐 Starting dashboard server...")
    print("📱 Access dashboard at: http://127.0.0.1:8050")
    print("🔄 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Start dashboard
    try:
        dashboard = DataQualityDashboard("config/config.yaml")
        dashboard.run_server(debug=True, host="127.0.0.1", port=8050)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Dashboard failed to start: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check if config/config.yaml exists")
        print("2. Ensure all requirements are installed: pip install -r requirements.txt")
        print("3. Check if port 8050 is available")

if __name__ == "__main__":
    main()