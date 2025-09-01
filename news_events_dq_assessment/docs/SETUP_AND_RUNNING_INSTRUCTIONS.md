# News Events Data Quality Assessment - Setup and Running Instructions

## Prerequisites

- Python 3.8 or higher
- Git
- Access to the news events dataset files

## 1. Environment Setup

### Clone the Repository
```bash
git clone https://github.com/Nitishvarma50/Data-Analyst---Data-Quality.git
cd Data-Analyst---Data-Quality
```

### Create Virtual Environment
```bash
# Windows Command Prompt
python -m venv fileenv
fileenv\Scripts\activate

# Linux/Mac
python -m venv fileenv
source fileenv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## 2. Configuration

### Update Configuration File
Edit `config/config.yaml` with your specific paths:

```yaml
data:
  raw_data_path: "path/to/your/news_events_*.jsonl"
  processed_data_path: "data/processed/"
  output_path: "output/"

database:
  type: "sqlite"  # or "postgresql", "mysql"
  connection_string: "data_quality.db"

quality:
  sample_size: 1000  # for testing, use "full" for complete dataset
  thresholds:
    completeness: 0.8
    uniqueness: 0.95
    validity: 0.9
    consistency: 0.7
    timeliness: 0.6
```

## 3. Running the Pipeline

### Option 1: Complete Pipeline (Recommended)
```bash
python Scripts/main_pipeline.py --mode full --config config/config.yaml
```

### Option 2: Step-by-Step Execution
```bash
# Step 1: Data Profiling
python EDA/data_profiler.py --input "path/to/data/*.jsonl" --output "data/profiling_results.json"

# Step 2: Quality Assessment
python data_quality/quality_evaluator.py --input "data/profiling_results.json" --output "data/quality_report.json"

# Step 3: Data Cleaning
python data_cleaning/data_cleaner.py --input "data/quality_report.json" --output "data/cleaned_data/"

# Step 4: Database Setup
python -c "from src.database.database_manager import DatabaseManager; db = DatabaseManager(); db.setup_database()"
```

## 4. Running the Dashboard

### Start the Dashboard
```bash
python Dashboard/start_dashboard.py
```

The dashboard will be available at: `http://localhost:8050`

### Dashboard Features
- **Quality Overview**: Overall quality scores and trends
- **Dimension Analysis**: Detailed breakdown by quality dimension
- **Data Profiling**: Statistical summaries and distributions
- **Cleaning Operations**: History of data remediation activities
- **Real-time Monitoring**: Current data quality status

## 5. Database Operations

### Initialize Database
```bash
python -c "
from src.database.database_manager import DatabaseManager
db = DatabaseManager()
db.setup_database()
print('Database initialized successfully!')
"
```

### Load Sample Data
```bash
python -c "
from src.database.data_loader import DataPipelineRunner
runner = DataPipelineRunner()
runner.load_sample_data()
print('Sample data loaded!')
"
```

## 6. Testing

### Run Unit Tests
```bash
python -m pytest test/ -v
```

### Test Specific Components
```bash
# Test data profiling
python test/test_data_profiling.py

# Test quality assessment
python test/test_quality_assessment.py

# Test data cleaning
python test/test_data_cleaning.py
```

## 7. Monitoring and Maintenance

### Automated Quality Checks
```bash
# Set up cron job (Linux/Mac) or Task Scheduler (Windows)
python Scripts/main_pipeline.py --mode incremental --config config/config.yaml
```

### Log Analysis
```bash
# View recent logs
tail -f logs/pipeline.log

# Search for errors
grep "ERROR" logs/pipeline.log
```

## 8. Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure you're in the correct directory
cd Data-Analyst---Data-Quality

# Check Python path
python -c "import sys; print(sys.path)"

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### Database Connection Issues
```bash
# Check database file permissions
ls -la data_quality.db

# Reinitialize database
rm data_quality.db
python -c "from src.database.database_manager import DatabaseManager; db = DatabaseManager(); db.setup_database()"
```

#### Memory Issues with Large Datasets
```bash
# Use sample mode for testing
python Scripts/main_pipeline.py --mode sample --config config/config.yaml

# Increase memory limit
export PYTHONPATH="${PYTHONPATH}:/path/to/project"
```

### Performance Optimization

#### For Large Datasets
1. Use chunked processing
2. Enable database indexing
3. Use parallel processing where available
4. Monitor memory usage

#### Database Optimization
```sql
-- Create additional indexes for your specific queries
CREATE INDEX idx_events_date_category ON events(found_at, category);
CREATE INDEX idx_companies_industry ON companies(industry);

-- Analyze table statistics
ANALYZE;
```

## 9. Output Files

After running the pipeline, you'll find:

- `data/processed/`: Cleaned and processed data files
- `data/quality_reports/`: Quality assessment reports
- `logs/`: Execution logs and error reports
- `output/`: Final analysis results and visualizations

## 10. Next Steps

1. **Customize Quality Metrics**: Modify thresholds in `config/config.yaml`
2. **Add New Data Sources**: Extend the data loader for additional formats
3. **Enhance Dashboard**: Add new visualizations and metrics
4. **Set Up Alerts**: Configure automated notifications for quality issues
5. **Scale Infrastructure**: Move to cloud-based processing for larger datasets

## Support

For issues and questions:
1. Check the logs in `logs/` directory
2. Review the troubleshooting section above
3. Check GitHub Issues for known problems
4. Create a new issue with detailed error information

## Version Information

- **Current Version**: 1.0.0
- **Last Updated**: September 2025
- **Python Compatibility**: 3.8+
- **Database Support**: SQLite, PostgreSQL, MySQL
