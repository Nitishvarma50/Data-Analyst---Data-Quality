# News Events Data Quality Assessment Pipeline

A comprehensive data quality assessment solution for news events datasets, designed to evaluate, profile, and improve data quality across multiple dimensions.

## 🚀 Project Overview

This project implements a robust data quality pipeline that:
- **Processes large JSONL files** containing news events data (~100MB each)
- **Assesses data quality** across 5 key dimensions (Completeness, Uniqueness, Validity, Consistency, Timeliness)
- **Provides automated remediation** and monitoring capabilities
- **Generates executive dashboards** and detailed reports
- **Supports multiple database backends** (SQLite, PostgreSQL, MySQL)

## 📁 Repository Structure

```
Data-Analyst---Data-Quality/
├── Scripts/                    # Main pipeline orchestration
│   └── main_pipeline.py       # Complete pipeline runner
├── EDA/                       # Exploratory Data Analysis
│   └── data_profiler.py      # Data profiling and exploration
├── data_cleaning/             # Data cleaning and remediation
│   └── data_cleaner.py       # Automated data cleaning
├── data_quality/              # Quality assessment engine
│   └── quality_evaluator.py  # DQ dimension evaluation
├── DDL/                       # Database schema and setup
│   └── database_schema.sql   # Complete database schema
├── Dashboard/                 # Interactive dashboard
│   ├── dashboard_app.py      # Main dashboard application
│   └── start_dashboard.py    # Dashboard launcher
├── docs/                      # Documentation and guides
│   └── SETUP_AND_RUNNING_INSTRUCTIONS.md
├── src/                       # Source code (original structure)
├── config/                    # Configuration files
├── data/                      # Data storage and reports
├── logs/                      # Execution logs
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🎯 Key Features

### 1. **Data Profiling & Exploration**
- Statistical analysis of all data fields
- Pattern recognition and anomaly detection
- Data type validation and schema analysis
- Memory-efficient processing of large files

### 2. **Multi-Dimensional Quality Assessment**
- **Completeness**: Missing value analysis and impact assessment
- **Uniqueness**: Duplicate detection and removal strategies
- **Validity**: Format validation (UUIDs, dates, confidence scores)
- **Consistency**: Standardization of categories and locations
- **Timeliness**: Data freshness and detection lag analysis

### 3. **Automated Data Cleaning**
- Intelligent missing value imputation
- Duplicate record identification and removal
- Data format standardization
- Quality score calculation and tracking

### 4. **Interactive Dashboard**
- Real-time quality metrics visualization
- Trend analysis and historical comparisons
- Drill-down capabilities for detailed analysis
- Export functionality for reports

### 5. **Database Integration**
- Optimized schema for news events data
- Support for multiple database engines
- Automated data loading and validation
- Performance monitoring and optimization

## 📊 Dataset Information

**Source Data**: 14 JSONL files (~100MB each)
**Total Size**: ~1.4GB of news events data
**Data Source**: News events extraction from various sources

### Data Schema Overview
Each JSONL record contains:
- **Main Event Data**: Event details, categories, confidence scores
- **Company Information**: Company names, domains, ticker symbols
- **Article Information**: Source articles, URLs, publication dates
- **Relationships**: Links between events, companies, and articles

### Key Event Categories
- `receives_award` / `recognized_as` - Awards and recognitions
- `launches` - Product/service launches
- `hires` - Personnel changes
- `receives_financing` - Funding events
- `is_developing` - Product development activities

## 🛠️ Quick Start

### Prerequisites
- Python 3.8 or higher
- Git
- Access to the news events dataset files

### Installation
```bash
# Clone the repository
git clone https://github.com/Nitishvarma50/Data-Analyst---Data-Quality.git
cd Data-Analyst---Data-Quality

# Create virtual environment
python -m venv fileenv
fileenv\Scripts\activate  # Windows
# source fileenv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline
```bash
# Complete pipeline (recommended)
python Scripts/main_pipeline.py --mode full --config config/config.yaml

# Sample mode for testing
python Scripts/main_pipeline.py --mode sample --config config/config.yaml
```

### Launch Dashboard
```bash
python Dashboard/start_dashboard.py
# Open http://localhost:8050 in your browser
```

## 📈 Data Quality Metrics

### Current Assessment Results
- **Overall Quality Score**: 79.86%
- **Completeness**: 76.04% (Missing values in critical fields)
- **Uniqueness**: 99.32% (Excellent duplicate handling)
- **Validity**: 95.81% (Strong format compliance)
- **Consistency**: 65.91% (Needs standardization improvement)
- **Timeliness**: 47.64% (Data freshness challenges)

### Key Findings
1. **High confidence events** (>0.8) show better quality scores
2. **Location data** needs standardization across sources
3. **Category variations** require mapping to standard terms
4. **Detection lag** averages 3-5 days for most event types

## 🔧 Configuration

### Quality Thresholds
```yaml
quality:
  thresholds:
    completeness: 0.8      # 80% minimum
    uniqueness: 0.95       # 95% minimum
    validity: 0.9          # 90% minimum
    consistency: 0.7       # 70% minimum
    timeliness: 0.6        # 60% minimum
```

### Database Configuration
```yaml
database:
  type: "sqlite"           # sqlite, postgresql, mysql
  connection_string: "data_quality.db"
  pool_size: 10
  max_overflow: 20
```

## 📊 Dashboard Features

### Main Views
1. **Quality Overview**: Executive summary with key metrics
2. **Dimension Analysis**: Detailed breakdown by quality dimension
3. **Trend Analysis**: Quality improvement over time
4. **Data Profiling**: Statistical summaries and distributions
5. **Cleaning Operations**: History of remediation activities

### Interactive Elements
- **Drill-down charts**: Click to explore specific data segments
- **Real-time updates**: Live quality monitoring
- **Export capabilities**: PDF reports and data exports
- **Filtering options**: Date ranges, categories, quality scores

## 🚀 Performance & Scalability

### Current Performance
- **Processing Speed**: ~10,000 records/minute
- **Memory Usage**: Optimized for 8GB+ systems
- **Database Performance**: Sub-second query response for standard operations

### Scalability Features
- **Chunked processing**: Handles files larger than available memory
- **Parallel processing**: Multi-threaded operations where applicable
- **Database optimization**: Indexed queries and connection pooling
- **Incremental updates**: Process only new/changed data

## 🧪 Testing

### Run Tests
```bash
# All tests
python -m pytest test/ -v

# Specific test suites
python -m pytest test/test_data_profiling.py -v
python -m pytest test/test_quality_assessment.py -v
python -m pytest test/test_data_cleaning.py -v
```

### Test Coverage
- **Unit Tests**: Core functionality validation
- **Integration Tests**: End-to-end pipeline testing
- **Performance Tests**: Load and stress testing
- **Quality Tests**: Data quality validation

## 📚 Documentation

- **[Setup & Running Instructions](docs/SETUP_AND_RUNNING_INSTRUCTIONS.md)**: Complete installation and usage guide
- **[API Documentation](docs/API.md)**: Code-level documentation
- **[Quality Metrics Guide](docs/QUALITY_METRICS.md)**: Detailed metric definitions
- **[Troubleshooting](docs/TROUBLESHOOTING.md)**: Common issues and solutions

## 🔄 Monitoring & Maintenance

### Automated Quality Checks
- **Scheduled assessments**: Daily/weekly quality evaluations
- **Alert system**: Notifications for quality degradation
- **Trend monitoring**: Long-term quality tracking
- **Performance metrics**: System health monitoring

### Maintenance Tasks
- **Database optimization**: Regular index maintenance
- **Log rotation**: Automated log file management
- **Backup procedures**: Data and configuration backups
- **Update management**: Dependency and security updates

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Code Standards
- **Python**: PEP 8 compliance
- **Documentation**: Docstring coverage for all functions
- **Testing**: Minimum 80% test coverage
- **Type Hints**: Use type annotations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Data Sources**: News events extraction systems
- **Open Source Libraries**: Pandas, SQLAlchemy, Dash, Plotly
- **Community**: Data quality practitioners and researchers

## 📞 Support

### Getting Help
1. **Documentation**: Check the docs/ directory first
2. **Issues**: Search existing GitHub issues
3. **New Issues**: Create detailed bug reports
4. **Discussions**: Use GitHub discussions for questions

### Contact
- **Repository**: https://github.com/Nitishvarma50/Data-Analyst---Data-Quality
- **Issues**: https://github.com/Nitishvarma50/Data-Analyst---Data-Quality/issues
- **Discussions**: https://github.com/Nitishvarma50/Data-Analyst---Data-Quality/discussions

---

**Last Updated**: September 2025  
**Version**: 1.0.0  
**Python Compatibility**: 3.8+  
**Database Support**: SQLite, PostgreSQL, MySQL