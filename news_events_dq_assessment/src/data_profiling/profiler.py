"""
Data Profiling Module for News Events Dataset
Generates comprehensive statistical profiles and business insights
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import json
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple, Optional
from loguru import logger
import yaml
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_loader import NewsEventsDataLoader, flatten_news_event_record


class NewsEventsProfiler:
    """Comprehensive data profiling for news events dataset"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.loader = NewsEventsDataLoader(self.config['data_paths']['raw_data_dir'])
        self.profile_results = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Default configuration if file not found"""
        return {
            'data_paths': {
                'raw_data_dir': "data/raw",
                'processed_data_dir': "data/processed"
            },
            'processing': {
                'max_records_sample': 10000,
                'chunk_size': 1000
            },
            'critical_fields': ['event_id', 'summary', 'category', 'found_at', 'confidence', 'company_name']
        }
    
    def generate_file_overview(self) -> Dict[str, Any]:
        """Generate overview of all files in the dataset"""
        
        logger.info("Generating file overview...")
        
        file_stats = self.loader.get_file_statistics()
        
        overview = {
            'dataset_summary': {
                'total_files': file_stats['total_files'],
                'total_size_mb': file_stats['total_size_mb'],
                'estimated_total_records': file_stats['estimated_total_records'],
                'avg_file_size_mb': round(file_stats['total_size_mb'] / file_stats['total_files'], 2) if file_stats['total_files'] > 0 else 0,
                'avg_records_per_file': file_stats['estimated_total_records'] // file_stats['total_files'] if file_stats['total_files'] > 0 else 0
            },
            'file_details': file_stats['file_details']
        }
        
        return overview
    
    def profile_sample_data(self, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """Generate comprehensive profile from sample data"""
        
        if sample_size is None:
            sample_size = self.config['processing']['max_records_sample']
        
        logger.info(f"Profiling sample data (size: {sample_size})...")
        
        # Load sample records
        sample_records = self.loader.load_sample_data(sample_size)
        
        if not sample_records:
            logger.error("No sample records loaded")
            return {}
        
        # Flatten records to DataFrame
        flattened_data = []
        for item in sample_records:
            try:
                events = flatten_news_event_record(item['record'])
                for event in events:
                    event['source_file'] = item['source_file']
                    flattened_data.append(event)
            except Exception as e:
                logger.warning(f"Error flattening record from {item.get('source_file', 'unknown')}: {e}")
                continue
        
        if not flattened_data:
            logger.error("No events could be flattened from sample data")
            return {}
        
        df = pd.DataFrame(flattened_data)
        
        logger.info(f"Flattened to {len(df)} events from {len(sample_records)} records")
        
        # Generate comprehensive profile
        profile = {
            'data_structure': self._analyze_data_structure(df),
            'column_profiles': self._generate_column_profiles(df),
            'business_insights': self._generate_business_insights(df),
            'data_quality_preview': self._preview_quality_issues(df),
            'sample_records': self._get_sample_records(df)
        }
        
        self.profile_results = profile
        return profile
    
    def _analyze_data_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the overall structure of the dataset"""
        
        return {
            'dimensions': {
                'total_records': len(df),
                'total_columns': len(df.columns),
                'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
            },
            'column_types': {
                'text_columns': len(df.select_dtypes(include=['object']).columns),
                'numeric_columns': len(df.select_dtypes(include=['int64', 'float64']).columns),
                'boolean_columns': len(df.select_dtypes(include=['bool']).columns),
                'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns)
            },
            'data_schema': {
                col: str(dtype) for col, dtype in df.dtypes.items()
            },
            'column_list': df.columns.tolist()
        }
    
    def _generate_column_profiles(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate detailed profiles for each column"""
        
        profiles = {}
        
        for col in df.columns:
            col_data = df[col]
            
            # Handle unhashable types (dict, list) that might still be present
            try:
                unique_count = int(col_data.nunique())
                unique_percentage = round((col_data.nunique() / len(col_data)) * 100, 2)
            except TypeError as e:
                if "unhashable type" in str(e):
                    logger.warning(f"Column '{col}' contains unhashable types. Converting to strings for analysis.")
                    # Convert unhashable types to strings for analysis
                    col_data_clean = col_data.apply(lambda x: str(x) if isinstance(x, (dict, list)) else x)
                    unique_count = int(col_data_clean.nunique())
                    unique_percentage = round((col_data_clean.nunique() / len(col_data_clean)) * 100, 2)
                else:
                    logger.error(f"Error calculating unique values for column '{col}': {e}")
                    unique_count = 0
                    unique_percentage = 0.0
            
            base_profile = {
                'data_type': str(col_data.dtype),
                'total_count': len(col_data),
                'null_count': int(col_data.isnull().sum()),
                'null_percentage': round((col_data.isnull().sum() / len(col_data)) * 100, 2),
                'unique_count': unique_count,
                'unique_percentage': unique_percentage
            }
            
            # Add type-specific analysis
            if col_data.dtype == 'object':
                # Text column analysis
                non_null_data = col_data.dropna()
                if len(non_null_data) > 0:
                    str_lengths = non_null_data.astype(str).str.len()
                    value_counts = non_null_data.value_counts()
                    
                    base_profile.update({
                        'avg_length': round(str_lengths.mean(), 2),
                        'max_length': int(str_lengths.max()),
                        'min_length': int(str_lengths.min()),
                        'most_frequent': value_counts.head(3).to_dict(),
                        'sample_values': non_null_data.head(5).tolist()
                    })
            
            elif col_data.dtype in ['int64', 'float64']:
                # Numeric column analysis
                numeric_data = col_data.dropna()
                if len(numeric_data) > 0:
                    base_profile.update({
                        'min_value': float(numeric_data.min()),
                        'max_value': float(numeric_data.max()),
                        'mean_value': round(float(numeric_data.mean()), 4),
                        'median_value': float(numeric_data.median()),
                        'std_deviation': round(float(numeric_data.std()), 4),
                        'zero_count': int((numeric_data == 0).sum()),
                        'negative_count': int((numeric_data < 0).sum()),
                        'quartiles': {
                            'q25': float(numeric_data.quantile(0.25)),
                            'q75': float(numeric_data.quantile(0.75))
                        }
                    })
            
            elif col_data.dtype == 'bool':
                # Boolean column analysis
                base_profile.update({
                    'true_count': int(col_data.sum()),
                    'false_count': int((~col_data).sum()),
                    'true_percentage': round((col_data.sum() / len(col_data)) * 100, 2)
                })
            
            profiles[col] = base_profile
        
        return profiles
    
    def _generate_business_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate business-relevant insights from the data"""
        
        insights = {}
        
        # Event category analysis
        if 'category' in df.columns:
            category_counts = df['category'].value_counts()
            insights['event_categories'] = {
                'total_categories': len(category_counts),
                'distribution': category_counts.to_dict(),
                'most_common': category_counts.head(5).to_dict(),
                'rare_categories': category_counts[category_counts == 1].index.tolist()[:10]  # Limit to first 10
            }
        
        # Company analysis
        if 'company_name' in df.columns:
            company_counts = df['company_name'].value_counts()
            insights['company_coverage'] = {
                'total_companies': int(df['company_name'].nunique()),
                'companies_with_multiple_events': int((company_counts > 1).sum()),
                'most_active_companies': company_counts.head(10).to_dict(),
                'single_event_companies': int((company_counts == 1).sum())
            }
        
        # Geographic distribution
        if 'location' in df.columns:
            location_counts = df['location'].value_counts()
            insights['geographic_distribution'] = {
                'total_locations': int(df['location'].nunique()),
                'location_coverage': location_counts.head(20).to_dict(),  # Top 20 locations
                'missing_location_rate': round((df['location'].isnull().sum() / len(df)) * 100, 2)
            }
        
        # Confidence score analysis
        if 'confidence' in df.columns:
            confidence_data = df['confidence'].dropna()
            if len(confidence_data) > 0:
                insights['confidence_analysis'] = {
                    'mean_confidence': round(float(confidence_data.mean()), 4),
                    'median_confidence': round(float(confidence_data.median()), 4),
                    'std_confidence': round(float(confidence_data.std()), 4),
                    'high_confidence_events': int((confidence_data >= 0.9).sum()),
                    'medium_confidence_events': int(((confidence_data >= 0.7) & (confidence_data < 0.9)).sum()),
                    'low_confidence_events': int((confidence_data < 0.7).sum()),
                    'confidence_distribution': {
                        '0.9-1.0': int((confidence_data >= 0.9).sum()),
                        '0.8-0.9': int(((confidence_data >= 0.8) & (confidence_data < 0.9)).sum()),
                        '0.7-0.8': int(((confidence_data >= 0.7) & (confidence_data < 0.8)).sum()),
                        '0.6-0.7': int(((confidence_data >= 0.6) & (confidence_data < 0.7)).sum()),
                        '< 0.6': int((confidence_data < 0.6).sum())
                    }
                }
        
        # Temporal analysis (FIXED VERSION)
        if 'found_at' in df.columns:
            df_temp = df.copy()
            df_temp['found_at_dt'] = pd.to_datetime(df_temp['found_at'], errors='coerce')
            
            valid_dates = df_temp['found_at_dt'].dropna()
            if len(valid_dates) > 0:
                # Convert to UTC and then remove timezone info to avoid the warning
                valid_dates_utc = valid_dates.dt.tz_convert('UTC').dt.tz_localize(None)
                
                # Create year-month strings manually to avoid Period serialization issues
                year_month_strings = valid_dates_utc.dt.strftime('%Y-%m')
                events_by_month_counts = year_month_strings.value_counts().sort_index()
                
                insights['temporal_analysis'] = {
                    'total_events_with_dates': len(valid_dates),
                    'date_range': {
                        'earliest': valid_dates.min().isoformat(),
                        'latest': valid_dates.max().isoformat(),
                        'span_days': int((valid_dates.max() - valid_dates.min()).days)
                    },
                    'events_by_year': valid_dates.dt.year.value_counts().sort_index().to_dict(),
                    'events_by_month': events_by_month_counts.head(12).to_dict(),  # Convert Period keys to strings
                    'recent_activity': {
                        'last_30_days': int((valid_dates > (datetime.now(timezone.utc) - pd.Timedelta(days=30))).sum()),
                        'last_90_days': int((valid_dates > (datetime.now(timezone.utc) - pd.Timedelta(days=90))).sum()),
                        'last_year': int((valid_dates > (datetime.now(timezone.utc) - pd.Timedelta(days=365))).sum())
                    }
                }
        
        # Financial events analysis
        if 'amount_normalized' in df.columns:
            financial_data = df[df['amount_normalized'].notna()].copy()
            if len(financial_data) > 0:
                insights['financial_events'] = {
                    'total_financial_events': len(financial_data),
                    'total_amount': float(financial_data['amount_normalized'].sum()),
                    'average_amount': round(float(financial_data['amount_normalized'].mean()), 2),
                    'median_amount': float(financial_data['amount_normalized'].median()),
                    'financing_types': financial_data['financing_type'].value_counts().to_dict() if 'financing_type' in financial_data.columns else {},
                    'largest_amounts': financial_data.nlargest(5, 'amount_normalized')[
                        ['company_name', 'amount_normalized', 'financing_type']
                    ].to_dict('records')
                }
        
        return insights
    
    def _preview_quality_issues(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Preview potential data quality issues"""
        
        issues = {}
        
        # Missing values in critical fields
        critical_fields = self.config.get('critical_fields', ['event_id', 'summary', 'category'])
        missing_critical = {}
        
        for field in critical_fields:
            if field in df.columns:
                missing_count = int(df[field].isnull().sum())
                missing_critical[field] = {
                    'missing_count': missing_count,
                    'missing_percentage': round((missing_count / len(df)) * 100, 2)
                }
        
        issues['completeness_preview'] = missing_critical
        
        # Duplicate detection
        duplicate_ids = int(df['event_id'].duplicated().sum()) if 'event_id' in df.columns else 0
        duplicate_summaries = int(df['summary'].duplicated().sum()) if 'summary' in df.columns else 0
        
        issues['uniqueness_preview'] = {
            'duplicate_event_ids': duplicate_ids,
            'duplicate_summaries': duplicate_summaries,
            'potential_duplicates_rate': round((duplicate_ids / len(df)) * 100, 2) if len(df) > 0 else 0
        }
        
        # Format validation preview
        if 'confidence' in df.columns:
            confidence_data = df['confidence'].dropna()
            invalid_confidence = int((~confidence_data.between(0, 1, inclusive='both')).sum())
            issues['validity_preview'] = {
                'invalid_confidence_scores': invalid_confidence,
                'confidence_out_of_range_rate': round((invalid_confidence / len(confidence_data)) * 100, 2) if len(confidence_data) > 0 else 0,
                'confidence_stats': {
                    'min': float(confidence_data.min()) if len(confidence_data) > 0 else None,
                    'max': float(confidence_data.max()) if len(confidence_data) > 0 else None
                }
            }
        
        # Category consistency
        if 'category' in df.columns:
            category_variations = df['category'].nunique()
            category_list = df['category'].unique().tolist()
            issues['consistency_preview'] = {
                'category_variations': int(category_variations),
                'category_list': [cat for cat in category_list if pd.notna(cat)][:20]  # Limit to first 20
            }
        
        return issues
    
    def _get_sample_records(self, df: pd.DataFrame, num_samples: int = 3) -> Dict[str, Any]:
        """Get sample records for different scenarios"""
        
        samples = {}
        
        # High quality record example
        if 'confidence' in df.columns and len(df) > 0:
            high_conf = df[df['confidence'] >= 0.9]
            if len(high_conf) > 0:
                sample_record = high_conf.iloc[0]
                samples['high_quality_example'] = {
                    'event_id': sample_record.get('event_id'),
                    'summary': sample_record.get('summary', '')[:100] + '...' if len(str(sample_record.get('summary', ''))) > 100 else sample_record.get('summary'),
                    'confidence': sample_record.get('confidence'),
                    'company_name': sample_record.get('company_name'),
                    'category': sample_record.get('category'),
                    'location': sample_record.get('location')
                }
        
        # Low quality record example
        if 'confidence' in df.columns and len(df) > 0:
            low_conf = df[df['confidence'] < 0.7]
            if len(low_conf) > 0:
                sample_record = low_conf.iloc[0]
                samples['low_quality_example'] = {
                    'event_id': sample_record.get('event_id'),
                    'summary': sample_record.get('summary', '')[:100] + '...' if len(str(sample_record.get('summary', ''))) > 100 else sample_record.get('summary'),
                    'confidence': sample_record.get('confidence'),
                    'company_name': sample_record.get('company_name'),
                    'category': sample_record.get('category')
                }
        
        # Missing data example
        critical_fields = ['summary', 'company_name', 'category']
        available_fields = [field for field in critical_fields if field in df.columns]
        
        if available_fields:
            missing_data_mask = df[available_fields].isnull().any(axis=1)
            if missing_data_mask.any():
                sample_record = df[missing_data_mask].iloc[0]
                missing_fields = [field for field in available_fields if pd.isnull(sample_record.get(field))]
                available_data = {k: v for k, v in sample_record.to_dict().items() 
                                if pd.notna(v) and k in available_fields}
                
                samples['missing_data_example'] = {
                    'event_id': sample_record.get('event_id'),
                    'missing_fields': missing_fields,
                    'available_data': available_data
                }
        
        return samples
    
    def generate_profiling_report(self) -> str:
        """Generate a comprehensive profiling report"""
        
        if not self.profile_results:
            logger.error("No profiling results available. Run profile_sample_data() first.")
            return ""
        
        profile = self.profile_results
        
        report = f"""# News Events Dataset - Data Profiling Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview
- **Source Directory**: {self.config['data_paths']['raw_data_dir']}
- **Sample Records Analyzed**: {profile['data_structure']['dimensions']['total_records']:,}
- **Total Columns**: {profile['data_structure']['dimensions']['total_columns']}
- **Memory Usage (Sample)**: {profile['data_structure']['dimensions']['memory_usage_mb']} MB

## Data Structure Analysis

### Column Distribution by Type
- Text Columns: {profile['data_structure']['column_types']['text_columns']}
- Numeric Columns: {profile['data_structure']['column_types']['numeric_columns']}
- Boolean Columns: {profile['data_structure']['column_types']['boolean_columns']}
- DateTime Columns: {profile['data_structure']['column_types']['datetime_columns']}

### Key Columns Identified
"""
        
        # Add column list
        columns = profile['data_structure']['column_list']
        core_columns = [col for col in columns if col in ['event_id', 'summary', 'category', 'confidence', 'company_name', 'found_at']]
        optional_columns = [col for col in columns if col not in core_columns]
        
        report += f"**Core Event Fields**: {', '.join(core_columns)}\n\n"
        report += f"**Extended Fields**: {len(optional_columns)} additional fields including location, amounts, products, etc.\n\n"
        
        # Business insights section
        if 'business_insights' in profile:
            insights = profile['business_insights']
            
            if 'event_categories' in insights:
                categories = insights['event_categories']
                report += f"## Event Categories Analysis\n"
                report += f"- **Total Categories**: {categories['total_categories']}\n"
                report += f"- **Most Common Event Types**:\n"
                for category, count in categories['most_common'].items():
                    report += f"  - {category}: {count} events\n"
                
                if categories['rare_categories']:
                    report += f"- **Rare Categories** (single occurrence): {len(categories['rare_categories'])} categories\n"
            
            if 'company_coverage' in insights:
                companies = insights['company_coverage']
                report += f"\n## Company Coverage Analysis\n"
                report += f"- **Total Companies**: {companies['total_companies']:,}\n"
                report += f"- **Companies with Multiple Events**: {companies['companies_with_multiple_events']:,}\n"
                report += f"- **Single Event Companies**: {companies['single_event_companies']:,}\n"
                report += f"- **Top Active Companies**:\n"
                for company, count in list(companies['most_active_companies'].items())[:5]:
                    report += f"  - {company}: {count} events\n"
            
            if 'confidence_analysis' in insights:
                confidence = insights['confidence_analysis']
                report += f"\n## Confidence Score Analysis\n"
                report += f"- **Mean Confidence**: {confidence['mean_confidence']}\n"
                report += f"- **High Confidence Events** (≥0.9): {confidence['high_confidence_events']:,}\n"
                report += f"- **Medium Confidence Events** (0.7-0.9): {confidence['medium_confidence_events']:,}\n"
                report += f"- **Low Confidence Events** (<0.7): {confidence['low_confidence_events']:,}\n"
            
            if 'temporal_analysis' in insights:
                temporal = insights['temporal_analysis']
                report += f"\n## Temporal Analysis\n"
                report += f"- **Date Range**: {temporal['date_range']['earliest']} to {temporal['date_range']['latest']}\n"
                report += f"- **Time Span**: {temporal['date_range']['span_days']:,} days\n"
                report += f"- **Recent Activity**:\n"
                report += f"  - Last 30 days: {temporal['recent_activity']['last_30_days']:,} events\n"
                report += f"  - Last 90 days: {temporal['recent_activity']['last_90_days']:,} events\n"
                report += f"  - Last year: {temporal['recent_activity']['last_year']:,} events\n"
            
            if 'financial_events' in insights:
                financial = insights['financial_events']
                report += f"\n## Financial Events Analysis\n"
                report += f"- **Total Financial Events**: {financial['total_financial_events']:,}\n"
                report += f"- **Total Amount**: ${financial['total_amount']:,.2f}\n"
                report += f"- **Average Amount**: ${financial['average_amount']:,.2f}\n"
                report += f"- **Median Amount**: ${financial['median_amount']:,.2f}\n"
        
        # Data quality preview
        if 'data_quality_preview' in profile:
            issues = profile['data_quality_preview']
            report += f"\n## Data Quality Preview\n"
            
            if 'completeness_preview' in issues:
                report += f"### Missing Data Analysis\n"
                for field, stats in issues['completeness_preview'].items():
                    if stats['missing_count'] > 0:
                        report += f"- **{field}**: {stats['missing_count']:,} missing ({stats['missing_percentage']}%)\n"
            
            if 'uniqueness_preview' in issues:
                uniq = issues['uniqueness_preview']
                if uniq['duplicate_event_ids'] > 0:
                    report += f"\n### Duplicate Issues Preview\n"
                    report += f"- **Duplicate Event IDs**: {uniq['duplicate_event_ids']:,}\n"
                    report += f"- **Duplicate Summaries**: {uniq['duplicate_summaries']:,}\n"
                    report += f"- **Duplication Rate**: {uniq['potential_duplicates_rate']}%\n"
            
            if 'validity_preview' in issues:
                validity = issues['validity_preview']
                if validity['invalid_confidence_scores'] > 0:
                    report += f"\n### Validity Issues Preview\n"
                    report += f"- **Invalid Confidence Scores**: {validity['invalid_confidence_scores']:,}\n"
                    report += f"- **Out of Range Rate**: {validity['confidence_out_of_range_rate']}%\n"
        
        return report
    
    def save_profile_results(self, output_path: Optional[str] = None) -> str:
        """Save profiling results to JSON file"""
        
        if not self.profile_results:
            logger.error("No profile results to save")
            return ""
        
        if output_path is None:
            output_dir = Path(self.config['data_paths']['processed_data_dir'])
        else:
            output_dir = Path(output_path)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save profile as JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        profile_file = output_dir / f"data_profile_{timestamp}.json"
        
        # Convert numpy types to Python native types for JSON serialization
        profile_json = self._convert_for_json(self.profile_results)
        
        with open(profile_file, 'w', encoding='utf-8') as f:
            json.dump(profile_json, f, indent=2, default=str)
        
        # Save report as markdown
        report_file = output_dir / f"profiling_report_{timestamp}.md"
        report_content = self.generate_profiling_report()
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Profile results saved to: {profile_file}")
        logger.info(f"Profile report saved to: {report_file}")
        
        return str(profile_file)
    
    def _convert_for_json(self, obj):
        """Convert numpy types and other non-JSON-serializable types"""
        if isinstance(obj, dict):
            return {str(key): self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, pd.Period):
            return str(obj)  # Convert Period objects to strings
        else:
            return obj
    
    def generate_comprehensive_profile(self, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """Generate comprehensive profile - alias for run_complete_profiling"""
        return self.run_complete_profiling(sample_size)
    
    def run_complete_profiling(self, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """Run the complete profiling pipeline"""
        
        logger.info("Starting comprehensive data profiling...")
        
        # Step 1: File overview
        logger.info("Step 1: Generating file overview...")
        file_overview = self.generate_file_overview()
        
        # Step 2: Sample data profiling
        logger.info("Step 2: Profiling sample data...")
        sample_profile = self.profile_sample_data(sample_size)
        
        # Step 3: Combine results
        complete_profile = {
            'file_overview': file_overview,
            'sample_analysis': sample_profile,
            'profiling_metadata': {
                'generated_at': datetime.now().isoformat(),
                'sample_size_used': sample_size or self.config['processing']['max_records_sample'],
                'config_used': self.config
            }
        }
        
        # Step 4: Save results
        logger.info("Step 3: Saving profiling results...")
        output_file = self.save_profile_results()
        
        logger.info("Data profiling completed successfully!")
        
        return complete_profile


if __name__ == "__main__":
    # Initialize and run profiling
    profiler = NewsEventsProfiler()
    
    # Run complete profiling
    results = profiler.run_complete_profiling(sample_size=5000)
    
    # Print summary
    print("Profiling completed!")
    print(f"Files analyzed: {results['file_overview']['dataset_summary']['total_files']}")
    print(f"Sample events profiled: {results['sample_analysis']['data_structure']['dimensions']['total_records']:,}")
    print(f"Companies identified: {results['sample_analysis']['business_insights']['company_coverage']['total_companies']:,}")
    def _preview_quality_issues(self, df: pd.DataFrame) -> Dict[str, Any]:

        """Preview potential data quality issues"""

        

        issues = {}

        

        # Missing values in critical fields

        critical_fields = self.config.get('critical_fields', ['event_id', 'summary', 'category'])

        missing_critical = {}

        

        for field in critical_fields:

            if field in df.columns:

                missing_count = int(df[field].isnull().sum())

                missing_critical[field] = {

                    'missing_count': missing_count,

                    'missing_percentage': round((missing_count / len(df)) * 100, 2)

                }

        

        issues['completeness_preview'] = missing_critical

        

        # Duplicate detection

        duplicate_ids = int(df['event_id'].duplicated().sum()) if 'event_id' in df.columns else 0

        duplicate_summaries = int(df['summary'].duplicated().sum()) if 'summary' in df.columns else 0

        

        issues['uniqueness_preview'] = {

            'duplicate_event_ids': duplicate_ids,

            'duplicate_summaries': duplicate_summaries,

            'potential_duplicates_rate': round((duplicate_ids / len(df)) * 100, 2) if len(df) > 0 else 0

        }

        

        # Format validation preview

        if 'confidence' in df.columns:

            confidence_data = df['confidence'].dropna()

            invalid_confidence = int((~confidence_data.between(0, 1, inclusive='both')).sum())

            issues['validity_preview'] = {

                'invalid_confidence_scores': invalid_confidence,

                'confidence_out_of_range_rate': round((invalid_confidence / len(confidence_data)) * 100, 2) if len(confidence_data) > 0 else 0,

                'confidence_stats': {

                    'min': float(confidence_data.min()) if len(confidence_data) > 0 else None,

                    'max': float(confidence_data.max()) if len(confidence_data) > 0 else None

                }

            }

        

        # Category consistency

        if 'category' in df.columns:

            category_variations = df['category'].nunique()

            category_list = df['category'].unique().tolist()

            issues['consistency_preview'] = {

                'category_variations': int(category_variations),

                'category_list': [cat for cat in category_list if pd.notna(cat)][:20]  # Limit to first 20

            }

        

        return issues

    

    def _get_sample_records(self, df: pd.DataFrame, num_samples: int = 3) -> Dict[str, Any]:

        """Get sample records for different scenarios"""

        

        samples = {}

        

        # High quality record example

        if 'confidence' in df.columns and len(df) > 0:

            high_conf = df[df['confidence'] >= 0.9]

            if len(high_conf) > 0:

                sample_record = high_conf.iloc[0]

                samples['high_quality_example'] = {

                    'event_id': sample_record.get('event_id'),

                    'summary': sample_record.get('summary', '')[:100] + '...' if len(str(sample_record.get('summary', ''))) > 100 else sample_record.get('summary'),

                    'confidence': sample_record.get('confidence'),

                    'company_name': sample_record.get('company_name'),

                    'category': sample_record.get('category'),

                    'location': sample_record.get('location')

                }

        

        # Low quality record example

        if 'confidence' in df.columns and len(df) > 0:

            low_conf = df[df['confidence'] < 0.7]

            if len(low_conf) > 0:

                sample_record = low_conf.iloc[0]

                samples['low_quality_example'] = {

                    'event_id': sample_record.get('event_id'),

                    'summary': sample_record.get('summary', '')[:100] + '...' if len(str(sample_record.get('summary', ''))) > 100 else sample_record.get('summary'),

                    'confidence': sample_record.get('confidence'),

                    'company_name': sample_record.get('company_name'),

                    'category': sample_record.get('category')

                }

        

        # Missing data example

        critical_fields = ['summary', 'company_name', 'category']

        available_fields = [field for field in critical_fields if field in df.columns]

        

        if available_fields:

            missing_data_mask = df[available_fields].isnull().any(axis=1)

            if missing_data_mask.any():

                sample_record = df[missing_data_mask].iloc[0]

                missing_fields = [field for field in available_fields if pd.isnull(sample_record.get(field))]

                available_data = {k: v for k, v in sample_record.to_dict().items() 

                                if pd.notna(v) and k in available_fields}

                

                samples['missing_data_example'] = {

                    'event_id': sample_record.get('event_id'),

                    'missing_fields': missing_fields,

                    'available_data': available_data

                }

        

        return samples

    

    def generate_profiling_report(self) -> str:

        """Generate a comprehensive profiling report"""

        

        if not self.profile_results:

            logger.error("No profiling results available. Run profile_sample_data() first.")

            return ""

        

        profile = self.profile_results

        

        report = f"""# News Events Dataset - Data Profiling Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}



## Dataset Overview

- **Source Directory**: {self.config['data_paths']['raw_data_dir']}

- **Sample Records Analyzed**: {profile['data_structure']['dimensions']['total_records']:,}

- **Total Columns**: {profile['data_structure']['dimensions']['total_columns']}

- **Memory Usage (Sample)**: {profile['data_structure']['dimensions']['memory_usage_mb']} MB



## Data Structure Analysis



### Column Distribution by Type

- Text Columns: {profile['data_structure']['column_types']['text_columns']}

- Numeric Columns: {profile['data_structure']['column_types']['numeric_columns']}

- Boolean Columns: {profile['data_structure']['column_types']['boolean_columns']}

- DateTime Columns: {profile['data_structure']['column_types']['datetime_columns']}



### Key Columns Identified

"""

        

        # Add column list

        columns = profile['data_structure']['column_list']

        core_columns = [col for col in columns if col in ['event_id', 'summary', 'category', 'confidence', 'company_name', 'found_at']]

        optional_columns = [col for col in columns if col not in core_columns]

        

        report += f"**Core Event Fields**: {', '.join(core_columns)}\n\n"

        report += f"**Extended Fields**: {len(optional_columns)} additional fields including location, amounts, products, etc.\n\n"

        

        # Business insights section

        if 'business_insights' in profile:

            insights = profile['business_insights']

            

            if 'event_categories' in insights:

                categories = insights['event_categories']

                report += f"## Event Categories Analysis\n"

                report += f"- **Total Categories**: {categories['total_categories']}\n"

                report += f"- **Most Common Event Types**:\n"

                for category, count in categories['most_common'].items():

                    report += f"  - {category}: {count} events\n"

                

                if categories['rare_categories']:

                    report += f"- **Rare Categories** (single occurrence): {len(categories['rare_categories'])} categories\n"

            

            if 'company_coverage' in insights:

                companies = insights['company_coverage']

                report += f"\n## Company Coverage Analysis\n"

                report += f"- **Total Companies**: {companies['total_companies']:,}\n"

                report += f"- **Companies with Multiple Events**: {companies['companies_with_multiple_events']:,}\n"

                report += f"- **Single Event Companies**: {companies['single_event_companies']:,}\n"

                report += f"- **Top Active Companies**:\n"

                for company, count in list(companies['most_active_companies'].items())[:5]:

                    report += f"  - {company}: {count} events\n"

            

            if 'confidence_analysis' in insights:

                confidence = insights['confidence_analysis']

                report += f"\n## Confidence Score Analysis\n"

                report += f"- **Mean Confidence**: {confidence['mean_confidence']}\n"

                report += f"- **High Confidence Events** (≥0.9): {confidence['high_confidence_events']:,}\n"

                report += f"- **Medium Confidence Events** (0.7-0.9): {confidence['medium_confidence_events']:,}\n"

                report += f"- **Low Confidence Events** (<0.7): {confidence['low_confidence_events']:,}\n"

            

            if 'temporal_analysis' in insights:

                temporal = insights['temporal_analysis']

                report += f"\n## Temporal Analysis\n"

                report += f"- **Date Range**: {temporal['date_range']['earliest']} to {temporal['date_range']['latest']}\n"

                report += f"- **Time Span**: {temporal['date_range']['span_days']:,} days\n"

                report += f"- **Recent Activity**:\n"

                report += f"  - Last 30 days: {temporal['recent_activity']['last_30_days']:,} events\n"

                report += f"  - Last 90 days: {temporal['recent_activity']['last_90_days']:,} events\n"

                report += f"  - Last year: {temporal['recent_activity']['last_year']:,} events\n"

            

            if 'financial_events' in insights:

                financial = insights['financial_events']

                report += f"\n## Financial Events Analysis\n"

                report += f"- **Total Financial Events**: {financial['total_financial_events']:,}\n"

                report += f"- **Total Amount**: ${financial['total_amount']:,.2f}\n"

                report += f"- **Average Amount**: ${financial['average_amount']:,.2f}\n"

                report += f"- **Median Amount**: ${financial['median_amount']:,.2f}\n"

        

        # Data quality preview

        if 'data_quality_preview' in profile:

            issues = profile['data_quality_preview']

            report += f"\n## Data Quality Preview\n"

            

            if 'completeness_preview' in issues:

                report += f"### Missing Data Analysis\n"

                for field, stats in issues['completeness_preview'].items():

                    if stats['missing_count'] > 0:

                        report += f"- **{field}**: {stats['missing_count']:,} missing ({stats['missing_percentage']}%)\n"

            

            if 'uniqueness_preview' in issues:

                uniq = issues['uniqueness_preview']

                if uniq['duplicate_event_ids'] > 0:

                    report += f"\n### Duplicate Issues Preview\n"

                    report += f"- **Duplicate Event IDs**: {uniq['duplicate_event_ids']:,}\n"

                    report += f"- **Duplicate Summaries**: {uniq['duplicate_summaries']:,}\n"

                    report += f"- **Duplication Rate**: {uniq['potential_duplicates_rate']}%\n"

            

            if 'validity_preview' in issues:

                validity = issues['validity_preview']

                if validity['invalid_confidence_scores'] > 0:

                    report += f"\n### Validity Issues Preview\n"

                    report += f"- **Invalid Confidence Scores**: {validity['invalid_confidence_scores']:,}\n"

                    report += f"- **Out of Range Rate**: {validity['confidence_out_of_range_rate']}%\n"

        

        return report

    

    def save_profile_results(self, output_path: Optional[str] = None) -> str:

        """Save profiling results to JSON file"""

        

        if not self.profile_results:

            logger.error("No profile results to save")

            return ""

        

        if output_path is None:

            output_dir = Path(self.config['data_paths']['processed_data_dir'])

        else:

            output_dir = Path(output_path)

        

        output_dir.mkdir(parents=True, exist_ok=True)

        

        # Save profile as JSON

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        profile_file = output_dir / f"data_profile_{timestamp}.json"

        

        # Convert numpy types to Python native types for JSON serialization

        profile_json = self._convert_for_json(self.profile_results)

        

        with open(profile_file, 'w', encoding='utf-8') as f:

            json.dump(profile_json, f, indent=2, default=str)

        

        # Save report as markdown

        report_file = output_dir / f"profiling_report_{timestamp}.md"

        report_content = self.generate_profiling_report()

        

        with open(report_file, 'w', encoding='utf-8') as f:

            f.write(report_content)

        

        logger.info(f"Profile results saved to: {profile_file}")

        logger.info(f"Profile report saved to: {report_file}")

        

        return str(profile_file)

    

    def _convert_for_json(self, obj):

        """Convert numpy types and other non-JSON-serializable types"""

        if isinstance(obj, dict):

            return {str(key): self._convert_for_json(value) for key, value in obj.items()}

        elif isinstance(obj, list):

            return [self._convert_for_json(item) for item in obj]

        elif isinstance(obj, np.integer):

            return int(obj)

        elif isinstance(obj, np.floating):

            return float(obj)

        elif isinstance(obj, np.bool_):

            return bool(obj)

        elif isinstance(obj, (pd.Timestamp, datetime)):

            return obj.isoformat()

        elif isinstance(obj, pd.Period):

            return str(obj)  # Convert Period objects to strings

        else:

            return obj

    

    def run_complete_profiling(self, sample_size: Optional[int] = None) -> Dict[str, Any]:

        """Run the complete profiling pipeline"""

        

        logger.info("Starting comprehensive data profiling...")

        

        # Step 1: File overview

        logger.info("Step 1: Generating file overview...")

        file_overview = self.generate_file_overview()

        

        # Step 2: Sample data profiling

        logger.info("Step 2: Profiling sample data...")

        sample_profile = self.profile_sample_data(sample_size)

        

        # Step 3: Combine results

        complete_profile = {

            'file_overview': file_overview,

            'sample_analysis': sample_profile,

            'profiling_metadata': {

                'generated_at': datetime.now().isoformat(),

                'sample_size_used': sample_size or self.config['processing']['max_records_sample'],

                'config_used': self.config

            }

        }

        

        # Step 4: Save results

        logger.info("Step 3: Saving profiling results...")

        output_file = self.save_profile_results()

        

        logger.info("Data profiling completed successfully!")

        

        return complete_profile





if __name__ == "__main__":

    # Initialize and run profiling

    profiler = NewsEventsProfiler()

    

    # Run complete profiling

    results = profiler.run_complete_profiling(sample_size=5000)

    

    # Print summary

    print("Profiling completed!")

    print(f"Files analyzed: {results['file_overview']['dataset_summary']['total_files']}")

    print(f"Sample events profiled: {results['sample_analysis']['data_structure']['dimensions']['total_records']:,}")

    print(f"Companies identified: {results['sample_analysis']['business_insights']['company_coverage']['total_companies']:,}")

    def _preview_quality_issues(self, df: pd.DataFrame) -> Dict[str, Any]:

        """Preview potential data quality issues"""

        

        issues = {}

        

        # Missing values in critical fields

        critical_fields = self.config.get('critical_fields', ['event_id', 'summary', 'category'])

        missing_critical = {}

        

        for field in critical_fields:

            if field in df.columns:

                missing_count = int(df[field].isnull().sum())

                missing_critical[field] = {

                    'missing_count': missing_count,

                    'missing_percentage': round((missing_count / len(df)) * 100, 2)

                }

        

        issues['completeness_preview'] = missing_critical

        

        # Duplicate detection

        duplicate_ids = int(df['event_id'].duplicated().sum()) if 'event_id' in df.columns else 0

        duplicate_summaries = int(df['summary'].duplicated().sum()) if 'summary' in df.columns else 0

        

        issues['uniqueness_preview'] = {

            'duplicate_event_ids': duplicate_ids,

            'duplicate_summaries': duplicate_summaries,

            'potential_duplicates_rate': round((duplicate_ids / len(df)) * 100, 2) if len(df) > 0 else 0

        }

        

        # Format validation preview

        if 'confidence' in df.columns:

            confidence_data = df['confidence'].dropna()

            invalid_confidence = int((~confidence_data.between(0, 1, inclusive='both')).sum())

            issues['validity_preview'] = {

                'invalid_confidence_scores': invalid_confidence,

                'confidence_out_of_range_rate': round((invalid_confidence / len(confidence_data)) * 100, 2) if len(confidence_data) > 0 else 0,

                'confidence_stats': {

                    'min': float(confidence_data.min()) if len(confidence_data) > 0 else None,

                    'max': float(confidence_data.max()) if len(confidence_data) > 0 else None

                }

            }

        

        # Category consistency

        if 'category' in df.columns:

            category_variations = df['category'].nunique()

            category_list = df['category'].unique().tolist()

            issues['consistency_preview'] = {

                'category_variations': int(category_variations),

                'category_list': [cat for cat in category_list if pd.notna(cat)][:20]  # Limit to first 20

            }

        

        return issues

    

    def _get_sample_records(self, df: pd.DataFrame, num_samples: int = 3) -> Dict[str, Any]:

        """Get sample records for different scenarios"""

        

        samples = {}

        

        # High quality record example

        if 'confidence' in df.columns and len(df) > 0:

            high_conf = df[df['confidence'] >= 0.9]

            if len(high_conf) > 0:

                sample_record = high_conf.iloc[0]

                samples['high_quality_example'] = {

                    'event_id': sample_record.get('event_id'),

                    'summary': sample_record.get('summary', '')[:100] + '...' if len(str(sample_record.get('summary', ''))) > 100 else sample_record.get('summary'),

                    'confidence': sample_record.get('confidence'),

                    'company_name': sample_record.get('company_name'),

                    'category': sample_record.get('category'),

                    'location': sample_record.get('location')

                }

        

        # Low quality record example

        if 'confidence' in df.columns and len(df) > 0:

            low_conf = df[df['confidence'] < 0.7]

            if len(low_conf) > 0:

                sample_record = low_conf.iloc[0]

                samples['low_quality_example'] = {

                    'event_id': sample_record.get('event_id'),

                    'summary': sample_record.get('summary', '')[:100] + '...' if len(str(sample_record.get('summary', ''))) > 100 else sample_record.get('summary'),

                    'confidence': sample_record.get('confidence'),

                    'company_name': sample_record.get('company_name'),

                    'category': sample_record.get('category')

                }

        

        # Missing data example

        critical_fields = ['summary', 'company_name', 'category']

        available_fields = [field for field in critical_fields if field in df.columns]

        

        if available_fields:

            missing_data_mask = df[available_fields].isnull().any(axis=1)

            if missing_data_mask.any():

                sample_record = df[missing_data_mask].iloc[0]

                missing_fields = [field for field in available_fields if pd.isnull(sample_record.get(field))]

                available_data = {k: v for k, v in sample_record.to_dict().items() 

                                if pd.notna(v) and k in available_fields}

                

                samples['missing_data_example'] = {

                    'event_id': sample_record.get('event_id'),

                    'missing_fields': missing_fields,

                    'available_data': available_data

                }

        

        return samples

    

    def generate_profiling_report(self) -> str:

        """Generate a comprehensive profiling report"""

        

        if not self.profile_results:

            logger.error("No profiling results available. Run profile_sample_data() first.")

            return ""

        

        profile = self.profile_results

        

        report = f"""# News Events Dataset - Data Profiling Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}



## Dataset Overview

- **Source Directory**: {self.config['data_paths']['raw_data_dir']}

- **Sample Records Analyzed**: {profile['data_structure']['dimensions']['total_records']:,}

- **Total Columns**: {profile['data_structure']['dimensions']['total_columns']}

- **Memory Usage (Sample)**: {profile['data_structure']['dimensions']['memory_usage_mb']} MB



## Data Structure Analysis



### Column Distribution by Type

- Text Columns: {profile['data_structure']['column_types']['text_columns']}

- Numeric Columns: {profile['data_structure']['column_types']['numeric_columns']}

- Boolean Columns: {profile['data_structure']['column_types']['boolean_columns']}

- DateTime Columns: {profile['data_structure']['column_types']['datetime_columns']}



### Key Columns Identified

"""

        

        # Add column list

        columns = profile['data_structure']['column_list']

        core_columns = [col for col in columns if col in ['event_id', 'summary', 'category', 'confidence', 'company_name', 'found_at']]

        optional_columns = [col for col in columns if col not in core_columns]

        

        report += f"**Core Event Fields**: {', '.join(core_columns)}\n\n"

        report += f"**Extended Fields**: {len(optional_columns)} additional fields including location, amounts, products, etc.\n\n"

        

        # Business insights section

        if 'business_insights' in profile:

            insights = profile['business_insights']

            

            if 'event_categories' in insights:

                categories = insights['event_categories']

                report += f"## Event Categories Analysis\n"

                report += f"- **Total Categories**: {categories['total_categories']}\n"

                report += f"- **Most Common Event Types**:\n"

                for category, count in categories['most_common'].items():

                    report += f"  - {category}: {count} events\n"

                

                if categories['rare_categories']:

                    report += f"- **Rare Categories** (single occurrence): {len(categories['rare_categories'])} categories\n"

            

            if 'company_coverage' in insights:

                companies = insights['company_coverage']

                report += f"\n## Company Coverage Analysis\n"

                report += f"- **Total Companies**: {companies['total_companies']:,}\n"

                report += f"- **Companies with Multiple Events**: {companies['companies_with_multiple_events']:,}\n"

                report += f"- **Single Event Companies**: {companies['single_event_companies']:,}\n"

                report += f"- **Top Active Companies**:\n"

                for company, count in list(companies['most_active_companies'].items())[:5]:

                    report += f"  - {company}: {count} events\n"

            

            if 'confidence_analysis' in insights:

                confidence = insights['confidence_analysis']

                report += f"\n## Confidence Score Analysis\n"

                report += f"- **Mean Confidence**: {confidence['mean_confidence']}\n"

                report += f"- **High Confidence Events** (≥0.9): {confidence['high_confidence_events']:,}\n"

                report += f"- **Medium Confidence Events** (0.7-0.9): {confidence['medium_confidence_events']:,}\n"

                report += f"- **Low Confidence Events** (<0.7): {confidence['low_confidence_events']:,}\n"

            

            if 'temporal_analysis' in insights:

                temporal = insights['temporal_analysis']

                report += f"\n## Temporal Analysis\n"

                report += f"- **Date Range**: {temporal['date_range']['earliest']} to {temporal['date_range']['latest']}\n"

                report += f"- **Time Span**: {temporal['date_range']['span_days']:,} days\n"

                report += f"- **Recent Activity**:\n"

                report += f"  - Last 30 days: {temporal['recent_activity']['last_30_days']:,} events\n"

                report += f"  - Last 90 days: {temporal['recent_activity']['last_90_days']:,} events\n"

                report += f"  - Last year: {temporal['recent_activity']['last_year']:,} events\n"

            

            if 'financial_events' in insights:

                financial = insights['financial_events']

                report += f"\n## Financial Events Analysis\n"

                report += f"- **Total Financial Events**: {financial['total_financial_events']:,}\n"

                report += f"- **Total Amount**: ${financial['total_amount']:,.2f}\n"

                report += f"- **Average Amount**: ${financial['average_amount']:,.2f}\n"

                report += f"- **Median Amount**: ${financial['median_amount']:,.2f}\n"

        

        # Data quality preview

        if 'data_quality_preview' in profile:

            issues = profile['data_quality_preview']

            report += f"\n## Data Quality Preview\n"

            

            if 'completeness_preview' in issues:

                report += f"### Missing Data Analysis\n"

                for field, stats in issues['completeness_preview'].items():

                    if stats['missing_count'] > 0:

                        report += f"- **{field}**: {stats['missing_count']:,} missing ({stats['missing_percentage']}%)\n"

            

            if 'uniqueness_preview' in issues:

                uniq = issues['uniqueness_preview']

                if uniq['duplicate_event_ids'] > 0:

                    report += f"\n### Duplicate Issues Preview\n"

                    report += f"- **Duplicate Event IDs**: {uniq['duplicate_event_ids']:,}\n"

                    report += f"- **Duplicate Summaries**: {uniq['duplicate_summaries']:,}\n"

                    report += f"- **Duplication Rate**: {uniq['potential_duplicates_rate']}%\n"

            

            if 'validity_preview' in issues:

                validity = issues['validity_preview']

                if validity['invalid_confidence_scores'] > 0:

                    report += f"\n### Validity Issues Preview\n"

                    report += f"- **Invalid Confidence Scores**: {validity['invalid_confidence_scores']:,}\n"

                    report += f"- **Out of Range Rate**: {validity['confidence_out_of_range_rate']}%\n"

        

        return report

    

    def save_profile_results(self, output_path: Optional[str] = None) -> str:

        """Save profiling results to JSON file"""

        

        if not self.profile_results:

            logger.error("No profile results to save")

            return ""

        

        if output_path is None:

            output_dir = Path(self.config['data_paths']['processed_data_dir'])

        else:

            output_dir = Path(output_path)

        

        output_dir.mkdir(parents=True, exist_ok=True)

        

        # Save profile as JSON

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        profile_file = output_dir / f"data_profile_{timestamp}.json"

        

        # Convert numpy types to Python native types for JSON serialization

        profile_json = self._convert_for_json(self.profile_results)

        

        with open(profile_file, 'w', encoding='utf-8') as f:

            json.dump(profile_json, f, indent=2, default=str)

        

        # Save report as markdown

        report_file = output_dir / f"profiling_report_{timestamp}.md"

        report_content = self.generate_profiling_report()

        

        with open(report_file, 'w', encoding='utf-8') as f:

            f.write(report_content)

        

        logger.info(f"Profile results saved to: {profile_file}")

        logger.info(f"Profile report saved to: {report_file}")

        

        return str(profile_file)

    

    def _convert_for_json(self, obj):

        """Convert numpy types and other non-JSON-serializable types"""

        if isinstance(obj, dict):

            return {str(key): self._convert_for_json(value) for key, value in obj.items()}

        elif isinstance(obj, list):

            return [self._convert_for_json(item) for item in obj]

        elif isinstance(obj, np.integer):

            return int(obj)

        elif isinstance(obj, np.floating):

            return float(obj)

        elif isinstance(obj, np.bool_):

            return bool(obj)

        elif isinstance(obj, (pd.Timestamp, datetime)):

            return obj.isoformat()

        elif isinstance(obj, pd.Period):

            return str(obj)  # Convert Period objects to strings

        else:

            return obj

    

    def run_complete_profiling(self, sample_size: Optional[int] = None) -> Dict[str, Any]:

        """Run the complete profiling pipeline"""

        

        logger.info("Starting comprehensive data profiling...")

        

        # Step 1: File overview

        logger.info("Step 1: Generating file overview...")

        file_overview = self.generate_file_overview()

        

        # Step 2: Sample data profiling

        logger.info("Step 2: Profiling sample data...")

        sample_profile = self.profile_sample_data(sample_size)

        

        # Step 3: Combine results

        complete_profile = {

            'file_overview': file_overview,

            'sample_analysis': sample_profile,

            'profiling_metadata': {

                'generated_at': datetime.now().isoformat(),

                'sample_size_used': sample_size or self.config['processing']['max_records_sample'],

                'config_used': self.config

            }

        }

        

        # Step 4: Save results

        logger.info("Step 3: Saving profiling results...")

        output_file = self.save_profile_results()

        

        logger.info("Data profiling completed successfully!")

        

        return complete_profile





if __name__ == "__main__":

    # Initialize and run profiling

    profiler = NewsEventsProfiler()

    

    # Run complete profiling

    results = profiler.run_complete_profiling(sample_size=5000)

    

    # Print summary

    print("Profiling completed!")

    print(f"Files analyzed: {results['file_overview']['dataset_summary']['total_files']}")

    print(f"Sample events profiled: {results['sample_analysis']['data_structure']['dimensions']['total_records']:,}")

    print(f"Companies identified: {results['sample_analysis']['business_insights']['company_coverage']['total_companies']:,}")
