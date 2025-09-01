"""
Data Loading Script for News Events Database
Integrates with your existing profiler, evaluator, and cleaner modules
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import yaml

# Import your existing modules (adjust paths as needed)
import sys
import os
from pathlib import Path

# Add src directory to Python path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from data_profiling.profiler import NewsEventsProfiler
from quality_assessment.quality_evaluator import DataQualityEvaluator  
from data_cleaning.data_cleaner import NewsEventsDataCleaner
from database.database_manager import DataQualityDatabase

class DataPipelineRunner:
    """
    Orchestrates the complete pipeline: Profile -> Assess -> Clean -> Load -> Store
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.dq_db = DataQualityDatabase(config_path)
        self.logger = self._setup_logging()
        
        # Initialize pipeline components
        self.profiler = NewsEventsProfiler(config_path)
        self.evaluator = DataQualityEvaluator(config_path)
        self.cleaner = NewsEventsDataCleaner(config_path)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Default configuration if file not found"""
        return {
            'data_paths': {
                'raw_data_dir': "data/raw",
                'processed_data_dir': "data/processed",
                'quality_reports_dir': "data/reports"
            },
            'processing': {
                'sample_size': 5000,
                'chunk_size': 1000
            }
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging with proper configuration"""
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Setup logging with file and console handlers
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers
        if not logger.handlers:
            # File handler
            file_handler = logging.FileHandler(
                log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"
            )
            file_handler.setLevel(logging.INFO)
            
            # Console handler  
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    def run_complete_pipeline(self, sample_size: Optional[int] = None, 
                            process_full_dataset: bool = False) -> Dict[str, Any]:
        """
        Run the complete data quality pipeline: Profile -> Assess -> Clean -> Store
        
        Args:
            sample_size: Number of records to process (None for default)
            process_full_dataset: Whether to process entire dataset
            
        Returns:
            Dict with pipeline results and database status
        """
        
        pipeline_results = {
            'start_time': datetime.now(),
            'steps_completed': [],
            'database_loaded': False,
            'assessment_id': None,
            'profiling_results': {},
            'assessment_results': {},
            'cleaning_results': {},
            'final_stats': {},
            'errors': []
        }
        
        try:
            # Step 1: Setup Database Schema
            self.logger.info("Step 1: Setting up database schema...")
            if self.dq_db.setup_database():
                pipeline_results['steps_completed'].append('database_setup')
                self.logger.info("Database schema created successfully")
            else:
                raise Exception("Failed to create database schema")
            
            # Step 2: Data Profiling
            self.logger.info("Step 2: Running data profiling...")
            try:
                profiling_results = self.profiler.generate_comprehensive_profile(
                    sample_size=sample_size
                )
                pipeline_results['profiling_results'] = profiling_results
                pipeline_results['steps_completed'].append('profiling_completed')
                self.logger.info(f"Data profiling completed: {profiling_results.get('total_records', 0)} records analyzed")
            except Exception as e:
                self.logger.warning(f"Profiling step failed: {str(e)}")
                pipeline_results['errors'].append(f"Profiling failed: {str(e)}")
            
            # Step 3: Quality Assessment
            self.logger.info("Step 3: Running quality assessment...")
            try:
                assessment_results = self.evaluator.run_comprehensive_assessment(
                    sample_size=sample_size
                )
                pipeline_results['assessment_results'] = assessment_results
                pipeline_results['steps_completed'].append('assessment_completed')
                
                # Store assessment in database
                assessment_summary = {
                    'total_records': assessment_results['assessment_metadata']['sample_size'],
                    'sample_size': assessment_results['assessment_metadata']['sample_size'],
                    'completeness_score': assessment_results['dimension_assessments']['completeness']['overall_score'],
                    'uniqueness_score': assessment_results['dimension_assessments']['uniqueness']['overall_score'],
                    'validity_score': assessment_results['dimension_assessments']['validity']['overall_score'],
                    'consistency_score': assessment_results['dimension_assessments']['consistency']['overall_score'],
                    'timeliness_score': assessment_results['dimension_assessments']['timeliness']['overall_score'],
                    'overall_score': assessment_results['overall_quality']['composite_score'],
                    'detailed_metrics': assessment_results['dimension_assessments']
                }
                
                assessment_id = self.dq_db.db_manager.store_quality_assessment(assessment_summary)
                pipeline_results['assessment_id'] = assessment_id
                pipeline_results['steps_completed'].append('assessment_stored')
                
                self.logger.info(f"Quality assessment completed: Overall score {assessment_results['overall_quality']['composite_score']}/100")
                
            except Exception as e:
                self.logger.warning(f"Assessment step failed: {str(e)}")
                pipeline_results['errors'].append(f"Assessment failed: {str(e)}")
            
            # Step 4: Data Cleaning
            self.logger.info("Step 4: Running data cleaning...")
            try:
                cleaning_results = self.cleaner.run_comprehensive_cleaning(
                    sample_size=sample_size,
                    full_dataset=process_full_dataset
                )
                pipeline_results['cleaning_results'] = cleaning_results
                pipeline_results['steps_completed'].append('cleaning_completed')
                
                # Store cleaning results
                cleaning_stats = {
                    'total_processed': cleaning_results['cleaning_stats']['records_processed'],
                    'duplicates_removed': cleaning_results['cleaning_stats']['duplicates_removed'],
                    'retention_rate': cleaning_results['retention_rate'] / 100,
                    'locations_standardized': cleaning_results['cleaning_stats']['locations_standardized'],
                    'format_fixes': cleaning_results['cleaning_stats']['invalid_data_fixed'],
                    'duplicate_breakdown': {
                        'total_duplicates_removed': cleaning_results['cleaning_stats']['duplicates_removed'],
                        'locations_standardized': cleaning_results['cleaning_stats']['locations_standardized'],
                        'categories_normalized': cleaning_results['cleaning_stats']['categories_normalized']
                    }
                }
                
                if self.dq_db.store_cleaning_results(cleaning_stats, pipeline_results.get('assessment_id')):
                    pipeline_results['steps_completed'].append('cleaning_stored')
                    self.logger.info("Cleaning results stored successfully")
                
                self.logger.info(f"Data cleaning completed: {cleaning_results['final_record_count']} clean records")
                
            except Exception as e:
                self.logger.warning(f"Cleaning step failed: {str(e)}")
                pipeline_results['errors'].append(f"Cleaning failed: {str(e)}")
            
            # Step 5: Load Cleaned Data into Database
            self.logger.info("Step 5: Loading cleaned data into database...")
            try:
                if 'cleaned_data_path' in cleaning_results:
                    if self.dq_db.load_cleaned_dataset(cleaning_results['cleaned_data_path']):
                        pipeline_results['database_loaded'] = True
                        pipeline_results['steps_completed'].append('data_loaded')
                        self.logger.info("Cleaned data loaded into database successfully")
                    else:
                        raise Exception("Failed to load cleaned data into database")
                else:
                    self.logger.warning("No cleaned data path available for database loading")
                    
            except Exception as e:
                self.logger.warning(f"Database loading failed: {str(e)}")
                pipeline_results['errors'].append(f"Database loading failed: {str(e)}")
            
            # Step 6: Generate Final Statistics and Reports
            self.logger.info("Step 6: Generating final statistics...")
            try:
                pipeline_results['final_stats'] = self.dq_db.db_manager.get_database_stats()
                pipeline_results['steps_completed'].append('final_stats_generated')
                
                # Generate and save final report
                final_report = self._generate_pipeline_report(pipeline_results)
                report_path = self._save_pipeline_report(final_report)
                pipeline_results['report_path'] = report_path
                
                self.logger.info(f"Pipeline report saved to: {report_path}")
                
            except Exception as e:
                self.logger.warning(f"Final statistics generation failed: {str(e)}")
                pipeline_results['errors'].append(f"Final stats failed: {str(e)}")
            
            pipeline_results['end_time'] = datetime.now()
            pipeline_results['duration'] = pipeline_results['end_time'] - pipeline_results['start_time']
            
            self.logger.info("Complete pipeline execution finished!")
            return pipeline_results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            pipeline_results['error'] = str(e)
            pipeline_results['end_time'] = datetime.now()
            return pipeline_results
    
    def load_existing_cleaned_data(self, cleaned_data_path: str) -> bool:
        """
        Load existing cleaned data file into the database
        
        Args:
            cleaned_data_path: Path to cleaned data file (CSV or JSON)
            
        Returns:
            bool: True if successful
        """
        
        if not Path(cleaned_data_path).exists():
            self.logger.error(f"Cleaned data file not found: {cleaned_data_path}")
            return False
        
        try:
            # Setup database first
            if not self.dq_db.setup_database():
                self.logger.error("Failed to setup database")
                return False
            
            # Load the cleaned data
            if self.dq_db.load_cleaned_dataset(cleaned_data_path):
                self.logger.info("Cleaned data loaded successfully")
                
                # Load actual data to get real statistics
                if cleaned_data_path.endswith('.csv'):
                    cleaned_df = pd.read_csv(cleaned_data_path)
                elif cleaned_data_path.endswith('.json'):
                    with open(cleaned_data_path, 'r') as f:
                        data = json.load(f)
                    cleaned_df = pd.json_normalize(data)
                else:
                    raise ValueError("Unsupported file format. Use CSV or JSON.")
                
                # Calculate real assessment results from the loaded data
                assessment_results = self._calculate_assessment_from_data(cleaned_df)
                
                assessment_id = self.dq_db.db_manager.store_quality_assessment(assessment_results)
                self.logger.info(f"Assessment results stored with ID: {assessment_id}")
                
                # Show final statistics
                stats = self.dq_db.db_manager.get_database_stats()
                self.logger.info(f"Database loaded with: {stats}")
                
                return True
            else:
                self.logger.error("Failed to load cleaned data")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in data loading: {str(e)}")
            return False
    
    def _calculate_assessment_from_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate quality assessment metrics from actual loaded data"""
        
        # Calculate completeness
        critical_fields = ['event_id', 'summary', 'category', 'confidence', 'company_name']
        available_critical = [f for f in critical_fields if f in df.columns]
        
        completeness_scores = []
        for field in available_critical:
            completeness = ((len(df) - df[field].isnull().sum()) / len(df)) * 100
            completeness_scores.append(completeness)
        
        avg_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0
        
        # Calculate uniqueness
        duplicate_rate = 0
        if 'event_id' in df.columns:
            duplicates = df['event_id'].duplicated().sum()
            duplicate_rate = (duplicates / len(df)) * 100
        
        uniqueness_score = max(0, 100 - duplicate_rate)
        
        # Calculate validity (simplified)
        validity_scores = []
        
        if 'confidence' in df.columns:
            valid_confidence = df['confidence'].between(0, 1, inclusive='both').sum()
            confidence_validity = (valid_confidence / len(df)) * 100
            validity_scores.append(confidence_validity)
        
        if 'event_id' in df.columns:
            # Check UUID format
            valid_uuids = 0
            for event_id in df['event_id'].dropna():
                try:
                    import uuid
                    uuid.UUID(str(event_id))
                    valid_uuids += 1
                except:
                    pass
            uuid_validity = (valid_uuids / df['event_id'].notna().sum()) * 100 if df['event_id'].notna().sum() > 0 else 100
            validity_scores.append(uuid_validity)
        
        avg_validity = sum(validity_scores) / len(validity_scores) if validity_scores else 100
        
        # Calculate consistency (simplified)
        consistency_score = 85  # Default - would need full analysis
        
        # Calculate timeliness
        timeliness_score = 50  # Default - would need date analysis
        if 'found_at' in df.columns:
            try:
                dates = pd.to_datetime(df['found_at'], errors='coerce').dropna()
                if len(dates) > 0:
                    current_time = datetime.now()
                    # Simple freshness calculation
                    recent_threshold = pd.Timestamp(current_time) - pd.Timedelta(days=30)
                    recent_count = (dates >= recent_threshold).sum()
                    timeliness_score = (recent_count / len(dates)) * 100
            except:
                pass
        
        # Calculate overall score
        overall_score = (avg_completeness * 0.25 + uniqueness_score * 0.20 + 
                        avg_validity * 0.25 + consistency_score * 0.15 + 
                        timeliness_score * 0.15)
        
        return {
            'total_records': len(df),
            'sample_size': len(df),
            'completeness_score': round(avg_completeness, 2),
            'uniqueness_score': round(uniqueness_score, 2),
            'validity_score': round(avg_validity, 2),
            'consistency_score': round(consistency_score, 2),
            'timeliness_score': round(timeliness_score, 2),
            'overall_score': round(overall_score, 2),
            'detailed_metrics': {
                'completeness': {
                    'critical_fields_completeness': round(avg_completeness, 2),
                    'missing_summaries': int(df['summary'].isnull().sum()) if 'summary' in df.columns else 0,
                    'missing_categories': int(df['category'].isnull().sum()) if 'category' in df.columns else 0,
                    'missing_confidence': int(df['confidence'].isnull().sum()) if 'confidence' in df.columns else 0
                },
                'uniqueness': {
                    'duplicate_rate': round(duplicate_rate, 2),
                    'unique_events': len(df) - (df['event_id'].duplicated().sum() if 'event_id' in df.columns else 0)
                },
                'validity': {
                    'confidence_validity': validity_scores[0] if len(validity_scores) > 0 else 100,
                    'uuid_validity': validity_scores[1] if len(validity_scores) > 1 else 100
                }
            }
        }
    
    def run_assessment_only(self, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Run only the assessment portion of the pipeline
        """
        
        self.logger.info("Running quality assessment only...")
        
        try:
            # Setup database
            self.dq_db.setup_database()
            
            # Run assessment
            assessment_results = self.evaluator.run_comprehensive_assessment(sample_size)
            
            # Store results
            assessment_summary = self._extract_assessment_summary(assessment_results)
            assessment_id = self.dq_db.db_manager.store_quality_assessment(assessment_summary)
            
            return {
                'assessment_id': assessment_id,
                'assessment_results': assessment_results,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Assessment failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def run_cleaning_only(self, sample_size: Optional[int] = None, 
                         full_dataset: bool = False) -> Dict[str, Any]:
        """
        Run only the cleaning portion of the pipeline
        """
        
        self.logger.info("Running data cleaning only...")
        
        try:
            # Run cleaning
            cleaning_results = self.cleaner.run_comprehensive_cleaning(
                sample_size=sample_size,
                full_dataset=full_dataset
            )
            
            return {
                'cleaning_results': cleaning_results,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Cleaning failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _extract_assessment_summary(self, assessment_results: Dict) -> Dict[str, Any]:
        """Extract key metrics from assessment results for database storage"""
        
        dimensions = assessment_results.get('dimension_assessments', {})
        
        return {
            'total_records': assessment_results['assessment_metadata']['sample_size'],
            'sample_size': assessment_results['assessment_metadata']['sample_size'],
            'completeness_score': dimensions.get('completeness', {}).get('overall_score', 0),
            'uniqueness_score': dimensions.get('uniqueness', {}).get('overall_score', 0),
            'validity_score': dimensions.get('validity', {}).get('overall_score', 0),
            'consistency_score': dimensions.get('consistency', {}).get('overall_score', 0),
            'timeliness_score': dimensions.get('timeliness', {}).get('overall_score', 0),
            'overall_score': assessment_results['overall_quality']['composite_score'],
            'detailed_metrics': dimensions
        }
    
    def verify_data_integrity(self) -> Dict[str, Any]:
        """
        Verify the loaded data integrity and generate summary report
        
        Returns:
            Dict with verification results
        """
        
        verification_results = {
            'timestamp': datetime.now(),
            'database_stats': {},
            'data_quality_check': {},
            'sample_records': {},
            'integrity_issues': []
        }
        
        try:
            # Get database statistics
            verification_results['database_stats'] = self.dq_db.db_manager.get_database_stats()
            
            # Run quality queries
            quality_data = self.dq_db.db_manager.execute_quality_queries()
            verification_results['data_quality_check'] = {
                'categories_found': len(quality_data.get('events_by_category', [])),
                'companies_found': len(quality_data.get('top_companies', [])),
                'locations_with_issues': len(quality_data.get('quality_by_location', [])),
                'recent_high_confidence': len(quality_data.get('recent_high_confidence', []))
            }
            
            # Sample records for verification
            with self.dq_db.db_manager.engine.connect() as conn:
                try:
                    # Sample events
                    sample_events = pd.read_sql("""
                        SELECT event_id, summary, category, confidence, found_at, location, amount
                        FROM events ORDER BY confidence DESC LIMIT 5
                    """, conn)
                    verification_results['sample_records']['top_confidence_events'] = sample_events.to_dict('records')
                    
                    # Category distribution
                    category_dist = pd.read_sql("""
                        SELECT category, COUNT(*) as count 
                        FROM events 
                        WHERE category IS NOT NULL
                        GROUP BY category 
                        ORDER BY count DESC
                        LIMIT 10
                    """, conn)
                    verification_results['sample_records']['category_distribution'] = category_dist.to_dict('records')
                
                except Exception as e:
                    self.logger.warning(f"Error running verification queries: {str(e)}")
                    verification_results['integrity_issues'].append(f"Query error: {str(e)}")
            
            self.logger.info("Data integrity verification completed")
            return verification_results
            
        except Exception as e:
            self.logger.error(f"Error in data verification: {str(e)}")
            verification_results['error'] = str(e)
            return verification_results
    
    def _generate_pipeline_report(self, pipeline_results: Dict) -> str:
        """Generate comprehensive pipeline execution report"""
        
        report = f"""# Data Pipeline Execution Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Pipeline Summary
- **Execution Duration**: {pipeline_results.get('duration', 'N/A')}
- **Steps Completed**: {len(pipeline_results['steps_completed'])}/6
- **Database Loaded**: {'Yes' if pipeline_results['database_loaded'] else 'No'}
- **Assessment ID**: {pipeline_results.get('assessment_id', 'N/A')}

## Steps Executed
"""
        
        for i, step in enumerate(pipeline_results['steps_completed'], 1):
            report += f"{i}. {step.replace('_', ' ').title()}\n"
        
        # Add profiling results if available
        if pipeline_results.get('profiling_results'):
            profiling = pipeline_results['profiling_results']
            report += f"""
## Data Profiling Results
- **Total Records Analyzed**: {profiling.get('total_records', 'N/A')}
- **Data Structure**: {profiling.get('data_structure_summary', 'N/A')}
"""
        
        # Add assessment results if available
        if pipeline_results.get('assessment_results'):
            assessment = pipeline_results['assessment_results']['overall_quality']
            report += f"""
## Quality Assessment Results
- **Overall Quality Score**: {assessment['composite_score']}/100
- **Completeness**: {assessment['dimension_scores']['completeness']}/100
- **Uniqueness**: {assessment['dimension_scores']['uniqueness']}/100
- **Validity**: {assessment['dimension_scores']['validity']}/100
- **Consistency**: {assessment['dimension_scores']['consistency']}/100
- **Timeliness**: {assessment['dimension_scores']['timeliness']}/100
"""
        
        # Add cleaning results if available
        if pipeline_results.get('cleaning_results'):
            cleaning = pipeline_results['cleaning_results']
            report += f"""
## Data Cleaning Results
- **Records Processed**: {cleaning['cleaning_stats']['records_processed']:,}
- **Records Retained**: {cleaning['final_record_count']:,}
- **Retention Rate**: {cleaning['retention_rate']:.2f}%
- **Duplicates Removed**: {cleaning['cleaning_stats']['duplicates_removed']:,}
- **Locations Standardized**: {cleaning['cleaning_stats']['locations_standardized']:,}
- **Categories Normalized**: {cleaning['cleaning_stats']['categories_normalized']:,}
"""
        
        # Add errors if any
        if pipeline_results.get('errors'):
            report += "\n## Issues Encountered\n"
            for error in pipeline_results['errors']:
                report += f"- {error}\n"
        
        # Add final database stats
        if pipeline_results.get('final_stats'):
            stats = pipeline_results['final_stats']
            report += f"""
## Final Database Statistics
- **Events Loaded**: {stats.get('events_count', 'N/A')}
- **Companies**: {stats.get('companies_count', 'N/A')}
- **Assessments**: {stats.get('assessments_count', 'N/A')}
- **Cleaning Operations**: {stats.get('cleaning_operations_count', 'N/A')}
"""
        
        return report
    
    def _save_pipeline_report(self, report_content: str) -> str:
        """Save pipeline report to file"""
        
        reports_dir = Path(self.config['data_paths'].get('quality_reports_dir', 'data/reports'))
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = reports_dir / f"pipeline_execution_report_{timestamp}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return str(report_path)
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'dq_db'):
            self.dq_db.close()


def main():
    """
    Main execution function with multiple operation modes
    """
    
    print("News Events Data Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = DataPipelineRunner("config/config.yaml")
    
    try:
        # Check for existing cleaned data
        cleaned_data_paths = [
            "data/processed/cleaned_events.json",
            "data/processed/cleaned_events.csv",
            "data/processed/cleaned_news_events_*.csv",  # Pattern for dated files
            "data/processed/cleaned_news_events_*.parquet"  # Pattern for parquet files
        ]
        
        existing_cleaned_data = None
        for path_pattern in cleaned_data_paths:
            if '*' in path_pattern:
                # Handle glob patterns
                import glob
                matches = glob.glob(path_pattern)
                if matches:
                    # Get most recent file
                    existing_cleaned_data = max(matches, key=os.path.getctime)
                    break
            else:
                if Path(path_pattern).exists():
                    existing_cleaned_data = path_pattern
                    break
        
        if existing_cleaned_data:
            print(f"Found existing cleaned data: {existing_cleaned_data}")
            print("Loading existing cleaned data into database...")
            
            success = pipeline.load_existing_cleaned_data(existing_cleaned_data)
            
            if success:
                print("Data loading completed successfully!")
                
                # Verify data integrity
                print("\nVerifying data integrity...")
                verification = pipeline.verify_data_integrity()
                
                print(f"Database Stats: {verification['database_stats']}")
                print(f"Quality Check: {verification['data_quality_check']}")
                
                # Show sample of loaded data
                if 'sample_records' in verification:
                    print(f"\nSample Categories: {verification['sample_records'].get('category_distribution', [])}")
            else:
                print("Data loading failed")
        
        else:
            print("No existing cleaned data found.")
            print("Running complete pipeline...")
            
            # Run complete pipeline
            results = pipeline.run_complete_pipeline(
                sample_size=5000,
                process_full_dataset=False
            )
            
            print(f"Pipeline completed with {len(results['steps_completed'])} steps")
            if results.get('errors'):
                print(f"Errors encountered: {len(results['errors'])}")
            
            if results.get('report_path'):
                print(f"Full report saved to: {results['report_path']}")
    
    except Exception as e:
        print(f"Pipeline execution failed: {str(e)}")
        logging.error(f"Pipeline execution failed: {str(e)}")
    
    finally:
        # Clean up
        pipeline.close()


if __name__ == "__main__":
    main()