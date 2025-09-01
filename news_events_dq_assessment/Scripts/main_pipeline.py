"""
News Events Data Quality Assessment - Main Pipeline
Orchestrates the complete data quality pipeline from raw data to database storage
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import your existing modules
try:
    from data_profiling.profiler import NewsEventsProfiler
    from quality_assessment.quality_evaluator import DataQualityEvaluator
    from data_cleaning.data_cleaner import NewsEventsDataCleaner
    from database.database_manager import DatabaseManager, DataQualityDatabase
    from database.data_loader import DataPipelineRunner
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all modules are properly installed and paths are correct")
    sys.exit(1)


class NewsEventsDataQualityPipeline:
    """
    Main pipeline orchestrator for news events data quality assessment
    Coordinates all steps from raw data processing to database storage
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize pipeline with configuration"""
        self.config_path = config_path
        self.logger = self._setup_logging()
        self.pipeline_stats = {
            'start_time': None,
            'end_time': None,
            'steps_completed': [],
            'step_results': {},
            'total_records_processed': 0,
            'final_clean_records': 0,
            'overall_quality_score': 0.0
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "pipeline.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)
    
    def run_complete_pipeline(self, mode: str = "sample") -> Dict[str, Any]:
        """
        Run the complete data quality pipeline
        
        Args:
            mode: "sample" for testing, "full" for complete dataset processing
            
        Returns:
            Dict with complete pipeline results
        """
        
        self.pipeline_stats['start_time'] = datetime.now()
        self.logger.info("=" * 80)
        self.logger.info("STARTING NEWS EVENTS DATA QUALITY ASSESSMENT PIPELINE")
        self.logger.info("=" * 80)
        
        try:
            # Step 1: Data Profiling
            profiling_results = self._run_data_profiling(mode)
            if not profiling_results:
                raise Exception("Data profiling failed")
            
            # Step 2: Quality Assessment
            quality_results = self._run_quality_assessment(profiling_results)
            if not quality_results:
                raise Exception("Quality assessment failed")
            
            # Step 3: Data Cleaning
            cleaning_results = self._run_data_cleaning(profiling_results)
            if not cleaning_results:
                raise Exception("Data cleaning failed")
            
            # Step 4: Database Setup and Loading
            database_results = self._run_database_operations(cleaning_results, quality_results)
            if not database_results:
                raise Exception("Database operations failed")
            
            # Step 5: Generate Final Summary
            self._generate_pipeline_summary()
            
            self.pipeline_stats['end_time'] = datetime.now()
            self.pipeline_stats['duration'] = self.pipeline_stats['end_time'] - self.pipeline_stats['start_time']
            
            self.logger.info("=" * 80)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            
            return self.pipeline_stats
            
        except Exception as e:
            self.logger.error(f"Pipeline failed at step: {len(self.pipeline_stats['steps_completed']) + 1}")
            self.logger.error(f"Error: {str(e)}")
            self.pipeline_stats['error'] = str(e)
            self.pipeline_stats['end_time'] = datetime.now()
            return self.pipeline_stats
    
    def _run_data_profiling(self, mode: str) -> Optional[Dict]:
        """Step 1: Data Profiling"""
        self.logger.info("Step 1: Starting Data Profiling...")
        
        try:
            # Initialize profiler
            profiler = NewsEventsProfiler(self.config_path)
            
            # Determine data source
            if mode == "sample":
                # Use existing cleaned data or run profiling on sample
                results = profiler.generate_comprehensive_profile(sample_size=5000)
            else:
                # Process full dataset
                results = profiler.generate_comprehensive_profile(sample_size=None)
            
            if results:
                self.pipeline_stats['steps_completed'].append('data_profiling')
                self.pipeline_stats['step_results']['profiling'] = results
                self.pipeline_stats['total_records_processed'] = results.get('total_records', 0)
                
                self.logger.info(f"Data profiling completed - {results.get('total_records', 0)} records analyzed")
                return results
            else:
                raise Exception("Profiling returned no results")
                
        except Exception as e:
            self.logger.error(f"Data profiling failed: {str(e)}")
            return None
    
    def _run_quality_assessment(self, profiling_results: Dict) -> Optional[Dict]:
        """Step 2: Quality Assessment"""
        self.logger.info("Step 2: Starting Quality Assessment...")
        
        try:
            # Initialize evaluator
            evaluator = DataQualityEvaluator(self.config_path)
            
            # Run quality evaluation
            quality_results = evaluator.run_comprehensive_assessment(sample_size=5000)
            
            if quality_results:
                self.pipeline_stats['steps_completed'].append('quality_assessment')
                self.pipeline_stats['step_results']['quality'] = quality_results
                self.pipeline_stats['overall_quality_score'] = quality_results.get('overall_score', 0.0)
                
                self.logger.info(f"Quality assessment completed - Overall score: {quality_results.get('overall_score', 0.0):.2f}/100")
                
                # Log key findings
                scores = quality_results.get('dimension_scores', {})
                for dimension, score in scores.items():
                    status = "PASS" if score >= 75 else "FAIL"
                    self.logger.info(f"  {dimension.capitalize()}: {score:.2f}/100 [{status}]")
                
                return quality_results
            else:
                raise Exception("Quality evaluation returned no results")
                
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {str(e)}")
            return None
    
    def _run_data_cleaning(self, profiling_results: Dict) -> Optional[Dict]:
        """Step 3: Data Cleaning"""
        self.logger.info("Step 3: Starting Data Cleaning...")
        
        try:
            # Initialize cleaner
            cleaner = NewsEventsDataCleaner(self.config_path)
            
            # Run cleaning operations
            cleaning_results = cleaner.run_comprehensive_cleaning(sample_size=5000, full_dataset=False)
            
            if cleaning_results:
                self.pipeline_stats['steps_completed'].append('data_cleaning')
                self.pipeline_stats['step_results']['cleaning'] = cleaning_results
                self.pipeline_stats['final_clean_records'] = cleaning_results.get('final_record_count', 0)
                
                self.logger.info(f"Data cleaning completed - {cleaning_results.get('final_record_count', 0)} clean records from {cleaning_results.get('cleaning_stats', {}).get('records_processed', 0)} original")
                self.logger.info(f"Retention rate: {cleaning_results.get('retention_rate', 0):.2f}%")
                
                return cleaning_results
            else:
                raise Exception("Cleaning returned no results")
                
        except Exception as e:
            self.logger.error(f"Data cleaning failed: {str(e)}")
            return None
    
    def _run_database_operations(self, cleaning_results: Dict, quality_results: Dict) -> Optional[Dict]:
        """Step 4: Database Setup and Data Loading"""
        self.logger.info("Step 4: Starting Database Operations...")
        
        try:
            # Initialize database manager
            db_pipeline = DataPipelineRunner(self.config_path)
            
            # Setup database schema
            if not db_pipeline.dq_db.setup_database():
                raise Exception("Failed to setup database schema")
            
            self.logger.info("Database schema created successfully")
            
            # Load cleaned data
            cleaned_data_path = cleaning_results.get('cleaned_data_path')
            if cleaned_data_path and Path(cleaned_data_path).exists():
                if db_pipeline.load_existing_cleaned_data(cleaned_data_path):
                    self.logger.info("Cleaned data loaded into database successfully")
                else:
                    raise Exception("Failed to load cleaned data into database")
            else:
                # Try to find existing cleaned data
                cleaned_data_paths = [
                    "data/processed/cleaned_news_events_*.csv",
                    "data/processed/cleaned_news_events_*.parquet"
                ]
                
                existing_cleaned_data = None
                for path_pattern in cleaned_data_paths:
                    import glob
                    matches = glob.glob(path_pattern)
                    if matches:
                        existing_cleaned_data = max(matches, key=os.path.getctime)
                        break
                
                if existing_cleaned_data:
                    if db_pipeline.load_existing_cleaned_data(existing_cleaned_data):
                        self.logger.info(f"Loaded existing cleaned data: {existing_cleaned_data}")
                    else:
                        raise Exception("Failed to load existing cleaned data")
                else:
                    raise Exception("No cleaned data found to load into database")
            
            # Store quality assessment results
            assessment_summary = {
                'total_records': quality_results['assessment_metadata']['sample_size'],
                'sample_size': quality_results['assessment_metadata']['sample_size'],
                'completeness_score': quality_results['dimension_assessments']['completeness']['overall_score'],
                'uniqueness_score': quality_results['dimension_assessments']['uniqueness']['overall_score'],
                'validity_score': quality_results['dimension_assessments']['validity']['overall_score'],
                'consistency_score': quality_results['dimension_assessments']['consistency']['overall_score'],
                'timeliness_score': quality_results['dimension_assessments']['timeliness']['overall_score'],
                'overall_score': quality_results['overall_quality']['composite_score'],
                'detailed_metrics': quality_results['dimension_assessments']
            }
            
            assessment_id = db_pipeline.dq_db.db_manager.store_quality_assessment(assessment_summary)
            
            if assessment_id > 0:
                self.logger.info(f"Quality assessment stored with ID: {assessment_id}")
            else:
                self.logger.warning("Failed to store quality assessment")
            
            # Store cleaning operation results
            cleaning_stats = {
                'total_processed': cleaning_results.get('cleaning_stats', {}).get('records_processed', 0),
                'duplicates_removed': cleaning_results.get('cleaning_stats', {}).get('duplicates_removed', 0),
                'retention_rate': cleaning_results.get('retention_rate', 0) / 100,
                'locations_standardized': cleaning_results.get('cleaning_stats', {}).get('locations_standardized', 0),
                'format_fixes': cleaning_results.get('cleaning_stats', {}).get('invalid_data_fixed', 0),
                'duplicate_breakdown': {
                    'total_duplicates_removed': cleaning_results.get('cleaning_stats', {}).get('duplicates_removed', 0),
                    'locations_standardized': cleaning_results.get('cleaning_stats', {}).get('locations_standardized', 0),
                    'categories_normalized': cleaning_results.get('cleaning_stats', {}).get('categories_normalized', 0)
                }
            }
            
            if db_pipeline.dq_db.store_cleaning_results(cleaning_stats, assessment_id):
                self.logger.info("Cleaning operation results stored successfully")
            else:
                self.logger.warning("Failed to store cleaning results")
            
            # Verify data integrity
            verification = db_pipeline.verify_data_integrity()
            
            self.pipeline_stats['steps_completed'].append('database_operations')
            self.pipeline_stats['step_results']['database'] = {
                'assessment_id': assessment_id,
                'database_stats': verification.get('database_stats', {}),
                'verification_results': verification
            }
            
            # Close database connection
            db_pipeline.dq_db.close()
            
            return {
                'assessment_id': assessment_id,
                'database_stats': verification.get('database_stats', {}),
                'verification': verification
            }
            
        except Exception as e:
            self.logger.error(f"Database operations failed: {str(e)}")
            return None
    
    def _generate_pipeline_summary(self):
        """Step 5: Generate comprehensive pipeline summary"""
        self.logger.info("Step 5: Generating Pipeline Summary...")
        
        try:
            summary = {
                'pipeline_execution': {
                    'start_time': self.pipeline_stats['start_time'].isoformat(),
                    'end_time': self.pipeline_stats['end_time'].isoformat() if self.pipeline_stats['end_time'] else None,
                    'duration_seconds': self.pipeline_stats.get('duration', {}).total_seconds() if self.pipeline_stats.get('duration') else None,
                    'steps_completed': self.pipeline_stats['steps_completed']
                },
                'data_processing': {
                    'total_records_processed': self.pipeline_stats['total_records_processed'],
                    'final_clean_records': self.pipeline_stats['final_clean_records'],
                    'overall_retention_rate': self.pipeline_stats['final_clean_records'] / max(self.pipeline_stats['total_records_processed'], 1)
                },
                'quality_scores': {
                    'overall_score': self.pipeline_stats['overall_quality_score']
                },
                'step_results': self.pipeline_stats['step_results']
            }
            
            # Save summary report
            reports_dir = Path("data/quality_reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = reports_dir / f"pipeline_summary_{timestamp}.json"
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"Pipeline summary saved to: {summary_file}")
            self.pipeline_stats['steps_completed'].append('summary_generated')
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary: {str(e)}")
    
    def run_database_only(self, cleaned_data_path: str) -> bool:
        """
        Run only database operations (if you already have cleaned data)
        
        Args:
            cleaned_data_path: Path to your existing cleaned data file
            
        Returns:
            bool: Success status
        """
        
        self.logger.info("Running Database-Only Pipeline...")
        self.pipeline_stats['start_time'] = datetime.now()
        
        try:
            # Initialize database pipeline
            db_pipeline = DataPipelineRunner(self.config_path)
            
            # Load existing cleaned data
            if db_pipeline.load_existing_cleaned_data(cleaned_data_path):
                self.pipeline_stats['steps_completed'].append('database_loaded')
                
                # Verify integrity
                verification = db_pipeline.verify_data_integrity()
                self.logger.info(f"Database Stats: {verification['database_stats']}")
                
                db_pipeline.dq_db.close()
                
                self.pipeline_stats['end_time'] = datetime.now()
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Database-only pipeline failed: {str(e)}")
            return False
    
    def run_assessment_only(self, sample_file_path: str) -> Dict[str, Any]:
        """
        Run only profiling and assessment (no cleaning or database)
        
        Args:
            sample_file_path: Path to sample data file
            
        Returns:
            Dict with assessment results
        """
        
        self.logger.info("Running Assessment-Only Pipeline...")
        
        try:
            # Initialize profiler and evaluator
            profiler = NewsEventsProfiler(self.config_path)
            evaluator = DataQualityEvaluator(self.config_path)
            
            # Run profiling
            profiling_results = profiler.generate_comprehensive_profile(sample_size=5000)
            if not profiling_results:
                raise Exception("Profiling failed")
            
            # Run assessment
            quality_results = evaluator.run_comprehensive_assessment(sample_size=5000)
            if not quality_results:
                raise Exception("Assessment failed")
            
            return {
                'profiling': profiling_results,
                'quality': quality_results,
                'overall_score': quality_results.get('overall_quality', {}).get('composite_score', 0.0)
            }
            
        except Exception as e:
            self.logger.error(f"Assessment-only pipeline failed: {str(e)}")
            return {}
    
    def print_pipeline_status(self):
        """Print current pipeline status and results"""
        print("\n" + "=" * 60)
        print("PIPELINE STATUS SUMMARY")
        print("=" * 60)
        
        if self.pipeline_stats['start_time']:
            print(f"Started: {self.pipeline_stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.pipeline_stats['end_time']:
            print(f"Completed: {self.pipeline_stats['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            duration = self.pipeline_stats['end_time'] - self.pipeline_stats['start_time']
            print(f"Duration: {duration}")
        
        print(f"\nSteps Completed: {len(self.pipeline_stats['steps_completed'])}/5")
        for i, step in enumerate(self.pipeline_stats['steps_completed'], 1):
            print(f"  {i}. {step.replace('_', ' ').title()}")
        
        if self.pipeline_stats['total_records_processed']:
            print(f"\nData Processing:")
            print(f"  Original Records: {self.pipeline_stats['total_records_processed']:,}")
            print(f"  Clean Records: {self.pipeline_stats['final_clean_records']:,}")
            
            if self.pipeline_stats['total_records_processed'] > 0:
                retention = self.pipeline_stats['final_clean_records'] / self.pipeline_stats['total_records_processed']
                print(f"  Retention Rate: {retention*100:.2f}%")
        
        if self.pipeline_stats['overall_quality_score']:
            print(f"\nOverall Quality Score: {self.pipeline_stats['overall_quality_score']:.2f}/100")
        
        # Show database stats if available
        db_results = self.pipeline_stats['step_results'].get('database', {})
        if db_results:
            db_stats = db_results.get('database_stats', {})
            if db_stats:
                print(f"\nDatabase Records:")
                for table, count in db_stats.items():
                    if count > 0:
                        print(f"  {table}: {count:,}")
        
        print("=" * 60)


def main():
    """Main execution function with command line argument support"""
    
    parser = argparse.ArgumentParser(description='News Events Data Quality Assessment Pipeline')
    parser.add_argument('--mode', choices=['sample', 'full', 'database-only', 'assessment-only'], 
                       default='sample', help='Pipeline execution mode')
    parser.add_argument('--data-path', type=str, help='Path to data file (for specific modes)')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file path')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = NewsEventsDataQualityPipeline(args.config)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🚀 News Events Data Quality Assessment Pipeline")
    print(f"Mode: {args.mode}")
    print(f"Config: {args.config}")
    print("-" * 50)
    
    # Execute based on mode
    if args.mode == 'sample':
        print("Running complete pipeline in sample mode...")
        results = pipeline.run_complete_pipeline('sample')
        
    elif args.mode == 'full':
        print("Running complete pipeline with full dataset...")
        results = pipeline.run_complete_pipeline('full')
        
    elif args.mode == 'database-only':
        if not args.data_path:
            print("Error: --data-path required for database-only mode")
            print("Example: python main.py --mode database-only --data-path data/processed/cleaned_events.json")
            return
        
        print(f"Loading data into database: {args.data_path}")
        success = pipeline.run_database_only(args.data_path)
        results = {'success': success}
        
    elif args.mode == 'assessment-only':
        data_path = args.data_path or "data/raw/sample_events.json"
        print(f"Running assessment only on: {data_path}")
        results = pipeline.run_assessment_only(data_path)
    
    # Print results
    pipeline.print_pipeline_status()
    
    # Show next steps
    if 'database_loaded' in str(results) or args.mode == 'database-only':
        print("\n🎯 NEXT STEPS:")
        print("1. Run dashboard: python dashboard/app.py")
        print("2. Set up monitoring: python src/monitoring/monitor.py")
        print("3. View database: Open news_events_dq.db in SQLite browser")
    
    return results


def quick_test():
    """Quick test function to verify all modules are working"""
    print("🧪 Running Quick Module Test...")
    
    try:
        # Test imports
        print("Testing imports...")
        pipeline = NewsEventsDataQualityPipeline()
        print("✅ Pipeline initialized successfully")
        
        # Test database connection
        db_manager = DatabaseManager()
        print("✅ Database manager initialized")
        
        # Test schema creation
        if db_manager.create_schema():
            print("✅ Database schema can be created")
        else:
            print("⚠️ Schema creation test failed")
        
        db_manager.close()
        print("✅ All modules working correctly!")
        
    except Exception as e:
        print(f"❌ Module test failed: {str(e)}")
        print("Please check your imports and file paths")


if __name__ == "__main__":
    # You can run different modes:
    
    # Quick test of modules
    if len(sys.argv) == 1:
        print("No arguments provided. Running quick test...")
        quick_test()
        print("\nTo run pipeline:")
        print("python main.py --mode sample              # Test with sample data")
        print("python main.py --mode database-only --data-path data/processed/cleaned_events.json")
        print("python main.py --mode assessment-only     # Just assess data quality")
        print("python main.py --mode full                # Process complete dataset")
    else:
        # Run main pipeline
        main()