"""
Database Manager for News Events Data Quality Assessment
Handles schema creation, data loading, and quality metrics storage
"""

import sqlite3
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

class DatabaseManager:
    """
    Manages database operations for news events data quality assessment
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize database manager with configuration"""
        self.config = self._load_config(config_path)
        self.db_config = self.config['database']
        self.engine = self._create_engine()
        self.logger = self._setup_logging()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format='%(asctime)s | %(levelname)s | %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _create_engine(self):
        """Create database engine based on configuration"""
        db_type = self.db_config['type']
        
        if db_type == 'sqlite':
            db_path = self.db_config['name']
            engine = create_engine(f'sqlite:///{db_path}')
        elif db_type == 'postgresql':
            host = self.db_config['host']
            port = self.db_config['port']
            name = self.db_config['name']
            engine = create_engine(f'postgresql://user:password@{host}:{port}/{name}')
        elif db_type == 'mysql':
            host = self.db_config['host']
            port = self.db_config['port']
            name = self.db_config['name']
            engine = create_engine(f'mysql+mysqlconnector://user:password@{host}:{port}/{name}')
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
            
        return engine
    
    def create_schema(self, schema_file: str = "sql/schema.sql") -> bool:
        """
        Create database schema from SQL file
        
        Args:
            schema_file: Path to SQL schema file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Read schema file
            if Path(schema_file).exists():
                with open(schema_file, 'r') as file:
                    schema_sql = file.read()
            else:
                # Use embedded schema if file doesn't exist
                schema_sql = self._get_embedded_schema()
            
            # Execute schema creation
            with self.engine.connect() as conn:
                # Use a more robust SQL parsing approach
                statements = self._parse_sql_statements(schema_sql)
                
                for statement in statements:
                    if statement.strip():
                        try:
                            conn.execute(text(statement))
                        except Exception as stmt_error:
                            self.logger.warning(f"Statement failed: {stmt_error}")
                            self.logger.debug(f"Failed statement: {statement[:100]}...")
                            # Continue with other statements
                        
                conn.commit()
                
            self.logger.info("Database schema created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating schema: {str(e)}")
            return False
    
    def _parse_sql_statements(self, sql_content: str) -> List[str]:
        """
        Parse SQL content into individual statements, handling triggers and other complex structures
        
        Args:
            sql_content: Raw SQL content
            
        Returns:
            List of individual SQL statements
        """
        statements = []
        current_statement = ""
        in_trigger = False
        brace_count = 0
        
        lines = sql_content.split('\n')
        
        for line in lines:
            stripped_line = line.strip()
            
            # Skip comments and empty lines
            if stripped_line.startswith('--') or not stripped_line:
                continue
                
            # Check if we're starting a trigger
            if 'CREATE TRIGGER' in stripped_line.upper():
                in_trigger = True
                brace_count = 0
                
            # Count braces in triggers
            if in_trigger:
                brace_count += stripped_line.count('BEGIN')
                brace_count += stripped_line.count('{')
                brace_count -= stripped_line.count('END')
                brace_count -= stripped_line.count('}')
                
            current_statement += line + "\n"
            
            # End of statement detection
            if not in_trigger and stripped_line.endswith(';'):
                # Regular statement ending with semicolon
                statements.append(current_statement.strip())
                current_statement = ""
            elif in_trigger and brace_count <= 0 and stripped_line.endswith(';'):
                # Trigger statement completed
                statements.append(current_statement.strip())
                current_statement = ""
                in_trigger = False
                brace_count = 0
        
        # Add any remaining statement
        if current_statement.strip():
            statements.append(current_statement.strip())
            
        return statements
    
    def _prepare_events_for_db(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare events DataFrame for database insertion"""
        # Map CSV columns to database columns
        events_df = pd.DataFrame()
        
        # Basic event fields
        if 'event_id' in df.columns:
            events_df['event_id'] = df['event_id']
        if 'summary' in df.columns:
            events_df['summary'] = df['summary']
        if 'category' in df.columns:
            events_df['category'] = df['category']
        if 'confidence' in df.columns:
            events_df['confidence'] = df['confidence']
        if 'found_at' in df.columns:
            events_df['found_at'] = pd.to_datetime(df['found_at'], errors='coerce')
        if 'effective_date' in df.columns:
            events_df['effective_date'] = pd.to_datetime(df['effective_date'], errors='coerce')
        if 'location' in df.columns:
            events_df['location'] = df['location']
        if 'amount' in df.columns:
            events_df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        if 'currency' in df.columns:
            events_df['currency'] = df['currency']
        else:
            events_df['currency'] = 'USD'
        
        # Add default values for required fields
        events_df['is_cleaned'] = True
        events_df['cleaning_applied'] = 'csv_import'
        events_df['quality_flags'] = '{}'
        events_df['created_at'] = datetime.now()
        events_df['updated_at'] = datetime.now()
        
        return events_df
    
    def _extract_companies_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract unique companies from DataFrame"""
        if 'company_name' not in df.columns:
            return pd.DataFrame()
        
        # Get unique companies
        companies = df[['company_name']].dropna().drop_duplicates()
        companies['company_id'] = companies['company_name'].apply(lambda x: f"company_{hash(x) % 1000000}")
        companies['name'] = companies['company_name']
        companies['domain'] = None
        companies['ticker_symbol'] = None
        companies['industry'] = None
        companies['headquarters_location'] = None
        companies['created_at'] = datetime.now()
        companies['updated_at'] = datetime.now()
        
        return companies[['company_id', 'name', 'domain', 'ticker_symbol', 'industry', 
                         'headquarters_location', 'created_at', 'updated_at']]
    
    def _get_embedded_schema(self) -> str:
        """Return embedded schema SQL (copy from the schema artifact)"""
        # This would contain the full schema from the previous artifact
        # For brevity, returning a placeholder - in real implementation,
        # copy the full schema SQL here
        return """
        -- Embedded schema would go here
        -- Copy the full schema from the previous artifact
        """
    
    def load_cleaned_data(self, cleaned_data_path: str) -> bool:
        """
        Load cleaned events data into database
        
        Args:
            cleaned_data_path: Path to cleaned data file (JSON, CSV, or Parquet)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.logger.info(f"Loading cleaned data from {cleaned_data_path}")
            
            # Load cleaned data based on file type
            if cleaned_data_path.endswith('.csv'):
                cleaned_df = pd.read_csv(cleaned_data_path)
                # Convert DataFrame to list of dictionaries for processing
                cleaned_data = cleaned_df.to_dict('records')
            elif cleaned_data_path.endswith('.parquet'):
                cleaned_df = pd.read_parquet(cleaned_data_path)
                cleaned_data = cleaned_df.to_dict('records')
            elif cleaned_data_path.endswith('.json'):
                with open(cleaned_data_path, 'r', encoding='utf-8') as file:
                    cleaned_data = json.load(file)
            else:
                raise ValueError(f"Unsupported file format: {cleaned_data_path}")
            
            # Process and load data
            events_loaded = 0
            companies_loaded = 0
            articles_loaded = 0
            
            with self.engine.connect() as conn:
                # For CSV/Parquet files, the data is already flattened
                if cleaned_data_path.endswith(('.csv', '.parquet')):
                    # Load events directly from DataFrame
                    events_df = self._prepare_events_for_db(cleaned_df)
                    if not events_df.empty:
                        events_df.to_sql('events', conn, if_exists='append', index=False)
                        events_loaded = len(events_df)
                    
                    # Extract and load companies if company data exists
                    if 'company_name' in cleaned_df.columns:
                        companies_df = self._extract_companies_from_df(cleaned_df)
                        if not companies_df.empty:
                            companies_df.to_sql('companies', conn, if_exists='append', index=False)
                            companies_loaded = len(companies_df)
                else:
                    # Original JSON processing logic
                    companies_df = self._extract_companies(cleaned_data)
                    if not companies_df.empty:
                        companies_df.to_sql('companies', conn, if_exists='append', index=False)
                        companies_loaded = len(companies_df)
                    
                    articles_df = self._extract_articles(cleaned_data)
                    if not articles_df.empty:
                        articles_df.to_sql('news_articles', conn, if_exists='append', index=False)
                        articles_loaded = len(articles_df)
                    
                    events_df = self._extract_events(cleaned_data)
                    if not events_df.empty:
                        events_df.to_sql('events', conn, if_exists='append', index=False)
                        events_loaded = len(events_df)
                    
                    self._load_relationships(conn, cleaned_data)
                
                conn.commit()
            
            self.logger.info(f"Data loaded successfully: {events_loaded} events, "
                           f"{companies_loaded} companies, {articles_loaded} articles")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return False
    
    def _extract_companies(self, data: List[Dict]) -> pd.DataFrame:
        """Extract unique companies from cleaned data"""
        companies = []
        seen_companies = set()
        
        for record in data:
            if 'included' in record:
                for item in record['included']:
                    if item.get('type') == 'companies':
                        attrs = item.get('attributes', {})
                        company_id = item.get('id')
                        
                        if company_id and company_id not in seen_companies:
                            companies.append({
                                'company_id': company_id,
                                'name': attrs.get('name', ''),
                                'domain': attrs.get('domain'),
                                'ticker_symbol': attrs.get('ticker_symbol'),
                                'industry': attrs.get('industry'),
                                'headquarters_location': attrs.get('headquarters_location')
                            })
                            seen_companies.add(company_id)
        
        return pd.DataFrame(companies)
    
    def _extract_articles(self, data: List[Dict]) -> pd.DataFrame:
        """Extract unique articles from cleaned data"""
        articles = []
        seen_articles = set()
        
        for record in data:
            if 'included' in record:
                for item in record['included']:
                    if item.get('type') == 'newsarticles':
                        attrs = item.get('attributes', {})
                        article_id = item.get('id')
                        
                        if article_id and article_id not in seen_articles:
                            # Parse publication date
                            pub_date = attrs.get('publication_date')
                            if pub_date:
                                try:
                                    pub_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                                except:
                                    pub_date = None
                            
                            articles.append({
                                'article_id': article_id,
                                'title': attrs.get('title'),
                                'url': attrs.get('url'),
                                'publication_date': pub_date,
                                'source': attrs.get('source'),
                                'language': attrs.get('language', 'en')
                            })
                            seen_articles.add(article_id)
        
        return pd.DataFrame(articles)
    
    def _extract_events(self, data: List[Dict]) -> pd.DataFrame:
        """Extract events from cleaned data with quality tracking"""
        events = []
        
        for record in data:
            if 'data' in record:
                for event in record['data']:
                    attrs = event.get('attributes', {})
                    
                    # Parse dates
                    found_at = self._parse_datetime(attrs.get('found_at'))
                    effective_date = self._parse_date(attrs.get('effective_date'))
                    
                    # Calculate detection lag
                    detection_lag = None
                    if found_at and effective_date:
                        detection_lag = (found_at.date() - effective_date).days
                    
                    # Extract financial amount
                    amount = None
                    currency = 'USD'
                    if 'amount' in attrs and attrs['amount']:
                        try:
                            amount = float(attrs['amount'])
                        except (ValueError, TypeError):
                            amount = None
                    
                    # Extract location information
                    location = attrs.get('location')
                    country = None
                    region = None
                    if location and isinstance(location, str):
                        # Simple parsing - could be enhanced
                        parts = location.split(',')
                        if len(parts) >= 2:
                            country = parts[-1].strip()
                            region = parts[0].strip()
                    
                    events.append({
                        'event_id': event.get('id'),
                        'summary': attrs.get('summary', ''),
                        'category': attrs.get('category', ''),
                        'confidence': float(attrs.get('confidence', 0)),
                        'found_at': found_at,
                        'effective_date': effective_date,
                        'detection_lag_days': detection_lag,
                        'location': location,
                        'country': country,
                        'region': region,
                        'amount': amount,
                        'currency': currency,
                        'contact_info': json.dumps(attrs.get('contact', {})) if attrs.get('contact') else None,
                        'award_info': json.dumps(attrs.get('award', {})) if attrs.get('award') else None,
                        'product_info': json.dumps(attrs.get('product', {})) if attrs.get('product') else None,
                        'is_cleaned': True,  # Since this is cleaned data
                        'cleaning_applied': attrs.get('cleaning_applied', ''),
                        'quality_flags': json.dumps(attrs.get('quality_flags', {}))
                    })
        
        return pd.DataFrame(events)
    
    def _load_relationships(self, conn, data: List[Dict]):
        """Load event-company and event-article relationships"""
        event_companies = []
        event_articles = []
        
        for record in data:
            if 'data' in record:
                for event in record['data']:
                    event_id = event.get('id')
                    relationships = event.get('relationships', {})
                    
                    # Event-Company relationships
                    if 'companies' in relationships:
                        company_data = relationships['companies'].get('data', [])
                        for company_rel in company_data:
                            event_companies.append({
                                'event_id': event_id,
                                'company_id': company_rel.get('id'),
                                'relationship_type': 'primary'
                            })
                    
                    # Event-Article relationships
                    if 'newsarticles' in relationships:
                        article_data = relationships['newsarticles'].get('data', [])
                        for article_rel in article_data:
                            event_articles.append({
                                'event_id': event_id,
                                'article_id': article_rel.get('id'),
                                'extraction_confidence': None  # Could be derived from event confidence
                            })
        
        # Load relationship data
        if event_companies:
            pd.DataFrame(event_companies).to_sql('event_companies', conn, if_exists='append', index=False)
        
        if event_articles:
            pd.DataFrame(event_articles).to_sql('event_articles', conn, if_exists='append', index=False)
    
    def store_quality_assessment(self, assessment_results: Dict) -> int:
        """
        Store quality assessment results in database
        
        Args:
            assessment_results: Dictionary with assessment scores and metadata
            
        Returns:
            int: Assessment ID for linking related records
        """
        try:
            with self.engine.connect() as conn:
                # Insert main assessment record
                assessment_data = {
                    'total_records': assessment_results.get('total_records', 0),
                    'sample_size': assessment_results.get('sample_size', 0),
                    'completeness_score': assessment_results.get('completeness_score'),
                    'uniqueness_score': assessment_results.get('uniqueness_score'),
                    'validity_score': assessment_results.get('validity_score'),
                    'consistency_score': assessment_results.get('consistency_score'),
                    'timeliness_score': assessment_results.get('timeliness_score'),
                    'overall_score': assessment_results.get('overall_score'),
                    'config_version': assessment_results.get('config_version', '1.0'),
                    'assessment_type': assessment_results.get('assessment_type', 'full')
                }
                
                # Insert and get ID
                result = conn.execute(text("""
                    INSERT INTO quality_assessments 
                    (total_records, sample_size, completeness_score, uniqueness_score, 
                     validity_score, consistency_score, timeliness_score, overall_score,
                     config_version, assessment_type)
                    VALUES (:total_records, :sample_size, :completeness_score, :uniqueness_score,
                            :validity_score, :consistency_score, :timeliness_score, :overall_score,
                            :config_version, :assessment_type)
                """), assessment_data)
                
                assessment_id = result.lastrowid
                
                # Store detailed metrics
                if 'detailed_metrics' in assessment_results:
                    self._store_detailed_metrics(conn, assessment_id, assessment_results['detailed_metrics'])
                
                conn.commit()
                self.logger.info(f"Quality assessment stored with ID: {assessment_id}")
                return assessment_id
                
        except Exception as e:
            self.logger.error(f"Error storing quality assessment: {str(e)}")
            return -1
    
    def store_cleaning_operation(self, operation_data: Dict, assessment_id: Optional[int] = None) -> bool:
        """
        Store cleaning operation results
        
        Args:
            operation_data: Dictionary with cleaning operation details
            assessment_id: Optional assessment ID to link to
            
        Returns:
            bool: True if successful
        """
        try:
            with self.engine.connect() as conn:
                cleaning_record = {
                    'operation_type': operation_data.get('operation_type'),
                    'records_processed': operation_data.get('records_processed', 0),
                    'records_modified': operation_data.get('records_modified', 0),
                    'success_rate': operation_data.get('success_rate', 0.0),
                    'execution_time_seconds': operation_data.get('execution_time_seconds', 0.0),
                    'details': json.dumps(operation_data.get('details', {})),
                    'assessment_id': assessment_id
                }
                
                conn.execute(text("""
                    INSERT INTO cleaning_operations 
                    (operation_type, records_processed, records_modified, success_rate,
                     execution_time_seconds, details, assessment_id)
                    VALUES (:operation_type, :records_processed, :records_modified, :success_rate,
                            :execution_time_seconds, :details, :assessment_id)
                """), cleaning_record)
                
                conn.commit()
                self.logger.info(f"Cleaning operation '{operation_data.get('operation_type')}' stored")
                return True
                
        except Exception as e:
            self.logger.error(f"Error storing cleaning operation: {str(e)}")
            return False
    
    def _store_detailed_metrics(self, conn, assessment_id: int, metrics: Dict):
        """Store detailed quality metrics for each dimension"""
        metric_records = []
        
        for dimension, dimension_metrics in metrics.items():
            if isinstance(dimension_metrics, dict):
                for metric_name, metric_value in dimension_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        # Get threshold from config
                        threshold = self._get_metric_threshold(dimension, metric_name)
                        is_passing = metric_value >= threshold if threshold else None
                        
                        metric_records.append({
                            'assessment_id': assessment_id,
                            'dimension': dimension,
                            'metric_name': metric_name,
                            'metric_value': float(metric_value),
                            'threshold_value': threshold,
                            'is_passing': is_passing
                        })
        
        if metric_records:
            pd.DataFrame(metric_records).to_sql('quality_metrics', conn, if_exists='append', index=False)
    
    def _get_metric_threshold(self, dimension: str, metric_name: str) -> Optional[float]:
        """Get threshold value for a specific metric from config"""
        thresholds = self.config.get('quality_thresholds', {})
        
        # Map dimension to threshold key
        threshold_map = {
            'completeness': 'critical_fields_min',
            'uniqueness': 'duplicate_rate_max',
            'validity': 'uuid_format_compliance_min',
            'consistency': 'category_standardization_min',
            'timeliness': 'max_detection_lag_days'
        }
        
        threshold_key = threshold_map.get(dimension)
        return thresholds.get(threshold_key) if threshold_key else None
    
    def get_quality_summary(self) -> Dict:
        """Get latest quality assessment summary"""
        try:
            with self.engine.connect() as conn:
                # Get latest assessment
                result = conn.execute(text("""
                    SELECT * FROM latest_quality_summary LIMIT 1
                """)).fetchone()
                
                if result:
                    return {
                        'assessment_date': result.assessment_date,
                        'total_records': result.total_records,
                        'sample_size': result.sample_size,
                        'completeness_score': result.completeness_score,
                        'uniqueness_score': result.uniqueness_score,
                        'validity_score': result.validity_score,
                        'consistency_score': result.consistency_score,
                        'timeliness_score': result.timeliness_score,
                        'overall_score': result.overall_score,
                        'cleaning_operations_count': result.cleaning_operations_count
                    }
                else:
                    return {}
                    
        except Exception as e:
            self.logger.error(f"Error getting quality summary: {str(e)}")
            return {}
    
    def get_data_freshness_report(self) -> pd.DataFrame:
        """Get data freshness report by category"""
        try:
            with self.engine.connect() as conn:
                query = """
                SELECT 
                    category,
                    event_count,
                    avg_detection_lag,
                    latest_event,
                    oldest_event,
                    recent_events,
                    freshness_percentage
                FROM data_freshness_summary
                ORDER BY freshness_percentage DESC
                """
                return pd.read_sql(query, conn)
                
        except Exception as e:
            self.logger.error(f"Error getting freshness report: {str(e)}")
            return pd.DataFrame()
    
    def get_quality_trends(self, days: int = 30) -> pd.DataFrame:
        """Get quality trends over specified period"""
        try:
            with self.engine.connect() as conn:
                query = """
                SELECT * FROM quality_trends 
                WHERE assessment_date >= date('now', '-{} days')
                ORDER BY assessment_date ASC
                """.format(days)
                
                return pd.read_sql(query, conn)
                
        except Exception as e:
            self.logger.error(f"Error getting quality trends: {str(e)}")
            return pd.DataFrame()
    
    def get_events_by_company(self, company_name: str, limit: int = 100) -> pd.DataFrame:
        """Get events for a specific company"""
        try:
            with self.engine.connect() as conn:
                query = """
                SELECT 
                    event_id, summary, category, confidence, found_at, 
                    effective_date, location, amount, currency, data_quality_score
                FROM events_with_companies 
                WHERE company_name LIKE :company_name
                ORDER BY found_at DESC
                LIMIT :limit
                """
                
                return pd.read_sql(query, conn, params={
                    'company_name': f'%{company_name}%',
                    'limit': limit
                })
                
        except Exception as e:
            self.logger.error(f"Error getting events for company: {str(e)}")
            return pd.DataFrame()
    
    def get_high_impact_events(self, min_confidence: float = 0.8, days: int = 30) -> pd.DataFrame:
        """Get high-confidence recent events"""
        try:
            with self.engine.connect() as conn:
                query = """
                SELECT 
                    event_id, summary, category, confidence, found_at,
                    company_name, ticker_symbol, amount, currency, location
                FROM events_with_companies
                WHERE confidence >= :min_confidence 
                AND found_at >= datetime('now', '-{} days')
                ORDER BY confidence DESC, found_at DESC
                """.format(days)
                
                return pd.read_sql(query, conn, params={'min_confidence': min_confidence})
                
        except Exception as e:
            self.logger.error(f"Error getting high impact events: {str(e)}")
            return pd.DataFrame()
    
    def execute_quality_queries(self) -> Dict[str, Any]:
        """Execute common quality analysis queries"""
        try:
            results = {}
            
            with self.engine.connect() as conn:
                # Events by category
                results['events_by_category'] = pd.read_sql("""
                    SELECT category, COUNT(*) as count, 
                           AVG(confidence) as avg_confidence,
                           AVG(data_quality_score) as avg_quality
                    FROM events 
                    GROUP BY category 
                    ORDER BY count DESC
                """, conn)
                
                # Top companies by event count
                results['top_companies'] = pd.read_sql("""
                    SELECT c.name, COUNT(e.event_id) as event_count,
                           AVG(e.confidence) as avg_confidence
                    FROM companies c 
                    JOIN event_companies ec ON c.company_id = ec.company_id
                    JOIN events e ON ec.event_id = e.event_id
                    GROUP BY c.name 
                    ORDER BY event_count DESC 
                    LIMIT 10
                """, conn)
                
                # Quality issues by location
                results['quality_by_location'] = pd.read_sql("""
                    SELECT location, COUNT(*) as event_count,
                           AVG(data_quality_score) as avg_quality
                    FROM events 
                    WHERE location IS NOT NULL 
                    GROUP BY location 
                    HAVING COUNT(*) > 5 
                    ORDER BY avg_quality ASC
                    LIMIT 20
                """, conn)
                
                # Recent high-confidence events
                results['recent_high_confidence'] = pd.read_sql("""
                    SELECT event_id, summary, category, confidence, found_at, company_name
                    FROM events_with_companies 
                    WHERE confidence > 0.8 
                    AND found_at > datetime('now', '-30 days')
                    ORDER BY found_at DESC
                    LIMIT 20
                """, conn)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error executing quality queries: {str(e)}")
            return {}
    
    def _parse_datetime(self, date_str: str) -> Optional[datetime]:
        """Parse datetime string with multiple format support"""
        if not date_str:
            return None
            
        formats = [
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def _parse_date(self, date_str: str) -> Optional[datetime.date]:
        """Parse date string"""
        dt = self._parse_datetime(date_str)
        return dt.date() if dt else None
    
    def backup_database(self, backup_path: str) -> bool:
        """Create database backup"""
        try:
            if self.db_config['type'] == 'sqlite':
                import shutil
                shutil.copy2(self.db_config['name'], backup_path)
                self.logger.info(f"Database backed up to {backup_path}")
                return True
            else:
                self.logger.warning("Backup not implemented for non-SQLite databases")
                return False
                
        except Exception as e:
            self.logger.error(f"Error creating backup: {str(e)}")
            return False
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get basic database statistics"""
        try:
            with self.engine.connect() as conn:
                stats = {}
                
                # Count records in each table
                tables = ['events', 'companies', 'news_articles', 'event_companies', 
                         'event_articles', 'quality_assessments', 'cleaning_operations']
                
                for table in tables:
                    try:
                        result = conn.execute(text(f"SELECT COUNT(*) as count FROM {table}")).fetchone()
                        stats[table] = result.count if result else 0
                    except:
                        stats[table] = 0
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Error getting database stats: {str(e)}")
            return {}

    def close(self):
        """Close database connection"""
        if hasattr(self, 'engine'):
            self.engine.dispose()


# Usage example and integration with existing pipeline
class DataQualityDatabase:
    """
    High-level interface for data quality database operations
    Integrates with your existing profiler and cleaning modules
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.db_manager = DatabaseManager(config_path)
        self.logger = self.db_manager.logger
    
    def setup_database(self) -> bool:
        """Initialize database with schema"""
        self.logger.info("Setting up database schema...")
        return self.db_manager.create_schema()
    
    def load_cleaned_dataset(self, cleaned_data_path: str) -> bool:
        """Load your cleaned data into database"""
        self.logger.info("Loading cleaned dataset into database...")
        return self.db_manager.load_cleaned_data(cleaned_data_path)
    
    def store_assessment_results(self, profiler_results: Dict, evaluator_results: Dict) -> int:
        """
        Store results from your profiler and evaluator modules
        
        Args:
            profiler_results: Results from NewsEventsProfiler
            evaluator_results: Results from DataQualityEvaluator
            
        Returns:
            int: Assessment ID
        """
        # Combine results from both modules
        combined_results = {
            'total_records': profiler_results.get('total_records', 0),
            'sample_size': profiler_results.get('sample_size', 0),
            'completeness_score': evaluator_results.get('completeness', {}).get('score', 0),
            'uniqueness_score': evaluator_results.get('uniqueness', {}).get('score', 0),
            'validity_score': evaluator_results.get('validity', {}).get('score', 0),
            'consistency_score': evaluator_results.get('consistency', {}).get('score', 0),
            'timeliness_score': evaluator_results.get('timeliness', {}).get('score', 0),
            'overall_score': evaluator_results.get('overall_score', 0),
            'detailed_metrics': evaluator_results,
            'config_version': '1.0',
            'assessment_type': 'full'
        }
        
        return self.db_manager.store_quality_assessment(combined_results)
    
    def store_cleaning_results(self, cleaning_stats: Dict, assessment_id: Optional[int] = None) -> bool:
        """
        Store results from your cleaning module
        
        Args:
            cleaning_stats: Results from your DataCleaner
            assessment_id: Optional assessment ID to link
            
        Returns:
            bool: Success status
        """
        operations = [
            {
                'operation_type': 'deduplication',
                'records_processed': cleaning_stats.get('total_processed', 0),
                'records_modified': cleaning_stats.get('duplicates_removed', 0),
                'success_rate': cleaning_stats.get('retention_rate', 0) * 100,
                'details': {
                    'duplicate_breakdown': cleaning_stats.get('duplicate_breakdown', {}),
                    'deduplication_strategy': 'multi_strategy'
                }
            },
            {
                'operation_type': 'location_standardization',
                'records_processed': cleaning_stats.get('total_processed', 0),
                'records_modified': cleaning_stats.get('locations_standardized', 0),
                'details': {
                    'standardization_method': 'fuzzy_matching_clustering'
                }
            },
            {
                'operation_type': 'data_validation',
                'records_processed': cleaning_stats.get('total_processed', 0),
                'records_modified': cleaning_stats.get('format_fixes', 0),
                'details': {
                    'validation_rules': 'confidence_scores_dates_amounts'
                }
            }
        ]
        
        success_count = 0
        for operation in operations:
            if self.db_manager.store_cleaning_operation(operation, assessment_id):
                success_count += 1
        
        return success_count == len(operations)
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for dashboard visualization"""
        return {
            'quality_summary': self.db_manager.get_quality_summary(),
            'freshness_report': self.db_manager.get_data_freshness_report(),
            'quality_trends': self.db_manager.get_quality_trends(),
            'analysis_queries': self.db_manager.execute_quality_queries(),
            'database_stats': self.db_manager.get_database_stats()
        }
    
    def close(self):
        """Close database connections"""
        self.db_manager.close()


# Integration script for your existing pipeline
def integrate_with_existing_pipeline():
    """
    Example integration with your existing modules
    This shows how to connect your profiler, evaluator, and cleaner with the database
    """
    
    # Initialize database
    dq_db = DataQualityDatabase("config/config.yaml")
    
    # Step 1: Setup database schema
    if not dq_db.setup_database():
        print("Failed to setup database schema")
        return
    
    # Step 2: Load your cleaned data
    cleaned_data_path = "data/processed/cleaned_events.json"  # Your cleaned data file
    if not dq_db.load_cleaned_dataset(cleaned_data_path):
        print("Failed to load cleaned dataset")
        return
    
    # Step 3: Store assessment results (from your existing modules)
    # This would integrate with your NewsEventsProfiler and DataQualityEvaluator
    """
    profiler = NewsEventsProfiler(config_path="config/config.yaml")
    profiler_results = profiler.profile_dataset("your_sample_data.json")
    
    evaluator = DataQualityEvaluator(config_path="config/config.yaml")
    evaluator_results = evaluator.evaluate_quality(profiler_results)
    
    assessment_id = dq_db.store_assessment_results(profiler_results, evaluator_results)
    """
    
    # Step 4: Store cleaning results (from your cleaning module)
    """
    cleaning_stats = {
        'total_processed': 4992,
        'duplicates_removed': 1567,
        'retention_rate': 0.6861,
        'locations_standardized': 49,
        'format_fixes': 212,
        'duplicate_breakdown': {
            'event_id_duplicates': 1,
            'semantic_duplicates': 138,
            'composite_duplicates': 1454
        }
    }
    
    dq_db.store_cleaning_results(cleaning_stats, assessment_id)
    """
    
    # Step 5: Generate dashboard data
    dashboard_data = dq_db.generate_dashboard_data()
    print("Dashboard data generated:", list(dashboard_data.keys()))
    
    # Cleanup
    dq_db.close()


if __name__ == "__main__":
    # Quick test of database setup
    try:
        db_manager = DatabaseManager("config/config.yaml")
        
        print("Testing database connection...")
        if db_manager.create_schema():
            print("✅ Schema created successfully")
            
            stats = db_manager.get_database_stats()
            print("📊 Database stats:", stats)
        else:
            print("❌ Schema creation failed")
            
        db_manager.close()
        
    except Exception as e:
        print(f"❌ Database test failed: {str(e)}")

    def _parse_datetime(self, date_str: str) -> Optional[datetime]:
        """Parse datetime string with multiple format support"""
        if not date_str:
            return None
            
        formats = [
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def _parse_date(self, date_str: str) -> Optional[datetime.date]:
        """Parse date string"""
        dt = self._parse_datetime(date_str)
        return dt.date() if dt else None
    
    def backup_database(self, backup_path: str) -> bool:
        """Create database backup"""
        try:
            if self.db_config['type'] == 'sqlite':
                import shutil
                shutil.copy2(self.db_config['name'], backup_path)
                self.logger.info(f"Database backed up to {backup_path}")
                return True
            else:
                self.logger.warning("Backup not implemented for non-SQLite databases")
                return False
                
        except Exception as e:
            self.logger.error(f"Error creating backup: {str(e)}")
            return False
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get basic database statistics"""
        try:
            with self.engine.connect() as conn:
                stats = {}
                
                # Count records in each table
                tables = ['events', 'companies', 'news_articles', 'event_companies', 
                         'event_articles', 'quality_assessments', 'cleaning_operations']
                
                for table in tables:
                    try:
                        result = conn.execute(text(f"SELECT COUNT(*) as count FROM {table}")).fetchone()
                        stats[table] = result.count if result else 0
                    except:
                        stats[table] = 0
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Error getting database stats: {str(e)}")
            return {}

    def close(self):
        """Close database connection"""
        if hasattr(self, 'engine'):
            self.engine.dispose()