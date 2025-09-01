"""
Data Quality Assessment Module for News Events Dataset
Evaluates data quality across 5 key dimensions with quantitative metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import re
import uuid
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import Counter, defaultdict
from loguru import logger
import yaml
from pathlib import Path
import sys
import os
from difflib import SequenceMatcher
import hashlib

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_loader import NewsEventsDataLoader, flatten_news_event_record


class DataQualityEvaluator:
    """Comprehensive data quality assessment across 5 dimensions"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.loader = NewsEventsDataLoader(self.config['data_paths']['raw_data_dir'])
        self.quality_results = {}
        
        # Define quality dimensions
        self.dimensions = [
            'completeness',
            'uniqueness', 
            'validity',
            'consistency',
            'timeliness'
        ]
        
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
            'quality_thresholds': {
                'completeness': {'critical_fields_min': 95.0, 'optional_fields_min': 70.0},
                'uniqueness': {'duplicate_rate_max': 1.0},
                'validity': {'confidence_score_min': 0.0, 'confidence_score_max': 1.0},
                'consistency': {'category_standardization_min': 95.0},
                'timeliness': {'max_detection_lag_days': 30}
            },
            'critical_fields': ['event_id', 'summary', 'category', 'found_at', 'confidence', 'company_name'],
            'optional_fields': ['location', 'amount', 'effective_date', 'contact', 'award', 'product']
        }
    
    def assess_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess data completeness across critical and optional fields
        
        Definition: Measures the extent to which data is present and not missing
        Business Impact: Missing critical fields can prevent proper event categorization and analysis
        """
        
        logger.info("Assessing completeness dimension...")
        
        critical_fields = self.config.get('critical_fields', [])
        optional_fields = self.config.get('optional_fields', [])
        thresholds = self.config['quality_thresholds']['completeness']
        
        # Analyze critical fields
        critical_analysis = {}
        critical_scores = []
        
        for field in critical_fields:
            if field in df.columns:
                total_count = len(df)
                missing_count = int(df[field].isnull().sum())
                completeness_rate = ((total_count - missing_count) / total_count) * 100
                
                critical_analysis[field] = {
                    'total_records': total_count,
                    'missing_count': missing_count,
                    'present_count': total_count - missing_count,
                    'completeness_percentage': round(completeness_rate, 2),
                    'meets_threshold': completeness_rate >= thresholds['critical_fields_min'],
                    'threshold': thresholds['critical_fields_min']
                }
                critical_scores.append(completeness_rate)
        
        # Analyze optional fields
        optional_analysis = {}
        optional_scores = []
        
        for field in optional_fields:
            if field in df.columns:
                total_count = len(df)
                missing_count = int(df[field].isnull().sum())
                completeness_rate = ((total_count - missing_count) / total_count) * 100
                
                optional_analysis[field] = {
                    'total_records': total_count,
                    'missing_count': missing_count,
                    'present_count': total_count - missing_count,
                    'completeness_percentage': round(completeness_rate, 2),
                    'meets_threshold': completeness_rate >= thresholds['optional_fields_min'],
                    'threshold': thresholds['optional_fields_min']
                }
                optional_scores.append(completeness_rate)
        
        # Overall completeness score
        avg_critical_completeness = np.mean(critical_scores) if critical_scores else 0
        avg_optional_completeness = np.mean(optional_scores) if optional_scores else 0
        overall_completeness = (avg_critical_completeness * 0.7) + (avg_optional_completeness * 0.3)
        
        # Identify most problematic fields
        all_fields_analysis = {**critical_analysis, **optional_analysis}
        problematic_fields = [
            field for field, analysis in all_fields_analysis.items() 
            if not analysis['meets_threshold']
        ]
        
        # Get sample missing records
        missing_examples = self._get_missing_data_examples(df, critical_fields + optional_fields)
        
        return {
            'dimension': 'completeness',
            'overall_score': round(overall_completeness, 2),
            'critical_fields_score': round(avg_critical_completeness, 2),
            'optional_fields_score': round(avg_optional_completeness, 2),
            'meets_quality_standards': avg_critical_completeness >= thresholds['critical_fields_min'],
            'analysis': {
                'critical_fields': critical_analysis,
                'optional_fields': optional_analysis
            },
            'issues_identified': {
                'problematic_fields_count': len(problematic_fields),
                'problematic_fields': problematic_fields,
                'most_incomplete_field': min(all_fields_analysis.items(), 
                                           key=lambda x: x[1]['completeness_percentage'])[0] if all_fields_analysis else None
            },
            'recommendations': self._generate_completeness_recommendations(critical_analysis, optional_analysis),
            'sample_issues': missing_examples
        }
    
    def assess_uniqueness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess data uniqueness and identify duplicates
        
        Definition: Measures the extent to which data values are unique and not duplicated
        Business Impact: Duplicate events can skew analytics and lead to double-counting
        """
        
        logger.info("Assessing uniqueness dimension...")
        
        threshold = self.config['quality_thresholds']['uniqueness']['duplicate_rate_max']
        
        # 1. Event ID duplicates
        event_id_duplicates = self._find_exact_duplicates(df, 'event_id')
        
        # 2. Summary duplicates (semantic duplicates)
        summary_duplicates = self._find_exact_duplicates(df, 'summary')
        
        # 3. Fuzzy duplicate detection on summaries
        fuzzy_duplicates = self._find_fuzzy_duplicates(df, 'summary', similarity_threshold=0.85)
        
        # 4. Multi-field duplicate detection
        composite_duplicates = self._find_composite_duplicates(df, 
            ['company_name', 'category', 'effective_date'])
        
        # Calculate overall duplicate rate
        total_duplicates = max(event_id_duplicates['duplicate_count'], 
                             summary_duplicates['duplicate_count'])
        duplicate_rate = (total_duplicates / len(df)) * 100 if len(df) > 0 else 0
        
        # Uniqueness score (inverse of duplicate rate)
        uniqueness_score = max(0, 100 - duplicate_rate)
        
        return {
            'dimension': 'uniqueness',
            'overall_score': round(uniqueness_score, 2),
            'duplicate_rate_percentage': round(duplicate_rate, 2),
            'meets_quality_standards': duplicate_rate <= threshold,
            'analysis': {
                'event_id_duplicates': event_id_duplicates,
                'summary_duplicates': summary_duplicates,
                'fuzzy_duplicates': fuzzy_duplicates,
                'composite_duplicates': composite_duplicates
            },
            'issues_identified': {
                'total_duplicate_records': int(total_duplicates),
                'highest_duplicate_type': self._identify_worst_duplicate_type(
                    event_id_duplicates, summary_duplicates, fuzzy_duplicates)
            },
            'recommendations': self._generate_uniqueness_recommendations(
                event_id_duplicates, summary_duplicates, fuzzy_duplicates)
        }
    
    def assess_validity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess data validity and format compliance
        
        Definition: Measures the extent to which data conforms to defined formats and business rules
        Business Impact: Invalid data can cause processing errors and unreliable analytics
        """
        
        logger.info("Assessing validity dimension...")
        
        validity_checks = {}
        validity_scores = []
        
        # 1. UUID format validation
        if 'event_id' in df.columns:
            uuid_validation = self._validate_uuid_format(df['event_id'])
            validity_checks['event_id_format'] = uuid_validation
            validity_scores.append(uuid_validation['validity_percentage'])
        
        # 2. Confidence score validation
        if 'confidence' in df.columns:
            confidence_validation = self._validate_confidence_scores(df['confidence'])
            validity_checks['confidence_scores'] = confidence_validation
            validity_scores.append(confidence_validation['validity_percentage'])
        
        # 3. Date format validation
        if 'found_at' in df.columns:
            date_validation = self._validate_date_formats(df['found_at'])
            validity_checks['date_formats'] = date_validation
            validity_scores.append(date_validation['validity_percentage'])
        
        # 4. Category value validation
        if 'category' in df.columns:
            category_validation = self._validate_category_values(df['category'])
            validity_checks['category_values'] = category_validation
            validity_scores.append(category_validation['validity_percentage'])
        
        # 5. Amount validation (for financial events)
        if 'amount_normalized' in df.columns:
            amount_validation = self._validate_amounts(df['amount_normalized'])
            validity_checks['amount_values'] = amount_validation
            validity_scores.append(amount_validation['validity_percentage'])
        
        # Overall validity score
        overall_validity = np.mean(validity_scores) if validity_scores else 0
        
        return {
            'dimension': 'validity',
            'overall_score': round(overall_validity, 2),
            'meets_quality_standards': overall_validity >= 95.0,  # High standard for validity
            'analysis': validity_checks,
            'issues_identified': self._summarize_validity_issues(validity_checks),
            'recommendations': self._generate_validity_recommendations(validity_checks)
        }
    
    def assess_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess data consistency and standardization
        
        Definition: Measures the extent to which data uses consistent formats and values
        Business Impact: Inconsistent data complicates aggregation and analysis
        """
        
        logger.info("Assessing consistency dimension...")
        
        consistency_checks = {}
        consistency_scores = []
        
        # 1. Category standardization
        if 'category' in df.columns:
            category_consistency = self._analyze_category_consistency(df['category'])
            consistency_checks['category_standardization'] = category_consistency
            consistency_scores.append(category_consistency['consistency_percentage'])
        
        # 2. Location standardization
        if 'location' in df.columns:
            location_consistency = self._analyze_location_consistency(df['location'])
            consistency_checks['location_standardization'] = location_consistency
            consistency_scores.append(location_consistency['consistency_percentage'])
        
        # 3. Company name consistency
        if 'company_name' in df.columns:
            company_consistency = self._analyze_company_name_consistency(df['company_name'])
            consistency_checks['company_name_standardization'] = company_consistency
            consistency_scores.append(company_consistency['consistency_percentage'])
        
        # 4. Amount format consistency
        if 'amount' in df.columns:
            amount_consistency = self._analyze_amount_format_consistency(df['amount'])
            consistency_checks['amount_format_consistency'] = amount_consistency
            consistency_scores.append(amount_consistency['consistency_percentage'])
        
        # Overall consistency score
        overall_consistency = np.mean(consistency_scores) if consistency_scores else 0
        threshold = self.config['quality_thresholds']['consistency']['category_standardization_min']
        
        return {
            'dimension': 'consistency',
            'overall_score': round(overall_consistency, 2),
            'meets_quality_standards': overall_consistency >= threshold,
            'analysis': consistency_checks,
            'issues_identified': self._summarize_consistency_issues(consistency_checks),
            'recommendations': self._generate_consistency_recommendations(consistency_checks)
        }
    
    def assess_timeliness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess data timeliness and freshness
        
        Definition: Measures how current and up-to-date the data is
        Business Impact: Stale data can lead to missed opportunities and outdated insights
        """
        
        logger.info("Assessing timeliness dimension...")
        
        timeliness_analysis = {}
        current_time = datetime.now(timezone.utc)
        
        # 1. Detection lag analysis (found_at vs published_at)
        if 'found_at' in df.columns and 'article_published_at' in df.columns:
            lag_analysis = self._analyze_detection_lag(df, current_time)
            timeliness_analysis['detection_lag'] = lag_analysis
        
        # 2. Data freshness analysis
        if 'found_at' in df.columns:
            freshness_analysis = self._analyze_data_freshness(df, current_time)
            timeliness_analysis['data_freshness'] = freshness_analysis
        
        # 3. Effective date timeliness
        if 'effective_date' in df.columns:
            effective_date_analysis = self._analyze_effective_date_timeliness(df, current_time)
            timeliness_analysis['effective_date_timeliness'] = effective_date_analysis
        
        # Calculate overall timeliness score
        timeliness_score = self._calculate_timeliness_score(timeliness_analysis, current_time)
        threshold_days = self.config['quality_thresholds']['timeliness']['max_detection_lag_days']
        
        return {
            'dimension': 'timeliness',
            'overall_score': round(timeliness_score, 2),
            'meets_quality_standards': timeliness_score >= 80.0,  # 80% threshold for timeliness
            'analysis': timeliness_analysis,
            'issues_identified': self._summarize_timeliness_issues(timeliness_analysis),
            'recommendations': self._generate_timeliness_recommendations(timeliness_analysis)
        }
    
    def run_comprehensive_assessment(self, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """Run complete data quality assessment across all dimensions"""
        
        logger.info("Starting comprehensive data quality assessment...")
        
        # Load and prepare data
        if sample_size is None:
            sample_size = self.config['processing']['max_records_sample']
        
        sample_records = self.loader.load_sample_data(sample_size)
        
        # Flatten data
        flattened_data = []
        for item in sample_records:
            try:
                events = flatten_news_event_record(item['record'])
                for event in events:
                    event['source_file'] = item['source_file']
                    flattened_data.append(event)
            except Exception as e:
                logger.warning(f"Error flattening record: {e}")
                continue
        
        df = pd.DataFrame(flattened_data)
        logger.info(f"Assessing quality for {len(df)} events")
        
        # Run assessments for each dimension
        assessment_results = {}
        
        # 1. Completeness Assessment
        assessment_results['completeness'] = self.assess_completeness(df)
        
        # 2. Uniqueness Assessment
        assessment_results['uniqueness'] = self.assess_uniqueness(df)
        
        # 3. Validity Assessment
        assessment_results['validity'] = self.assess_validity(df)
        
        # 4. Consistency Assessment
        assessment_results['consistency'] = self.assess_consistency(df)
        
        # 5. Timeliness Assessment
        assessment_results['timeliness'] = self.assess_timeliness(df)
        
        # Calculate overall data quality score
        overall_score = self._calculate_overall_quality_score(assessment_results)
        
        # Generate final results
        final_results = {
            'assessment_metadata': {
                'assessment_date': datetime.now().isoformat(),
                'sample_size': len(df),
                'total_files_analyzed': self.loader.total_files,
                'assessment_scope': 'sample_analysis'
            },
            'overall_quality': {
                'composite_score': overall_score,
                'dimension_scores': {
                    dim: results['overall_score'] 
                    for dim, results in assessment_results.items()
                },
                'standards_compliance': {
                    dim: results['meets_quality_standards'] 
                    for dim, results in assessment_results.items()
                }
            },
            'dimension_assessments': assessment_results,
            'priority_issues': self._identify_priority_issues(assessment_results),
            'improvement_roadmap': self._generate_improvement_roadmap(assessment_results)
        }
        
        self.quality_results = final_results
        logger.info(f"Assessment completed. Overall quality score: {overall_score}")
        
        return final_results
    
    # Helper methods for each dimension
    
    def _find_exact_duplicates(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Find exact duplicate values in a column"""
        if column not in df.columns:
            return {'duplicate_count': 0, 'unique_count': 0, 'examples': []}
        
        col_data = df[column].dropna()
        duplicates = col_data[col_data.duplicated(keep=False)]
        
        return {
            'duplicate_count': int(len(duplicates)),
            'unique_count': int(col_data.nunique()),
            'duplicate_percentage': round((len(duplicates) / len(col_data)) * 100, 2) if len(col_data) > 0 else 0,
            'examples': duplicates.value_counts().head(5).to_dict()
        }
    
    def _find_fuzzy_duplicates(self, df: pd.DataFrame, column: str, similarity_threshold: float = 0.85) -> Dict[str, Any]:
        """Find fuzzy duplicates using string similarity"""
        if column not in df.columns:
            return {'fuzzy_duplicate_pairs': 0, 'examples': []}
        
        text_data = df[column].dropna().astype(str)
        fuzzy_pairs = []
        
        # Sample for performance (limit to 1000 records for fuzzy matching)
        if len(text_data) > 1000:
            text_sample = text_data.sample(1000, random_state=42)
        else:
            text_sample = text_data
        
        unique_texts = text_sample.unique()
        
        for i, text1 in enumerate(unique_texts):
            for text2 in unique_texts[i+1:]:
                similarity = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
                if similarity >= similarity_threshold:
                    fuzzy_pairs.append({
                        'text1': text1[:100] + '...' if len(text1) > 100 else text1,
                        'text2': text2[:100] + '...' if len(text2) > 100 else text2,
                        'similarity': round(similarity, 3)
                    })
        
        return {
            'fuzzy_duplicate_pairs': len(fuzzy_pairs),
            'examples': fuzzy_pairs[:5],  # Top 5 examples
            'similarity_threshold_used': similarity_threshold
        }
    
    def _find_composite_duplicates(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """Find duplicates based on combination of columns"""
        available_columns = [col for col in columns if col in df.columns]
        
        if not available_columns:
            return {'composite_duplicate_count': 0, 'examples': []}
        
        # Create composite key
        df_temp = df[available_columns].copy()
        df_temp = df_temp.fillna('NULL')  # Replace NaN with string for grouping
        
        # Create hash of combined values
        df_temp['composite_key'] = df_temp.apply(
            lambda row: hashlib.md5('|'.join(str(row[col]) for col in available_columns).encode()).hexdigest(), 
            axis=1
        )
        
        duplicates = df_temp[df_temp['composite_key'].duplicated(keep=False)]
        
        return {
            'composite_duplicate_count': int(len(duplicates)),
            'fields_analyzed': available_columns,
            'duplicate_groups': duplicates['composite_key'].value_counts().head(5).to_dict()
        }
    
    def _validate_uuid_format(self, series: pd.Series) -> Dict[str, Any]:
        """Validate UUID format compliance"""
        non_null_data = series.dropna()
        
        valid_uuids = 0
        invalid_examples = []
        
        for value in non_null_data:
            try:
                uuid.UUID(str(value))
                valid_uuids += 1
            except (ValueError, TypeError):
                if len(invalid_examples) < 5:
                    invalid_examples.append(str(value))
        
        validity_percentage = (valid_uuids / len(non_null_data)) * 100 if len(non_null_data) > 0 else 0
        
        return {
            'total_checked': len(non_null_data),
            'valid_count': valid_uuids,
            'invalid_count': len(non_null_data) - valid_uuids,
            'validity_percentage': round(validity_percentage, 2),
            'invalid_examples': invalid_examples
        }
    
    def _validate_confidence_scores(self, series: pd.Series) -> Dict[str, Any]:
        """Validate confidence scores are between 0 and 1"""
        non_null_data = series.dropna()
        
        valid_range = non_null_data.between(0, 1, inclusive='both')
        valid_count = int(valid_range.sum())
        invalid_values = non_null_data[~valid_range]
        
        validity_percentage = (valid_count / len(non_null_data)) * 100 if len(non_null_data) > 0 else 0
        
        return {
            'total_checked': len(non_null_data),
            'valid_count': valid_count,
            'invalid_count': len(invalid_values),
            'validity_percentage': round(validity_percentage, 2),
            'invalid_examples': invalid_values.head(5).tolist(),
            'out_of_range_stats': {
                'below_zero': int((non_null_data < 0).sum()),
                'above_one': int((non_null_data > 1).sum())
            }
        }
    
    def _validate_date_formats(self, series: pd.Series) -> Dict[str, Any]:
        """Validate ISO date format compliance"""
        non_null_data = series.dropna()
        
        valid_dates = pd.to_datetime(non_null_data, errors='coerce')
        valid_count = int(valid_dates.notna().sum())
        invalid_examples = non_null_data[valid_dates.isna()].head(5).tolist()
        
        validity_percentage = (valid_count / len(non_null_data)) * 100 if len(non_null_data) > 0 else 0
        
        return {
            'total_checked': len(non_null_data),
            'valid_count': valid_count,
            'invalid_count': len(non_null_data) - valid_count,
            'validity_percentage': round(validity_percentage, 2),
            'invalid_examples': invalid_examples
        }
    
    def _validate_category_values(self, series: pd.Series) -> Dict[str, Any]:
        """Validate category values against expected business categories"""
        
        # Define expected categories based on sample data patterns
        expected_categories = {
            'receives_award', 'recognized_as', 'launches', 'hires', 'receives_financing',
            'is_developing', 'announces', 'partners_with', 'acquires', 'expands'
        }
        
        non_null_data = series.dropna()
        category_counts = non_null_data.value_counts()
        
        valid_categories = set(category_counts.index) & expected_categories
        invalid_categories = set(category_counts.index) - expected_categories
        
        valid_count = int(non_null_data.isin(valid_categories).sum())
        validity_percentage = (valid_count / len(non_null_data)) * 100 if len(non_null_data) > 0 else 0
        
        return {
            'total_checked': len(non_null_data),
            'valid_count': valid_count,
            'invalid_count': len(non_null_data) - valid_count,
            'validity_percentage': round(validity_percentage, 2),
            'expected_categories': list(expected_categories),
            'found_categories': list(category_counts.index),
            'invalid_categories': list(invalid_categories),
            'category_distribution': category_counts.to_dict()
        }
    
    def _validate_amounts(self, series: pd.Series) -> Dict[str, Any]:
        """Validate amount values are positive numbers"""
        non_null_data = series.dropna()
        
        # Check for numeric type and positive values
        valid_amounts = non_null_data[(non_null_data >= 0) & pd.to_numeric(non_null_data, errors='coerce').notna()]
        valid_count = len(valid_amounts)
        
        validity_percentage = (valid_count / len(non_null_data)) * 100 if len(non_null_data) > 0 else 0
        
        invalid_values = non_null_data[
            (non_null_data < 0) | pd.to_numeric(non_null_data, errors='coerce').isna()
        ]
        
        return {
            'total_checked': len(non_null_data),
            'valid_count': valid_count,
            'invalid_count': len(invalid_values),
            'validity_percentage': round(validity_percentage, 2),
            'invalid_examples': invalid_values.head(5).tolist(),
            'negative_amounts': int((non_null_data < 0).sum())
        }
    
    def _analyze_category_consistency(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze consistency of category values"""
        non_null_data = series.dropna()
        category_counts = non_null_data.value_counts()
        
        # Look for potential inconsistencies (similar category names)
        categories = list(category_counts.index)
        potential_inconsistencies = []
        
        for i, cat1 in enumerate(categories):
            for cat2 in categories[i+1:]:
                similarity = SequenceMatcher(None, cat1.lower(), cat2.lower()).ratio()
                if 0.7 <= similarity < 1.0:  # Similar but not identical
                    potential_inconsistencies.append({
                        'category1': cat1,
                        'category2': cat2,
                        'similarity': round(similarity, 3),
                        'count1': category_counts[cat1],
                        'count2': category_counts[cat2]
                    })
        
        # Calculate consistency score
        # Higher score for fewer similar categories
        consistency_percentage = max(0, 100 - (len(potential_inconsistencies) * 5))
        
        return {
            'total_categories': len(categories),
            'consistency_percentage': round(consistency_percentage, 2),
            'potential_inconsistencies': potential_inconsistencies[:10],  # Top 10
            'category_distribution': category_counts.to_dict(),
            'standardization_opportunities': len(potential_inconsistencies)
        }
    
    def _analyze_location_consistency(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze consistency of location values"""
        non_null_data = series.dropna()
        location_counts = non_null_data.value_counts()
        
        # Look for potential location variations
        locations = list(location_counts.index)
        location_variations = []
        
        for i, loc1 in enumerate(locations):
            for loc2 in locations[i+1:]:
                # Check for similar location names
                if self._are_similar_locations(str(loc1), str(loc2)):
                    location_variations.append({
                        'location1': loc1,
                        'location2': loc2,
                        'count1': location_counts[loc1],
                        'count2': location_counts[loc2]
                    })
        
        consistency_percentage = max(0, 100 - (len(location_variations) * 3))
        
        return {
            'total_locations': len(locations),
            'consistency_percentage': round(consistency_percentage, 2),
            'location_variations': location_variations[:10],
            'top_locations': location_counts.head(10).to_dict(),
            'standardization_opportunities': len(location_variations)
        }
    
    def _analyze_company_name_consistency(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze consistency of company names"""
        non_null_data = series.dropna()
        company_counts = non_null_data.value_counts()
        
        # Look for potential company name variations
        companies = list(company_counts.index)
        name_variations = []
        
        for i, comp1 in enumerate(companies):
            for comp2 in companies[i+1:]:
                if self._are_similar_company_names(str(comp1), str(comp2)):
                    name_variations.append({
                        'company1': comp1,
                        'company2': comp2,
                        'count1': company_counts[comp1],
                        'count2': company_counts[comp2]
                    })
        
        consistency_percentage = max(0, 100 - (len(name_variations) * 2))
        
        return {
            'total_companies': len(companies),
            'consistency_percentage': round(consistency_percentage, 2),
            'name_variations': name_variations[:10],
            'top_companies': company_counts.head(10).to_dict(),
            'standardization_opportunities': len(name_variations)
        }
    
    def _analyze_amount_format_consistency(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze consistency of amount formats"""
        non_null_data = series.dropna().astype(str)
        
        # Pattern analysis for amount formats
        format_patterns = {
            'currency_symbol': 0,
            'comma_separated': 0,
            'decimal_places': 0,
            'scientific_notation': 0,
            'plain_number': 0
        }
        
        for value in non_null_data:
            if '$' in value or '€' in value or '£' in value:
                format_patterns['currency_symbol'] += 1
            if ',' in value:
                format_patterns['comma_separated'] += 1
            if '.' in value and value.count('.') == 1:
                format_patterns['decimal_places'] += 1
            if 'e' in value.lower() or 'E' in value:
                format_patterns['scientific_notation'] += 1
            if re.match(r'^\d+$', value.replace(',', '').replace('.', '')):
                format_patterns['plain_number'] += 1
        
        # Calculate consistency score based on format uniformity
        total_values = len(non_null_data)
        most_common_format_count = max(format_patterns.values()) if format_patterns.values() else 0
        consistency_percentage = (most_common_format_count / total_values) * 100 if total_values > 0 else 0
        
        return {
            'total_checked': total_values,
            'consistency_percentage': round(consistency_percentage, 2),
            'format_patterns': format_patterns,
            'format_examples': non_null_data.head(5).tolist()
        }
    
    def _analyze_detection_lag(self, df: pd.DataFrame, current_time: datetime) -> Dict[str, Any]:
        """Analyze lag between article publication and event detection"""
        
        # Convert date columns
        df_temp = df.copy()
        df_temp['found_at_dt'] = pd.to_datetime(df_temp['found_at'], errors='coerce')
        df_temp['published_at_dt'] = pd.to_datetime(df_temp['article_published_at'], errors='coerce')
        
        # Calculate lag where both dates are available
        valid_data = df_temp[
            df_temp['found_at_dt'].notna() & df_temp['published_at_dt'].notna()
        ].copy()
        
        if len(valid_data) == 0:
            return {'analysis_possible': False, 'reason': 'No valid date pairs found'}
        
        valid_data['detection_lag_hours'] = (
            valid_data['found_at_dt'] - valid_data['published_at_dt']
        ).dt.total_seconds() / 3600
        
        lag_data = valid_data['detection_lag_hours']
        
        return {
            'analysis_possible': True,
            'total_analyzed': len(valid_data),
            'lag_statistics': {
                'mean_lag_hours': round(float(lag_data.mean()), 2),
                'median_lag_hours': round(float(lag_data.median()), 2),
                'max_lag_hours': round(float(lag_data.max()), 2),
                'min_lag_hours': round(float(lag_data.min()), 2)
            },
            'lag_distribution': {
                'real_time_0_1h': int((lag_data <= 1).sum()),
                'fast_1_24h': int(((lag_data > 1) & (lag_data <= 24)).sum()),
                'delayed_1_7d': int(((lag_data > 24) & (lag_data <= 168)).sum()),
                'slow_over_7d': int((lag_data > 168).sum()),
                'negative_lag': int((lag_data < 0).sum())  # Found before published
            }
        }
    
    def _analyze_data_freshness(self, df: pd.DataFrame, current_time: datetime) -> Dict[str, Any]:
        """Analyze how fresh/recent the data is"""
        
        df_temp = df.copy()
        df_temp['found_at_dt'] = pd.to_datetime(df_temp['found_at'], errors='coerce')
        
        valid_dates = df_temp['found_at_dt'].dropna()
        
        if len(valid_dates) == 0:
            return {'analysis_possible': False, 'reason': 'No valid found_at dates'}
        
        # Convert current_time to timezone-naive
        current_time_naive = current_time.replace(tzinfo=None) if current_time.tzinfo else current_time
        
        # Ensure all dates are timezone-naive by converting the pandas series
        valid_dates_naive = valid_dates.dt.tz_localize(None) if valid_dates.dt.tz is not None else valid_dates
        
        # Calculate age of data using pandas datetime arithmetic
        data_ages = (pd.Timestamp(current_time_naive) - valid_dates_naive).dt.days
        
        freshness_distribution = {
            'last_7_days': int((data_ages <= 7).sum()),
            'last_30_days': int((data_ages <= 30).sum()),
            'last_90_days': int((data_ages <= 90).sum()),
            'last_year': int((data_ages <= 365).sum()),
            'over_1_year': int((data_ages > 365).sum())
        }
        
        # Calculate freshness score
        recent_data_percentage = (freshness_distribution['last_30_days'] / len(valid_dates)) * 100
        
        return {
            'analysis_possible': True,
            'total_analyzed': len(valid_dates),
            'freshness_score': round(recent_data_percentage, 2),
            'data_age_stats': {
                'newest_days_ago': int(data_ages.min()),
                'oldest_days_ago': int(data_ages.max()),
                'mean_age_days': round(float(data_ages.mean()), 1),
                'median_age_days': int(data_ages.median())
            },
            'freshness_distribution': freshness_distribution
        }

    def _analyze_effective_date_timeliness(self, df: pd.DataFrame, current_time: datetime) -> Dict[str, Any]:
        """Analyze timeliness of effective dates"""
        
        df_temp = df.copy()
        df_temp['effective_date_dt'] = pd.to_datetime(df_temp['effective_date'], errors='coerce')
        
        valid_dates = df_temp['effective_date_dt'].dropna()
        
        if len(valid_dates) == 0:
            return {'analysis_possible': False, 'reason': 'No valid effective dates'}
        
        # Convert current_time to timezone-naive for comparison
        current_time_naive = current_time.replace(tzinfo=None) if current_time.tzinfo else current_time
        
        # Analyze effective date patterns
        future_dates = int((valid_dates > current_time_naive).sum())
        past_dates = int((valid_dates <= current_time_naive).sum())
        
        return {
            'analysis_possible': True,
            'total_analyzed': len(valid_dates),
            'future_events': future_dates,
            'past_events': past_dates,
            'future_percentage': round((future_dates / len(valid_dates)) * 100, 2)
        }
    
    def _calculate_timeliness_score(self, timeliness_analysis: Dict, current_time: datetime) -> float:
        """Calculate overall timeliness score"""
        scores = []
        
        if 'data_freshness' in timeliness_analysis and timeliness_analysis['data_freshness'].get('analysis_possible'):
            freshness_score = timeliness_analysis['data_freshness']['freshness_score']
            scores.append(freshness_score)
        
        if 'detection_lag' in timeliness_analysis and timeliness_analysis['detection_lag'].get('analysis_possible'):
            lag_analysis = timeliness_analysis['detection_lag']
            # Score based on real-time and fast detection rates
            fast_detection_rate = (
                lag_analysis['lag_distribution']['real_time_0_1h'] + 
                lag_analysis['lag_distribution']['fast_1_24h']
            ) / lag_analysis['total_analyzed'] * 100
            scores.append(fast_detection_rate)
        
        return np.mean(scores) if scores else 50.0  # Default middle score if no analysis possible
    
    # Utility helper methods
    
    def _are_similar_locations(self, loc1: str, loc2: str) -> bool:
        """Check if two location strings are similar"""
        # Simple similarity check for locations
        similarity = SequenceMatcher(None, loc1.lower(), loc2.lower()).ratio()
        return 0.8 <= similarity < 1.0
    
    def _are_similar_company_names(self, comp1: str, comp2: str) -> bool:
        """Check if two company names are similar"""
        # Remove common suffixes for comparison
        suffixes = ['Inc.', 'Ltd.', 'LLC', 'Corp.', 'Corporation', 'Limited', 'Company', 'Co.']
        
        clean_comp1 = comp1
        clean_comp2 = comp2
        
        for suffix in suffixes:
            clean_comp1 = re.sub(rf'\b{re.escape(suffix)}\b', '', clean_comp1, flags=re.IGNORECASE).strip()
            clean_comp2 = re.sub(rf'\b{re.escape(suffix)}\b', '', clean_comp2, flags=re.IGNORECASE).strip()
        
        similarity = SequenceMatcher(None, clean_comp1.lower(), clean_comp2.lower()).ratio()
        return 0.85 <= similarity < 1.0
    
    def _get_missing_data_examples(self, df: pd.DataFrame, fields: List[str]) -> List[Dict]:
        """Get examples of records with missing data"""
        examples = []
        available_fields = [field for field in fields if field in df.columns]
        
        for field in available_fields:
            missing_mask = df[field].isnull()
            if missing_mask.any():
                sample_record = df[missing_mask].iloc[0]
                examples.append({
                    'missing_field': field,
                    'event_id': sample_record.get('event_id'),
                    'available_data': {
                        k: v for k, v in sample_record.to_dict().items() 
                        if pd.notna(v) and k in available_fields[:5]  # Limit to first 5 fields
                    }
                })
        
        return examples[:5]  # Limit to 5 examples
    
    def _identify_worst_duplicate_type(self, *duplicate_analyses) -> str:
        """Identify the worst type of duplication"""
        max_duplicates = 0
        worst_type = "none"
        
        type_names = ['event_id', 'summary', 'fuzzy_similarity']
        
        for i, analysis in enumerate(duplicate_analyses):
            if isinstance(analysis, dict):
                duplicate_count = analysis.get('duplicate_count', 0)
                if duplicate_count > max_duplicates:
                    max_duplicates = duplicate_count
                    worst_type = type_names[i] if i < len(type_names) else f"type_{i}"
        
        return worst_type
    
    def _calculate_overall_quality_score(self, assessment_results: Dict) -> float:
        """Calculate weighted overall data quality score"""
        
        # Define weights for each dimension
        weights = {
            'completeness': 0.25,  # Most critical
            'validity': 0.25,      # Most critical  
            'uniqueness': 0.20,    # High importance
            'consistency': 0.15,   # Medium importance
            'timeliness': 0.15     # Medium importance
        }
        
        weighted_score = 0
        total_weight = 0
        
        for dimension, weight in weights.items():
            if dimension in assessment_results:
                score = assessment_results[dimension]['overall_score']
                weighted_score += score * weight
                total_weight += weight
        
        return round(weighted_score / total_weight if total_weight > 0 else 0, 2)
    
    def _identify_priority_issues(self, assessment_results: Dict) -> List[Dict]:
        """Identify top priority issues across all dimensions"""
        
        priority_issues = []
        
        for dimension, results in assessment_results.items():
            if not results.get('meets_quality_standards', True):
                issue = {
                    'dimension': dimension,
                    'severity': 'high' if results['overall_score'] < 70 else 'medium',
                    'score': results['overall_score'],
                    'description': f"{dimension.title()} score ({results['overall_score']}%) below standards",
                    'impact': self._get_dimension_impact_description(dimension)
                }
                priority_issues.append(issue)
        
        # Sort by severity and score
        priority_issues.sort(key=lambda x: (x['severity'] == 'high', -x['score']), reverse=True)
        
        return priority_issues[:5]  # Top 5 priority issues
    
    def _get_dimension_impact_description(self, dimension: str) -> str:
        """Get business impact description for each dimension"""
        impacts = {
            'completeness': 'Missing critical data prevents accurate event categorization and analysis',
            'uniqueness': 'Duplicate events lead to inflated metrics and incorrect business insights',
            'validity': 'Invalid data formats cause processing errors and unreliable analytics',
            'consistency': 'Inconsistent values complicate aggregation and reporting',
            'timeliness': 'Stale data reduces relevance for real-time business decisions'
        }
        return impacts.get(dimension, 'Quality issues impact data reliability')
    
    def _generate_improvement_roadmap(self, assessment_results: Dict) -> List[Dict]:
        """Generate prioritized improvement roadmap"""
        
        roadmap = []
        
        # High priority fixes
        for dimension, results in assessment_results.items():
            if results['overall_score'] < 80:
                roadmap.append({
                    'priority': 'high',
                    'dimension': dimension,
                    'current_score': results['overall_score'],
                    'target_improvement': '15-20 points',
                    'estimated_effort': 'medium',
                    'recommendations': results.get('recommendations', [])[:3]  # Top 3 recommendations
                })
        
        # Medium priority optimizations
        for dimension, results in assessment_results.items():
            if 80 <= results['overall_score'] < 95:
                roadmap.append({
                    'priority': 'medium',
                    'dimension': dimension,
                    'current_score': results['overall_score'],
                    'target_improvement': '5-10 points',
                    'estimated_effort': 'low',
                    'recommendations': results.get('recommendations', [])[:2]  # Top 2 recommendations
                })
        
        return roadmap
    
    # Recommendation generators
    
    def _generate_completeness_recommendations(self, critical_analysis: Dict, optional_analysis: Dict) -> List[str]:
        """Generate recommendations for improving completeness"""
        recommendations = []
        
        # Critical field recommendations
        for field, analysis in critical_analysis.items():
            if not analysis['meets_threshold']:
                recommendations.append(
                    f"Implement data validation rules for {field} - currently {analysis['completeness_percentage']}% complete"
                )
        
        # Optional field recommendations
        high_value_optional = {
            field: analysis for field, analysis in optional_analysis.items()
            if analysis['completeness_percentage'] > 50  # Fields with some data
        }
        
        if high_value_optional:
            recommendations.append(
                "Focus on improving optional fields with existing data: " + 
                ", ".join(list(high_value_optional.keys())[:3])
            )
        
        recommendations.append("Implement data collection improvements at source systems")
        recommendations.append("Add validation checks to prevent missing critical data")
        
        return recommendations
    
    def _generate_uniqueness_recommendations(self, event_id_dups: Dict, summary_dups: Dict, fuzzy_dups: Dict) -> List[str]:
        """Generate recommendations for improving uniqueness"""
        recommendations = []
        
        if event_id_dups.get('duplicate_count', 0) > 0:
            recommendations.append(
                f"Implement unique constraint on event_id - found {event_id_dups['duplicate_count']} duplicates"
            )
        
        if summary_dups.get('duplicate_count', 0) > 0:
            recommendations.append(
                f"Review summary extraction logic - found {summary_dups['duplicate_count']} identical summaries"
            )
        
        if fuzzy_dups.get('fuzzy_duplicate_pairs', 0) > 0:
            recommendations.append(
                f"Implement semantic deduplication - found {fuzzy_dups['fuzzy_duplicate_pairs']} similar summaries"
            )
        
        recommendations.append("Implement automated duplicate detection in data pipeline")
        recommendations.append("Add composite key validation for event uniqueness")
        
        return recommendations
    
    def _generate_validity_recommendations(self, validity_checks: Dict) -> List[str]:
        """Generate recommendations for improving validity"""
        recommendations = []
        
        for check_name, check_results in validity_checks.items():
            if check_results.get('validity_percentage', 100) < 95:
                field_name = check_name.replace('_', ' ').title()
                recommendations.append(
                    f"Fix {field_name} validation - {check_results.get('invalid_count', 0)} invalid values found"
                )
        
        recommendations.append("Implement schema validation at data ingestion")
        recommendations.append("Add format validation rules for all structured fields")
        recommendations.append("Create data validation dashboard for monitoring")
        
        return recommendations
    
    def _generate_consistency_recommendations(self, consistency_checks: Dict) -> List[str]:
        """Generate recommendations for improving consistency"""
        recommendations = []
        
        for check_name, check_results in consistency_checks.items():
            if check_results.get('consistency_percentage', 100) < 90:
                opportunities = check_results.get('standardization_opportunities', 0)
                if opportunities > 0:
                    field_name = check_name.replace('_', ' ').title()
                    recommendations.append(
                        f"Standardize {field_name} - found {opportunities} inconsistencies"
                    )
        
        recommendations.append("Create data dictionaries with standard values")
        recommendations.append("Implement automated standardization rules")
        recommendations.append("Regular consistency monitoring and cleanup processes")
        
        return recommendations
    
    def _generate_timeliness_recommendations(self, timeliness_analysis: Dict) -> List[str]:
        """Generate recommendations for improving timeliness"""
        recommendations = []
        
        if 'detection_lag' in timeliness_analysis:
            lag_data = timeliness_analysis['detection_lag']
            if lag_data.get('analysis_possible') and lag_data.get('lag_statistics', {}).get('mean_lag_hours', 0) > 24:
                recommendations.append(
                    f"Improve detection speed - average lag is {lag_data['lag_statistics']['mean_lag_hours']:.1f} hours"
                )
        
        if 'data_freshness' in timeliness_analysis:
            freshness_data = timeliness_analysis['data_freshness']
            if freshness_data.get('analysis_possible') and freshness_data.get('freshness_score', 0) < 50:
                recommendations.append(
                    f"Increase data refresh frequency - only {freshness_data['freshness_score']:.1f}% of data is recent"
                )
        
        recommendations.append("Implement real-time or near-real-time data ingestion")
        recommendations.append("Set up automated alerts for stale data detection")
        recommendations.append("Create data freshness monitoring dashboard")
        
        return recommendations
    
    def _summarize_validity_issues(self, validity_checks: Dict) -> Dict[str, Any]:
        """Summarize validity issues across all checks"""
        total_invalid = 0
        total_checked = 0
        worst_validation = None
        worst_score = 100
        
        for check_name, results in validity_checks.items():
            invalid_count = results.get('invalid_count', 0)
            checked_count = results.get('total_checked', 0)
            validity_score = results.get('validity_percentage', 100)
            
            total_invalid += invalid_count
            total_checked += checked_count
            
            if validity_score < worst_score:
                worst_score = validity_score
                worst_validation = check_name
        
        return {
            'total_invalid_values': int(total_invalid),
            'total_values_checked': int(total_checked),
            'overall_invalid_rate': round((total_invalid / total_checked) * 100, 2) if total_checked > 0 else 0,
            'worst_validation_check': worst_validation,
            'worst_validation_score': round(worst_score, 2)
        }
    
    def _summarize_consistency_issues(self, consistency_checks: Dict) -> Dict[str, Any]:
        """Summarize consistency issues across all checks"""
        total_inconsistencies = 0
        worst_consistency = None
        worst_score = 100
        
        for check_name, results in consistency_checks.items():
            inconsistencies = results.get('standardization_opportunities', 0)
            consistency_score = results.get('consistency_percentage', 100)
            
            total_inconsistencies += inconsistencies
            
            if consistency_score < worst_score:
                worst_score = consistency_score
                worst_consistency = check_name
        
        return {
            'total_inconsistencies': int(total_inconsistencies),
            'worst_consistency_area': worst_consistency,
            'worst_consistency_score': round(worst_score, 2),
            'areas_needing_standardization': len([
                check for check, results in consistency_checks.items()
                if results.get('consistency_percentage', 100) < 90
            ])
        }
    
    def _summarize_timeliness_issues(self, timeliness_analysis: Dict) -> Dict[str, Any]:
        """Summarize timeliness issues"""
        issues = {}
        
        if 'data_freshness' in timeliness_analysis:
            freshness = timeliness_analysis['data_freshness']
            if freshness.get('analysis_possible'):
                issues['stale_data_percentage'] = round(100 - freshness.get('freshness_score', 0), 2)
                issues['oldest_data_days'] = freshness.get('data_age_stats', {}).get('oldest_days_ago', 0)
        
        if 'detection_lag' in timeliness_analysis:
            lag = timeliness_analysis['detection_lag']
            if lag.get('analysis_possible'):
                issues['slow_detection_events'] = lag.get('lag_distribution', {}).get('slow_over_7d', 0)
                issues['average_detection_lag_hours'] = lag.get('lag_statistics', {}).get('mean_lag_hours', 0)
        
        return issues
    
    def generate_assessment_report(self) -> str:
        """Generate comprehensive assessment report"""
        
        if not self.quality_results:
            return "No assessment results available. Run run_comprehensive_assessment() first."
        
        results = self.quality_results
        
        report = f"""# Data Quality Assessment Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- **Overall Data Quality Score**: {results['overall_quality']['composite_score']}/100
- **Sample Size Analyzed**: {results['assessment_metadata']['sample_size']:,} events
- **Standards Compliance**: {sum(results['overall_quality']['standards_compliance'].values())}/5 dimensions

## Quality Dimension Scores
"""
        
        # Add dimension scores
        for dimension, score in results['overall_quality']['dimension_scores'].items():
            compliance = "✅" if results['overall_quality']['standards_compliance'][dimension] else "❌"
            report += f"- **{dimension.title()}**: {score}/100 {compliance}\n"
        
        report += "\n## Priority Issues\n"
        
        for issue in results['priority_issues']:
            report += f"### {issue['severity'].upper()}: {issue['dimension'].title()} ({issue['score']}/100)\n"
            report += f"{issue['description']}\n"
            report += f"**Business Impact**: {issue['impact']}\n\n"
        
        report += "## Improvement Roadmap\n"
        
        for item in results['improvement_roadmap']:
            report += f"### {item['priority'].upper()} Priority: {item['dimension'].title()}\n"
            report += f"- **Current Score**: {item['current_score']}/100\n"
            report += f"- **Target Improvement**: {item['target_improvement']}\n"
            report += f"- **Estimated Effort**: {item['estimated_effort']}\n"
            report += "**Key Actions**:\n"
            for rec in item['recommendations']:
                report += f"  - {rec}\n"
            report += "\n"
        
        return report
    
    def save_assessment_results(self, output_path: Optional[str] = None) -> str:
        """Save assessment results to files"""
        
        if not self.quality_results:
            logger.error("No assessment results to save")
            return ""
        
        if output_path is None:
            output_dir = Path(self.config['data_paths']['quality_reports_dir'])
        else:
            output_dir = Path(output_path)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save assessment as JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        assessment_file = output_dir / f"quality_assessment_{timestamp}.json"
        
        with open(assessment_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(self.quality_results, f, indent=2, default=str)
        
        # Save report as markdown
        report_file = output_dir / f"quality_report_{timestamp}.md"
        report_content = self.generate_assessment_report()
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Assessment results saved to: {assessment_file}")
        logger.info(f"Assessment report saved to: {report_file}")
        
        return str(assessment_file)


if __name__ == "__main__":
    # Initialize and run assessment
    evaluator = DataQualityEvaluator()
    
    # Run comprehensive assessment
    results = evaluator.run_comprehensive_assessment(sample_size=5000)
    
    # Save results
    evaluator.save_assessment_results()
    
    # Print summary
    print("Data Quality Assessment completed!")
    print(f"Overall Quality Score: {results['overall_quality']['composite_score']}/100")
    print(f"Dimensions meeting standards: {sum(results['overall_quality']['standards_compliance'].values())}/5")
    print(f"Priority issues identified: {len(results['priority_issues'])}")