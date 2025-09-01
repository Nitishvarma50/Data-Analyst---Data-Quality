"""
Data Cleaning Module for News Events Dataset
Addresses critical data quality issues identified in assessment
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import re
import uuid
import json
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import Counter, defaultdict
from loguru import logger
import yaml
from pathlib import Path
import sys
import os
from difflib import SequenceMatcher
import hashlib
from dataclasses import dataclass

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_loader import NewsEventsDataLoader, flatten_news_event_record


@dataclass
class CleaningStats:
    """Statistics tracking for cleaning operations"""
    records_processed: int = 0
    records_cleaned: int = 0
    duplicates_removed: int = 0
    locations_standardized: int = 0
    categories_normalized: int = 0
    invalid_data_fixed: int = 0
    missing_data_filled: int = 0


class LocationStandardizer:
    """Handles location standardization and normalization"""
    
    def __init__(self):
        self.location_mappings = {}
        self.standard_locations = set()
        self._build_location_mappings()
    
    def _build_location_mappings(self):
        """Build comprehensive location mapping rules"""
        
        # Country standardization
        self.country_mappings = {
            'USA': 'United States',
            'US': 'United States',
            'U.S.': 'United States',
            'United States of America': 'United States',
            'UK': 'United Kingdom',
            'U.K.': 'United Kingdom',
            'Britain': 'United Kingdom',
            'Great Britain': 'United Kingdom',
        }
        
        # State/Province standardization
        self.state_mappings = {
            'CA': 'California',
            'NY': 'New York',
            'TX': 'Texas',
            'FL': 'Florida',
            'IL': 'Illinois',
            'PA': 'Pennsylvania',
            'OH': 'Ohio',
            'GA': 'Georgia',
            'NC': 'North Carolina',
            'MI': 'Michigan',
        }
        
        # City standardization
        self.city_mappings = {
            'NYC': 'New York City',
            'SF': 'San Francisco',
            'LA': 'Los Angeles',
            'Chi': 'Chicago',
            'Philly': 'Philadelphia',
        }
        
        # Common location patterns to standardize
        self.location_patterns = [
            (r'\b(St\.?)\b', 'Saint'),  # St. -> Saint
            (r'\b(Mt\.?)\b', 'Mount'),  # Mt. -> Mount
            (r'\b(N\.?)\b', 'North'),   # N. -> North
            (r'\b(S\.?)\b', 'South'),   # S. -> South
            (r'\b(E\.?)\b', 'East'),    # E. -> East
            (r'\b(W\.?)\b', 'West'),    # W. -> West
        ]
    
    def standardize_location(self, location: str) -> str:
        """Standardize a single location string"""
        if pd.isna(location) or not isinstance(location, str):
            return location
        
        # Clean and normalize
        cleaned = location.strip()
        
        # Apply pattern replacements
        for pattern, replacement in self.location_patterns:
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        
        # Apply country mappings
        for abbrev, full_name in self.country_mappings.items():
            cleaned = re.sub(rf'\b{re.escape(abbrev)}\b', full_name, cleaned, flags=re.IGNORECASE)
        
        # Apply state mappings
        for abbrev, full_name in self.state_mappings.items():
            cleaned = re.sub(rf'\b{re.escape(abbrev)}\b', full_name, cleaned, flags=re.IGNORECASE)
        
        # Apply city mappings
        for abbrev, full_name in self.city_mappings.items():
            cleaned = re.sub(rf'\b{re.escape(abbrev)}\b', full_name, cleaned, flags=re.IGNORECASE)
        
        # Normalize whitespace and capitalization
        cleaned = ' '.join(cleaned.split())  # Remove extra whitespace
        cleaned = cleaned.title()  # Proper case
        
        return cleaned
    
    def build_location_clusters(self, locations: pd.Series) -> Dict[str, List[str]]:
        """Build clusters of similar locations for standardization"""
        
        unique_locations = locations.dropna().unique()
        clusters = defaultdict(list)
        processed = set()
        
        for loc1 in unique_locations:
            if loc1 in processed:
                continue
            
            cluster = [loc1]
            processed.add(loc1)
            
            for loc2 in unique_locations:
                if loc2 in processed:
                    continue
                
                if self._are_similar_locations(loc1, loc2):
                    cluster.append(loc2)
                    processed.add(loc2)
            
            if len(cluster) > 1:
                # Use the most frequent location as the standard
                location_counts = locations.value_counts()
                standard_location = max(cluster, key=lambda x: location_counts.get(x, 0))
                clusters[standard_location] = cluster
        
        return dict(clusters)
    
    def _are_similar_locations(self, loc1: str, loc2: str, threshold: float = 0.85) -> bool:
        """Check if two locations are similar enough to be the same"""
        similarity = SequenceMatcher(None, loc1.lower(), loc2.lower()).ratio()
        return similarity >= threshold


class CategoryNormalizer:
    """Handles category standardization and normalization"""
    
    def __init__(self):
        self.category_mappings = {}
        self._build_category_mappings()
    
    def _build_category_mappings(self):
        """Build category standardization rules"""
        
        # Standard business event categories
        self.standard_categories = {
            'receives_award': ['receives_award', 'awarded', 'wins_award', 'honored'],
            'recognized_as': ['recognized_as', 'named_as', 'designated_as'],
            'launches': ['launches', 'introduces', 'unveils', 'releases'],
            'receives_financing': ['receives_financing', 'raises_funding', 'secures_investment'],
            'hires': ['hires', 'appoints', 'recruits', 'employs'],
            'announces': ['announces', 'declares', 'reveals'],
            'partners_with': ['partners_with', 'collaborates_with', 'teams_up'],
            'acquires': ['acquires', 'purchases', 'buys'],
            'expands': ['expands', 'grows', 'extends'],
            'is_developing': ['is_developing', 'develops', 'creates']
        }
        
        # Build reverse mapping
        for standard, variants in self.standard_categories.items():
            for variant in variants:
                self.category_mappings[variant.lower()] = standard
    
    def normalize_category(self, category: str) -> str:
        """Normalize a single category value"""
        if pd.isna(category) or not isinstance(category, str):
            return category
        
        cleaned_category = category.lower().strip()
        
        # Direct mapping
        if cleaned_category in self.category_mappings:
            return self.category_mappings[cleaned_category]
        
        # Fuzzy matching for close variants
        for standard_cat in self.standard_categories.keys():
            if SequenceMatcher(None, cleaned_category, standard_cat).ratio() >= 0.8:
                return standard_cat
        
        # Return original if no match found
        return category


class DataValidator:
    """Handles data validation and correction"""
    
    @staticmethod
    def validate_and_fix_confidence(confidence_series: pd.Series) -> pd.Series:
        """Validate and fix confidence scores"""
        cleaned = confidence_series.copy()
        
        # Convert to numeric, coercing errors to NaN
        cleaned = pd.to_numeric(cleaned, errors='coerce')
        
        # Clip values to [0, 1] range
        cleaned = cleaned.clip(lower=0, upper=1)
        
        return cleaned
    
    @staticmethod
    def validate_and_fix_event_ids(event_id_series: pd.Series) -> pd.Series:
        """Validate and fix event ID formats"""
        cleaned = event_id_series.copy()
        
        def fix_uuid(value):
            if pd.isna(value):
                return value
            
            try:
                # Try to parse as UUID
                uuid.UUID(str(value))
                return str(value)
            except (ValueError, TypeError):
                # Generate new UUID if invalid
                logger.warning(f"Invalid UUID format: {value}. Generating new UUID.")
                return str(uuid.uuid4())
        
        cleaned = cleaned.apply(fix_uuid)
        return cleaned
    
    @staticmethod
    def validate_and_fix_dates(date_series: pd.Series, column_name: str = "date") -> pd.Series:
        """Validate and fix date formats"""
        cleaned = date_series.copy()
        
        # Convert to datetime, coercing errors to NaT
        cleaned = pd.to_datetime(cleaned, errors='coerce')
        
        # Log invalid dates
        invalid_mask = cleaned.isna() & date_series.notna()
        if invalid_mask.any():
            invalid_count = invalid_mask.sum()
            logger.warning(f"Found {invalid_count} invalid dates in {column_name}")
        
        return cleaned
    
    @staticmethod
    def validate_and_fix_amounts(amount_series: pd.Series) -> pd.Series:
        """Validate and fix amount values"""
        cleaned = amount_series.copy()
        
        # Convert to numeric
        cleaned = pd.to_numeric(cleaned, errors='coerce')
        
        # Set negative amounts to NaN (assume invalid)
        negative_mask = cleaned < 0
        if negative_mask.any():
            logger.warning(f"Found {negative_mask.sum()} negative amounts. Setting to NaN.")
            cleaned[negative_mask] = np.nan
        
        return cleaned


class NewsEventsDataCleaner:
    """Comprehensive data cleaning pipeline for news events"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.loader = NewsEventsDataLoader(self.config['data_paths']['raw_data_dir'])
        self.location_standardizer = LocationStandardizer()
        self.category_normalizer = CategoryNormalizer()
        self.cleaning_stats = CleaningStats()
        self.cleaning_log = []
    
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
                'chunk_size': 1000,
                'max_records_sample': None  # None means process all data
            }
        }
    
    
    def clean_sample_data(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Clean sample data and return cleaned DataFrame"""
        if sample_size is None:
            logger.info("Starting data cleaning for ALL available data")
            all_records = self._load_all_data()
        else:
            logger.info(f"Starting data cleaning for sample size: {sample_size}")
            all_records = self.loader.load_sample_data(sample_size)
        
        # Flatten to DataFrame
        flattened_data = []
        for item in all_records:
            try:
                events = flatten_news_event_record(item['record'])
                for event in events:
                    event['source_file'] = item['source_file']
                    flattened_data.append(event)
            except Exception as e:
                logger.warning(f"Error flattening record: {e}")
                continue
        
        df = pd.DataFrame(flattened_data)
        self.cleaning_stats.records_processed = len(df)
        
        logger.info(f"Loaded {len(df)} events for cleaning")
        
        # Apply cleaning steps
        cleaned_df = self._apply_comprehensive_cleaning(df)
        
        logger.info(f"Cleaning completed. {len(cleaned_df)} records after cleaning")
        return cleaned_df
    
    def _apply_comprehensive_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all cleaning steps to the DataFrame"""
        
        cleaned_df = df.copy()
        initial_count = len(cleaned_df)
        
        # Step 1: Remove exact duplicates
        logger.info("Step 1: Removing duplicate records...")
        cleaned_df = self._remove_duplicates(cleaned_df)
        
        # Step 2: Standardize locations
        logger.info("Step 2: Standardizing locations...")
        cleaned_df = self._standardize_locations(cleaned_df)
        
        # Step 3: Normalize categories
        logger.info("Step 3: Normalizing categories...")
        cleaned_df = self._normalize_categories(cleaned_df)
        
        # Step 4: Validate and fix data formats
        logger.info("Step 4: Validating and fixing data formats...")
        cleaned_df = self._validate_and_fix_formats(cleaned_df)
        
        # Step 5: Handle missing data
        logger.info("Step 5: Handling missing data...")
        cleaned_df = self._handle_missing_data(cleaned_df)
        
        # Step 6: Final validation
        logger.info("Step 6: Final validation...")
        cleaned_df = self._final_validation_cleanup(cleaned_df)
        
        # Update stats
        self.cleaning_stats.records_cleaned = len(cleaned_df)
        
        # Log summary
        self._log_cleaning_summary(initial_count, len(cleaned_df))
        
        return cleaned_df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records using multiple strategies"""
        
        initial_count = len(df)
        
        # Strategy 1: Remove exact event_id duplicates
        if 'event_id' in df.columns:
            before_count = len(df)
            df = df.drop_duplicates(subset=['event_id'], keep='first')
            id_duplicates_removed = before_count - len(df)
            
            if id_duplicates_removed > 0:
                logger.info(f"Removed {id_duplicates_removed} duplicate event IDs")
                self.cleaning_log.append(f"Removed {id_duplicates_removed} duplicate event IDs")
        
        # Strategy 2: Remove semantic duplicates (similar summaries from same company)
        if all(col in df.columns for col in ['summary', 'company_name']):
            df = self._remove_semantic_duplicates(df)
        
        # Strategy 3: Remove composite duplicates
        composite_fields = ['company_name', 'category', 'effective_date']
        available_fields = [field for field in composite_fields if field in df.columns]
        
        if len(available_fields) >= 2:
            before_count = len(df)
            df = df.drop_duplicates(subset=available_fields, keep='first')
            composite_duplicates_removed = before_count - len(df)
            
            if composite_duplicates_removed > 0:
                logger.info(f"Removed {composite_duplicates_removed} composite duplicates")
                self.cleaning_log.append(f"Removed {composite_duplicates_removed} composite duplicates")
        
        total_duplicates_removed = initial_count - len(df)
        self.cleaning_stats.duplicates_removed = total_duplicates_removed
        
        return df
    
    def _remove_semantic_duplicates(self, df: pd.DataFrame, similarity_threshold: float = 0.9) -> pd.DataFrame:
        """Remove semantically similar summaries from the same company"""
        
        # Group by company to check for similar summaries within each company
        duplicates_to_remove = []
        
        for company, group in df.groupby('company_name'):
            if len(group) <= 1:
                continue
            
            summaries = group['summary'].dropna().tolist()
            indices = group['summary'].dropna().index.tolist()
            
            # Compare summaries within this company
            for i, (summary1, idx1) in enumerate(zip(summaries, indices)):
                for summary2, idx2 in zip(summaries[i+1:], indices[i+1:]):
                    similarity = SequenceMatcher(None, str(summary1).lower(), str(summary2).lower()).ratio()
                    
                    if similarity >= similarity_threshold:
                        # Keep the one with higher confidence, or first one if equal
                        conf1 = df.loc[idx1, 'confidence'] if 'confidence' in df.columns else 0
                        conf2 = df.loc[idx2, 'confidence'] if 'confidence' in df.columns else 0
                        
                        if conf1 >= conf2:
                            duplicates_to_remove.append(idx2)
                        else:
                            duplicates_to_remove.append(idx1)
        
        # Remove semantic duplicates
        if duplicates_to_remove:
            df = df.drop(index=duplicates_to_remove)
            logger.info(f"Removed {len(duplicates_to_remove)} semantic duplicates")
            self.cleaning_log.append(f"Removed {len(duplicates_to_remove)} semantic duplicates")
        
        return df
    
    def _standardize_locations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize location values"""
        
        if 'location' not in df.columns:
            return df
        
        initial_unique_locations = df['location'].nunique()
        
        # Apply basic standardization
        df['location'] = df['location'].apply(self.location_standardizer.standardize_location)
        
        # Build and apply location clusters
        location_clusters = self.location_standardizer.build_location_clusters(df['location'])
        
        # Apply cluster mappings
        for standard_location, variants in location_clusters.items():
            for variant in variants:
                df.loc[df['location'] == variant, 'location'] = standard_location
        
        final_unique_locations = df['location'].nunique()
        locations_standardized = initial_unique_locations - final_unique_locations
        
        self.cleaning_stats.locations_standardized = locations_standardized
        
        if locations_standardized > 0:
            logger.info(f"Standardized {locations_standardized} location variations")
            self.cleaning_log.append(f"Standardized {locations_standardized} location variations")
        
        return df
    
    def _normalize_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize category values"""
        
        if 'category' not in df.columns:
            return df
        
        initial_unique_categories = df['category'].nunique()
        
        # Apply category normalization
        df['category'] = df['category'].apply(self.category_normalizer.normalize_category)
        
        final_unique_categories = df['category'].nunique()
        categories_normalized = initial_unique_categories - final_unique_categories
        
        self.cleaning_stats.categories_normalized = categories_normalized
        
        if categories_normalized > 0:
            logger.info(f"Normalized {categories_normalized} category variations")
            self.cleaning_log.append(f"Normalized {categories_normalized} category variations")
        
        return df
    
    def _validate_and_fix_formats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix data formats"""
        
        fixes_applied = 0
        
        # Fix confidence scores
        if 'confidence' in df.columns:
            original_confidence = df['confidence'].copy()
            df['confidence'] = DataValidator.validate_and_fix_confidence(df['confidence'])
            
            # Count fixes
            confidence_fixes = ((original_confidence != df['confidence']) & 
                              original_confidence.notna() & df['confidence'].notna()).sum()
            fixes_applied += confidence_fixes
            
            if confidence_fixes > 0:
                logger.info(f"Fixed {confidence_fixes} confidence score format issues")
                self.cleaning_log.append(f"Fixed {confidence_fixes} confidence scores")
        
        # Fix event IDs
        if 'event_id' in df.columns:
            original_ids = df['event_id'].copy()
            df['event_id'] = DataValidator.validate_and_fix_event_ids(df['event_id'])
            
            # Count fixes (new UUIDs generated)
            id_fixes = (original_ids != df['event_id']).sum()
            fixes_applied += id_fixes
            
            if id_fixes > 0:
                logger.info(f"Fixed {id_fixes} invalid event ID formats")
                self.cleaning_log.append(f"Fixed {id_fixes} event ID formats")
        
        # Fix date formats
        date_columns = ['found_at', 'effective_date', 'article_published_at']
        for col in date_columns:
            if col in df.columns:
                original_dates = df[col].copy()
                df[col] = DataValidator.validate_and_fix_dates(df[col], col)
                
                # Count fixes
                date_fixes = (original_dates.notna() & df[col].isna()).sum()
                fixes_applied += date_fixes
                
                if date_fixes > 0:
                    logger.info(f"Found {date_fixes} invalid dates in {col}")
                    self.cleaning_log.append(f"Found {date_fixes} invalid dates in {col}")
        
        # Fix amount values
        amount_columns = ['amount_normalized', 'amount']
        for col in amount_columns:
            if col in df.columns:
                original_amounts = df[col].copy()
                df[col] = DataValidator.validate_and_fix_amounts(df[col])
                
                # Count fixes
                amount_fixes = (original_amounts.notna() & df[col].isna()).sum()
                fixes_applied += amount_fixes
                
                if amount_fixes > 0:
                    logger.info(f"Fixed {amount_fixes} invalid amounts in {col}")
                    self.cleaning_log.append(f"Fixed {amount_fixes} amounts in {col}")
        
        self.cleaning_stats.invalid_data_fixed = fixes_applied
        return df
    
    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data using business rules"""
        
        missing_data_filled = 0
        
        # Strategy 1: Fill missing categories based on summary text patterns
        if 'category' in df.columns and 'summary' in df.columns:
            missing_category_mask = df['category'].isna()
            if missing_category_mask.any():
                filled_categories = self._infer_categories_from_summaries(
                    df.loc[missing_category_mask, 'summary']
                )
                df.loc[missing_category_mask, 'category'] = filled_categories
                
                category_fills = filled_categories.notna().sum()
                missing_data_filled += category_fills
                
                if category_fills > 0:
                    logger.info(f"Inferred {category_fills} missing categories from summaries")
                    self.cleaning_log.append(f"Inferred {category_fills} categories from text patterns")
        
        # Strategy 2: Fill missing confidence scores with conservative estimates
        if 'confidence' in df.columns:
            missing_confidence_mask = df['confidence'].isna()
            if missing_confidence_mask.any():
                # Use median confidence as conservative estimate
                median_confidence = df['confidence'].median()
                if pd.notna(median_confidence):
                    df.loc[missing_confidence_mask, 'confidence'] = median_confidence
                    confidence_fills = missing_confidence_mask.sum()
                    missing_data_filled += confidence_fills
                    
                    logger.info(f"Filled {confidence_fills} missing confidence scores with median value")
                    self.cleaning_log.append(f"Filled {confidence_fills} confidence scores with median")
        
        # Strategy 3: Fill missing effective_date with found_at date
        if 'effective_date' in df.columns and 'found_at' in df.columns:
            missing_effective_mask = df['effective_date'].isna() & df['found_at'].notna()
            if missing_effective_mask.any():
                df.loc[missing_effective_mask, 'effective_date'] = df.loc[missing_effective_mask, 'found_at']
                effective_date_fills = missing_effective_mask.sum()
                missing_data_filled += effective_date_fills
                
                if effective_date_fills > 0:
                    logger.info(f"Filled {effective_date_fills} missing effective dates with found_at")
                    self.cleaning_log.append(f"Filled {effective_date_fills} effective dates")
        
        self.cleaning_stats.missing_data_filled = missing_data_filled
        return df
    
    def _infer_categories_from_summaries(self, summaries: pd.Series) -> pd.Series:
        """Infer categories from summary text patterns"""
        
        # Define category keywords
        category_keywords = {
            'receives_award': ['award', 'honored', 'recognized', 'winner', 'wins'],
            'launches': ['launch', 'introduce', 'unveil', 'release', 'debut'],
            'receives_financing': ['funding', 'investment', 'financing', 'capital', 'raise'],
            'hires': ['hire', 'appoint', 'recruit', 'join', 'employee'],
            'announces': ['announce', 'declare', 'reveal', 'disclose'],
            'partners_with': ['partner', 'collaborate', 'alliance', 'team'],
            'acquires': ['acquire', 'purchase', 'buy', 'acquisition'],
            'expands': ['expand', 'grow', 'extension', 'increase']
        }
        
        inferred_categories = []
        
        for summary in summaries:
            if pd.isna(summary):
                inferred_categories.append(np.nan)
                continue
            
            summary_lower = str(summary).lower()
            best_match = None
            max_matches = 0
            
            # Find category with most keyword matches
            for category, keywords in category_keywords.items():
                matches = sum(1 for keyword in keywords if keyword in summary_lower)
                if matches > max_matches:
                    max_matches = matches
                    best_match = category
            
            inferred_categories.append(best_match)
        
        return pd.Series(inferred_categories, index=summaries.index)
    
    def _final_validation_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final validation and cleanup"""
        
        initial_count = len(df)
        
        # Remove records with missing critical fields
        critical_fields = self.config.get('critical_fields', ['event_id', 'summary'])
        available_critical_fields = [field for field in critical_fields if field in df.columns]
        
        if available_critical_fields:
            # Keep records that have at least the most critical fields
            essential_fields = ['event_id', 'summary']
            available_essential = [field for field in essential_fields if field in df.columns]
            
            if available_essential:
                # Remove records missing ALL essential fields
                valid_mask = df[available_essential].notna().any(axis=1)
                df = df[valid_mask]
                
                removed_count = initial_count - len(df)
                if removed_count > 0:
                    logger.info(f"Removed {removed_count} records with missing essential data")
                    self.cleaning_log.append(f"Removed {removed_count} records missing essential fields")
        
        # Add data quality flags
        df = self._add_quality_flags(df)
        
        return df
    
    def _add_quality_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add quality assessment flags to cleaned data"""
        
        # Add quality score flag
        df['data_quality_score'] = 100.0  # Start with perfect score
        
        # Reduce score for missing optional fields
        optional_fields = ['location', 'amount_normalized', 'effective_date', 'contact']
        available_optional = [field for field in optional_fields if field in df.columns]
        
        for field in available_optional:
            missing_mask = df[field].isna()
            df.loc[missing_mask, 'data_quality_score'] -= 10  # -10 points per missing optional field
        
        # Reduce score for low confidence
        if 'confidence' in df.columns:
            low_confidence_mask = df['confidence'] < 0.7
            df.loc[low_confidence_mask, 'data_quality_score'] -= 15  # -15 points for low confidence
        
        # Ensure score doesn't go below 0
        df['data_quality_score'] = df['data_quality_score'].clip(lower=0)
        
        # Add cleaning flags
        df['was_cleaned'] = True
        df['cleaning_timestamp'] = datetime.now(timezone.utc).isoformat()
        
        return df
    
    def _log_cleaning_summary(self, initial_count: int, final_count: int):
        """Log comprehensive cleaning summary"""
        
        stats = self.cleaning_stats
        
        summary = f"""
Cleaning Summary:
- Records processed: {stats.records_processed:,}
- Records retained: {final_count:,}
- Records removed: {initial_count - final_count:,}
- Duplicates removed: {stats.duplicates_removed:,}
- Locations standardized: {stats.locations_standardized:,}
- Categories normalized: {stats.categories_normalized:,}
- Format issues fixed: {stats.invalid_data_fixed:,}
- Missing data filled: {stats.missing_data_filled:,}
        """
        
        logger.info(summary)
        self.cleaning_log.append(f"Cleaning completed: {final_count:,} clean records from {initial_count:,} original")
    
    def generate_cleaning_report(self) -> str:
        """Generate comprehensive cleaning report"""
        
        stats = self.cleaning_stats
        
        report = f"""# Data Cleaning Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Cleaning Summary
- **Records Processed**: {stats.records_processed:,}
- **Records Cleaned**: {stats.records_cleaned:,}
- **Data Retention Rate**: {(stats.records_cleaned / stats.records_processed * 100):.2f}%

## Cleaning Operations Performed

### Duplicate Removal
- **Duplicates Removed**: {stats.duplicates_removed:,}
- **Deduplication Rate**: {(stats.duplicates_removed / stats.records_processed * 100):.2f}%

### Location Standardization
- **Locations Standardized**: {stats.locations_standardized:,}
- **Standardization Impact**: Reduced location variations significantly

### Category Normalization
- **Categories Normalized**: {stats.categories_normalized:,}
- **Normalization Impact**: Improved category consistency

### Data Validation & Format Fixes
- **Format Issues Fixed**: {stats.invalid_data_fixed:,}
- **Missing Data Filled**: {stats.missing_data_filled:,}

## Cleaning Log
"""
        
        for log_entry in self.cleaning_log:
            report += f"- {log_entry}\n"
        
        report += f"""
## Data Quality Improvements
The cleaning process addressed the following quality issues:
- **Consistency**: Location and category standardization
- **Validity**: Format validation and correction
- **Completeness**: Strategic missing data imputation
- **Uniqueness**: Multi-strategy duplicate removal

## Next Steps
1. Validate cleaned data quality using DataQualityEvaluator
2. Load cleaned data into database schema
3. Set up monitoring for ongoing quality maintenance
        """
        
        return report
    
    def save_cleaned_data(self, cleaned_df: pd.DataFrame, output_path: Optional[str] = None) -> str:
        """Save cleaned data to files"""
        
        if output_path is None:
            output_dir = Path(self.config['data_paths']['processed_data_dir'])
        else:
            output_dir = Path(output_path)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cleaned data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save as CSV for analysis
        csv_file = output_dir / f"cleaned_news_events_{timestamp}.csv"
        cleaned_df.to_csv(csv_file, index=False, encoding='utf-8')
        
        # Save as Parquet for efficient storage
        parquet_file = output_dir / f"cleaned_news_events_{timestamp}.parquet"
        cleaned_df.to_parquet(parquet_file, index=False)
        
        # Save cleaning statistics
        stats_file = output_dir / f"cleaning_stats_{timestamp}.json"
        stats_dict = {
            'cleaning_stats': {
                'records_processed': self.cleaning_stats.records_processed,
                'records_cleaned': self.cleaning_stats.records_cleaned,
                'duplicates_removed': self.cleaning_stats.duplicates_removed,
                'locations_standardized': self.cleaning_stats.locations_standardized,
                'categories_normalized': self.cleaning_stats.categories_normalized,
                'invalid_data_fixed': self.cleaning_stats.invalid_data_fixed,
                'missing_data_filled': self.cleaning_stats.missing_data_filled
            },
            'cleaning_log': self.cleaning_log,
            'cleaning_timestamp': datetime.now().isoformat()
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_dict, f, indent=2, default=str)
        
        # Save cleaning report
        report_file = output_dir / f"cleaning_report_{timestamp}.md"
        report_content = self.generate_cleaning_report()
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Cleaned data saved to: {csv_file}")
        logger.info(f"Parquet format saved to: {parquet_file}")
        logger.info(f"Cleaning statistics saved to: {stats_file}")
        logger.info(f"Cleaning report saved to: {report_file}")
        
        return str(csv_file)
    
    def clean_and_validate_full_dataset(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Clean the full dataset file by file and validate results"""
        
        logger.info("Starting full dataset cleaning...")
        
        if output_path is None:
            output_dir = Path(self.config['data_paths']['processed_data_dir'])
        else:
            output_dir = Path(output_path)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_cleaned_data = []
        total_stats = CleaningStats()
        
        # Process each file
        for file_path in self.loader.get_json_files():
            logger.info(f"Processing file: {file_path.name}")
            
            # Load file data
            file_records = self.loader.load_file_data(file_path)
            
            # Flatten records
            flattened_data = []
            for record in file_records:
                try:
                    events = flatten_news_event_record(record)
                    for event in events:
                        event['source_file'] = file_path.name
                        flattened_data.append(event)
                except Exception as e:
                    logger.warning(f"Error flattening record from {file_path.name}: {e}")
                    continue
            
            if not flattened_data:
                logger.warning(f"No valid events found in {file_path.name}")
                continue
            
            # Clean file data
            file_df = pd.DataFrame(flattened_data)
            self.cleaning_stats = CleaningStats()  # Reset for this file
            
            cleaned_file_df = self._apply_comprehensive_cleaning(file_df)
            
            # Accumulate stats
            total_stats.records_processed += self.cleaning_stats.records_processed
            total_stats.records_cleaned += self.cleaning_stats.records_cleaned
            total_stats.duplicates_removed += self.cleaning_stats.duplicates_removed
            total_stats.locations_standardized += self.cleaning_stats.locations_standardized
            total_stats.categories_normalized += self.cleaning_stats.categories_normalized
            total_stats.invalid_data_fixed += self.cleaning_stats.invalid_data_fixed
            total_stats.missing_data_filled += self.cleaning_stats.missing_data_filled
            
            all_cleaned_data.append(cleaned_file_df)
            
            logger.info(f"Completed {file_path.name}: {len(cleaned_file_df)} clean records")
        
        # Combine all cleaned data
        if all_cleaned_data:
            final_cleaned_df = pd.concat(all_cleaned_data, ignore_index=True)
            
            # Remove cross-file duplicates
            initial_combined_count = len(final_cleaned_df)
            final_cleaned_df = final_cleaned_df.drop_duplicates(subset=['event_id'], keep='first')
            cross_file_duplicates = initial_combined_count - len(final_cleaned_df)
            
            if cross_file_duplicates > 0:
                logger.info(f"Removed {cross_file_duplicates} cross-file duplicates")
                total_stats.duplicates_removed += cross_file_duplicates
            
            total_stats.records_cleaned = len(final_cleaned_df)
            
            # Save final results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            final_csv = output_dir / f"news_events_cleaned_full_{timestamp}.csv"
            final_parquet = output_dir / f"news_events_cleaned_full_{timestamp}.parquet"
            
            final_cleaned_df.to_csv(final_csv, index=False, encoding='utf-8')
            final_cleaned_df.to_parquet(final_parquet, index=False)
            
            # Update cleaning stats for final report
            self.cleaning_stats = total_stats
            
            # Generate final report
            final_report = output_dir / f"full_dataset_cleaning_report_{timestamp}.md"
            report_content = self.generate_cleaning_report()
            
            with open(final_report, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"Full dataset cleaning completed!")
            logger.info(f"Final cleaned dataset: {len(final_cleaned_df):,} records")
            logger.info(f"Saved to: {final_csv}")
            
            return {
                'cleaned_data_path': str(final_csv),
                'parquet_path': str(final_parquet),
                'final_record_count': len(final_cleaned_df),
                'cleaning_stats': {
                    'records_processed': total_stats.records_processed,
                    'records_cleaned': total_stats.records_cleaned,
                    'duplicates_removed': total_stats.duplicates_removed,
                    'locations_standardized': total_stats.locations_standardized,
                    'categories_normalized': total_stats.categories_normalized,
                    'invalid_data_fixed': total_stats.invalid_data_fixed,
                    'missing_data_filled': total_stats.missing_data_filled
                },
                'retention_rate': round((total_stats.records_cleaned / total_stats.records_processed) * 100, 2)
            }
        
        else:
            logger.error("No data could be processed and cleaned")
            return {}
    
    def validate_cleaning_effectiveness(self, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate that cleaning operations were effective"""
        
        logger.info("Validating cleaning effectiveness...")
        
        validation_results = {}
        
        # Check duplicate removal effectiveness
        if 'event_id' in cleaned_df.columns:
            remaining_duplicates = cleaned_df['event_id'].duplicated().sum()
            validation_results['duplicate_validation'] = {
                'remaining_duplicates': int(remaining_duplicates),
                'deduplication_success': remaining_duplicates == 0
            }
        
        # Check location standardization effectiveness
        if 'location' in cleaned_df.columns:
            location_variations = self._count_location_variations(cleaned_df['location'])
            validation_results['location_validation'] = {
                'remaining_variations': location_variations,
                'standardization_effective': location_variations < 50  # Reasonable threshold
            }
        
        # Check category normalization effectiveness
        if 'category' in cleaned_df.columns:
            category_stats = cleaned_df['category'].value_counts()
            validation_results['category_validation'] = {
                'total_categories': len(category_stats),
                'category_distribution': category_stats.to_dict(),
                'normalization_effective': len(category_stats) <= 20  # Reasonable number of categories
            }
        
        # Check confidence score validation
        if 'confidence' in cleaned_df.columns:
            confidence_data = cleaned_df['confidence'].dropna()
            out_of_range = ((confidence_data < 0) | (confidence_data > 1)).sum()
            validation_results['confidence_validation'] = {
                'out_of_range_values': int(out_of_range),
                'validation_success': out_of_range == 0
            }
        
        # Calculate overall cleaning effectiveness score
        effectiveness_scores = []
        for validation in validation_results.values():
            if 'success' in str(validation).lower():
                # Look for success indicators
                success_indicators = [v for k, v in validation.items() if 'success' in k.lower()]
                if success_indicators:
                    effectiveness_scores.append(100 if success_indicators[0] else 0)
        
        overall_effectiveness = np.mean(effectiveness_scores) if effectiveness_scores else 0
        
        validation_results['overall_effectiveness'] = {
            'effectiveness_score': round(overall_effectiveness, 2),
            'cleaning_successful': overall_effectiveness >= 80
        }
        
        return validation_results
    
    def _count_location_variations(self, locations: pd.Series) -> int:
        """Count potential location variations that still exist"""
        
        unique_locations = locations.dropna().unique()
        variation_count = 0
        
        for i, loc1 in enumerate(unique_locations):
            for loc2 in unique_locations[i+1:]:
                similarity = SequenceMatcher(None, str(loc1).lower(), str(loc2).lower()).ratio()
                if 0.8 <= similarity < 1.0:  # Still similar but not identical
                    variation_count += 1
        
        return variation_count
    
    def run_comprehensive_cleaning(self, sample_size: Optional[int] = None, 
                                 full_dataset: bool = False) -> Dict[str, Any]:
        """Run the complete cleaning pipeline"""
        
        logger.info("Starting comprehensive data cleaning pipeline...")
        
        if full_dataset:
            # Clean entire dataset
            results = self.clean_and_validate_full_dataset()
            
            # Load cleaned data for validation
            if results and 'cleaned_data_path' in results:
                cleaned_df = pd.read_csv(results['cleaned_data_path'])
                validation_results = self.validate_cleaning_effectiveness(cleaned_df)
                results['validation_results'] = validation_results
        
        else:
            # Clean sample data
            cleaned_df = self.clean_sample_data(sample_size)
            
            # Save cleaned sample
            output_path = self.save_cleaned_data(cleaned_df)
            
            # Validate cleaning effectiveness
            validation_results = self.validate_cleaning_effectiveness(cleaned_df)
            
            results = {
                'cleaned_data_path': output_path,
                'final_record_count': len(cleaned_df),
                'cleaning_stats': {
                    'records_processed': self.cleaning_stats.records_processed,
                    'records_cleaned': self.cleaning_stats.records_cleaned,
                    'duplicates_removed': self.cleaning_stats.duplicates_removed,
                    'locations_standardized': self.cleaning_stats.locations_standardized,
                    'categories_normalized': self.cleaning_stats.categories_normalized,
                    'invalid_data_fixed': self.cleaning_stats.invalid_data_fixed,
                    'missing_data_filled': self.cleaning_stats.missing_data_filled
                },
                'validation_results': validation_results,
                'retention_rate': round((self.cleaning_stats.records_cleaned / self.cleaning_stats.records_processed) * 100, 2)
            }
        
        logger.info("Comprehensive cleaning completed!")
        return results


# Specialized cleaning utilities

class CompanyNameStandardizer:
    """Advanced company name standardization"""
    
    def __init__(self):
        self.company_suffixes = [
            'Inc.', 'Incorporated', 'Corp.', 'Corporation', 'Ltd.', 'Limited', 
            'LLC', 'Co.', 'Company', 'LP', 'LLP', 'PLC', 'AG', 'GmbH'
        ]
        self.company_mappings = {}
    
    def standardize_company_name(self, company_name: str) -> str:
        """Standardize company name format"""
        if pd.isna(company_name) or not isinstance(company_name, str):
            return company_name
        
        # Clean the name
        cleaned = company_name.strip()
        
        # Standardize suffixes
        for suffix in self.company_suffixes:
            # Remove multiple variations of the same suffix
            pattern = rf'\b{re.escape(suffix)}\b'
            if re.search(pattern, cleaned, re.IGNORECASE):
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()
                # Add back standardized suffix
                cleaned = f"{cleaned} {suffix}"
                break
        
        return cleaned


class DataQualityValidator:
    """Post-cleaning validation utilities"""
    
    @staticmethod
    def validate_critical_fields(df: pd.DataFrame, critical_fields: List[str]) -> Dict[str, Any]:
        """Validate that critical fields meet quality requirements"""
        
        validation_results = {}
        
        for field in critical_fields:
            if field in df.columns:
                missing_count = df[field].isnull().sum()
                completeness_rate = ((len(df) - missing_count) / len(df)) * 100
                
                validation_results[field] = {
                    'completeness_percentage': round(completeness_rate, 2),
                    'passes_validation': completeness_rate >= 95.0,  # 95% threshold
                    'missing_count': int(missing_count)
                }
        
        overall_critical_score = np.mean([
            result['completeness_percentage'] 
            for result in validation_results.values()
        ])
        
        validation_results['overall_critical_validation'] = {
            'average_completeness': round(overall_critical_score, 2),
            'all_fields_pass': all(result['passes_validation'] for result in validation_results.values()),
            'fields_analyzed': len(validation_results)
        }
        
        return validation_results
    
    @staticmethod
    def validate_business_rules(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate business-specific data rules"""
        
        validation_results = {}
        
        # Rule 1: Events should have reasonable confidence scores
        if 'confidence' in df.columns:
            confidence_data = df['confidence'].dropna()
            low_confidence_count = (confidence_data < 0.3).sum()
            high_confidence_count = (confidence_data >= 0.8).sum()
            
            validation_results['confidence_distribution'] = {
                'low_confidence_events': int(low_confidence_count),
                'high_confidence_events': int(high_confidence_count),
                'average_confidence': round(float(confidence_data.mean()), 3),
                'confidence_distribution_healthy': low_confidence_count < (len(confidence_data) * 0.1)
            }
        
        # Rule 2: Financial events should have positive amounts
        if 'amount_normalized' in df.columns:
            amount_data = df['amount_normalized'].dropna()
            negative_amounts = (amount_data < 0).sum()
            zero_amounts = (amount_data == 0).sum()
            
            validation_results['financial_data_validation'] = {
                'negative_amounts': int(negative_amounts),
                'zero_amounts': int(zero_amounts),
                'valid_amounts': int(len(amount_data) - negative_amounts),
                'amounts_validation_passed': negative_amounts == 0
            }
        
        # Rule 3: Events should have recent timestamps (business relevance)
        if 'found_at' in df.columns:
            date_data = pd.to_datetime(df['found_at'], errors='coerce').dropna()
            current_time = datetime.now(timezone.utc)
            
            # Check data recency
            days_old = (current_time - date_data.dt.tz_convert('UTC')).dt.days
            recent_events = (days_old <= 365).sum()  # Within last year
            very_old_events = (days_old > 1095).sum()  # Older than 3 years
            
            validation_results['temporal_validation'] = {
                'recent_events_count': int(recent_events),
                'very_old_events_count': int(very_old_events),
                'average_age_days': round(float(days_old.mean()), 1),
                'temporal_distribution_healthy': very_old_events < (len(date_data) * 0.05)
            }
        
        return validation_results


if __name__ == "__main__":
    # Initialize cleaner
    cleaner = NewsEventsDataCleaner()
    
    # Run comprehensive cleaning on sample
    logger.info("Running comprehensive cleaning on sample data...")
    results = cleaner.run_comprehensive_cleaning(sample_size=5000, full_dataset=False)
    
    # Print results
    print("\nData Cleaning Results:")
    print(f"Records processed: {results['cleaning_stats']['records_processed']:,}")
    print(f"Records cleaned: {results['cleaning_stats']['records_cleaned']:,}")
    print(f"Retention rate: {results['retention_rate']:.2f}%")
    print(f"Duplicates removed: {results['cleaning_stats']['duplicates_removed']:,}")
    print(f"Locations standardized: {results['cleaning_stats']['locations_standardized']:,}")
    print(f"Categories normalized: {results['cleaning_stats']['categories_normalized']:,}")
    
    # Validation results
    validation = results['validation_results']
    print(f"\nCleaning Effectiveness: {validation['overall_effectiveness']['effectiveness_score']:.1f}%")
    print(f"Cleaning successful: {validation['overall_effectiveness']['cleaning_successful']}")
    
    # Option to run full dataset cleaning
    # run_full = input("\nRun full dataset cleaning? (y/n): ").lower().strip()
    # if run_full == 'y':
    #     logger.info("Starting full dataset cleaning...")
    #     full_results = cleaner.run_comprehensive_cleaning(full_dataset=True)
    #     print(f"Full dataset cleaned: {full_results.get('final_record_count', 0):,} records")