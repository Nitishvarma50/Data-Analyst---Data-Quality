"""
Data Loader Utility for News Events JSONL Files
Handles large file processing with memory-efficient chunking
"""

import json
import jsonlines
import pandas as pd
from pathlib import Path
from typing import Generator, Dict, List, Any, Optional
from loguru import logger
import os


class NewsEventsDataLoader:
    """Efficiently load and process large JSONL files containing news events data"""
    
    def __init__(self, data_directory: str):
        self.data_directory = Path(data_directory)
        self.file_list = []
        self.total_files = 0
        self._scan_files()
    
    def _scan_files(self):
        """Scan directory for JSONL files"""
        if not self.data_directory.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_directory}")
        
        # Look for both .jsonl and .json files
        jsonl_files = list(self.data_directory.glob("*.jsonl"))
        json_files = list(self.data_directory.glob("*.json"))
        
        self.file_list = jsonl_files + json_files
        self.total_files = len(self.file_list)
        
        logger.info(f"Found {self.total_files} JSON/JSONL files in {self.data_directory}")
        
        # Log file sizes
        for file_path in self.file_list:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"  {file_path.name}: {size_mb:.1f} MB")

    def get_json_files(self) -> List[Path]:
            """
            Return the list of discovered JSON/JSONL files.
            Compatible with cleaner's expectation.
            """
            return self.file_list
    
    def load_sample_data(self, sample_size: int = 1000) -> List[Dict]:
        """Load a sample of records from all files for initial analysis"""
        sample_records = []
        records_per_file = max(1, sample_size // self.total_files)
        
        logger.info(f"Loading sample data: {records_per_file} records per file")
        
        for file_path in self.file_list:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    count = 0
                    for line in f:
                        if count >= records_per_file:
                            break
                        try:
                            record = json.loads(line.strip())
                            sample_records.append({
                                'source_file': file_path.name,
                                'record': record
                            })
                            count += 1
                        except json.JSONDecodeError as e:
                            logger.warning(f"JSON decode error in {file_path.name}: {e}")
                            continue
                            
            except Exception as e:
                logger.error(f"Error reading file {file_path.name}: {e}")
                continue
        
        logger.info(f"Loaded {len(sample_records)} sample records")
        return sample_records
    
    def stream_all_files(self, chunk_size: int = 1000) -> Generator[List[Dict], None, None]:
        """Stream all files in chunks for memory-efficient processing"""
        
        for file_path in self.file_list:
            logger.info(f"Processing file: {file_path.name}")
            
            chunk = []
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            record = json.loads(line.strip())
                            chunk.append({
                                'source_file': file_path.name,
                                'line_number': line_num,
                                'record': record
                            })
                            
                            if len(chunk) >= chunk_size:
                                yield chunk
                                chunk = []
                                
                        except json.JSONDecodeError as e:
                            logger.warning(f"JSON decode error in {file_path.name} line {line_num}: {e}")
                            continue
                    
                    # Yield remaining records in chunk
                    if chunk:
                        yield chunk
                        
            except Exception as e:
                logger.error(f"Error processing file {file_path.name}: {e}")
                continue
    
    def get_file_statistics(self) -> Dict[str, Any]:
        """Get basic statistics about the files in the dataset"""
        stats = {
            'total_files': self.total_files,
            'file_details': [],
            'total_size_mb': 0,
            'estimated_total_records': 0
        }
        
        for file_path in self.file_list:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            # Estimate record count by sampling first 100 lines
            estimated_records = self._estimate_record_count(file_path)
            
            file_info = {
                'filename': file_path.name,
                'size_mb': round(file_size_mb, 2),
                'estimated_records': estimated_records
            }
            
            stats['file_details'].append(file_info)
            stats['total_size_mb'] += file_size_mb
            stats['estimated_total_records'] += estimated_records
        
        stats['total_size_mb'] = round(stats['total_size_mb'], 2)
        
        return stats
    
    def _estimate_record_count(self, file_path: Path, sample_lines: int = 100) -> int:
        """Estimate total records in file based on sample"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sample_records = 0
                sample_size = 0
                
                for i, line in enumerate(f):
                    if i >= sample_lines:
                        break
                    sample_size += len(line.encode('utf-8'))
                    try:
                        json.loads(line.strip())
                        sample_records += 1
                    except json.JSONDecodeError:
                        continue
                
                if sample_records == 0:
                    return 0
                
                # Estimate based on file size
                total_file_size = file_path.stat().st_size
                avg_record_size = sample_size / sample_records
                estimated_total = int(total_file_size / avg_record_size)
                
                return estimated_total
                
        except Exception as e:
            logger.warning(f"Could not estimate record count for {file_path.name}: {e}")
            return 0


def flatten_news_event_record(record: Dict) -> List[Dict]:
    """
    Flatten a single news event record from nested JSON to flat structure
    
    Args:
        record: Single JSONL record containing data and included sections
        
    Returns:
        List of flattened event dictionaries
    """
    flattened_events = []
    
    # Create lookup for included items (companies and articles)
    included_lookup = {}
    for item in record.get('included', []):
        included_lookup[item.get('id')] = item
    
    # Process each event in the data array
    for event in record.get('data', []):
        flat_event = {
            # Core event attributes
            'event_id': event.get('id'),
            'event_type': event.get('type'),
            **event.get('attributes', {})
        }
        
        # Extract related company information
        company_ref = event.get('relationships', {}).get('company1', {}).get('data', {})
        company_id = company_ref.get('id')
        
        if company_id and company_id in included_lookup:
            company_data = included_lookup[company_id].get('attributes', {})
            flat_event.update({
                'company_id': company_id,
                'company_name': company_data.get('company_name'),
                'company_domain': company_data.get('domain'),
                'company_ticker': company_data.get('ticker')
            })
        
        # Extract related article information
        article_ref = event.get('relationships', {}).get('most_relevant_source', {}).get('data', {})
        article_id = article_ref.get('id')
        
        if article_id and article_id in included_lookup:
            article_data = included_lookup[article_id].get('attributes', {})
            flat_event.update({
                'article_id': article_id,
                'article_title': article_data.get('title'),
                'article_url': article_data.get('url'),
                'article_published_at': article_data.get('published_at'),
                'article_author': article_data.get('author'),
                'article_body': article_data.get('body'),
                'article_image_url': article_data.get('image_url')
            })
        
        # FIXED: Clean up nested structures to avoid issues in DataFrame
        # Collect changes first, then apply them to avoid "dictionary changed size" error
        updates_to_apply = {}
        keys_to_process = list(flat_event.keys())  # Create snapshot of keys
        
        for key in keys_to_process:
            value = flat_event[key]
            if isinstance(value, (list, dict)):
                if key == 'location_data' and isinstance(value, list) and value:
                    # Extract first location data
                    updates_to_apply[f'{key}_first'] = value[0] if value else None
                    updates_to_apply[key] = str(value) if value else None
                elif isinstance(value, list):
                    updates_to_apply[key] = str(value) if value else None
                else:
                    updates_to_apply[key] = str(value) if value else None
        
        # Apply all updates at once
        flat_event.update(updates_to_apply)
        
        # ADDITIONAL FIX: Ensure no unhashable types remain
        final_cleanup = {}
        for key, value in flat_event.items():
            if isinstance(value, (dict, list, set, tuple)):
                final_cleanup[key] = str(value) if value is not None else None
            else:
                final_cleanup[key] = value
        
        flat_event = final_cleanup
        
        flattened_events.append(flat_event)
    
    return flattened_events


if __name__ == "__main__":
    # Test the data loader
    loader = NewsEventsDataLoader("E:/Nitish Files/Datasets-2025-08-08/Datasets-2025-08-08")
    
    # Get file statistics
    stats = loader.get_file_statistics()
    print("File Statistics:")
    print(f"Total files: {stats['total_files']}")
    print(f"Total size: {stats['total_size_mb']} MB")
    print(f"Estimated records: {stats['estimated_total_records']:,}")
    
    # Load sample data
    sample_data = loader.load_sample_data(sample_size=500)
    print(f"Loaded {len(sample_data)} sample records")