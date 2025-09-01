# News Events Dataset - Data Profiling Report
Generated: 2025-09-01 10:26:44

## Dataset Overview
- **Source Directory**: E:/Nitish Files/Datasets-2025-08-08/Datasets-2025-08-08
- **Sample Records Analyzed**: 4,992
- **Total Columns**: 44
- **Memory Usage (Sample)**: 27.6 MB

## Data Structure Analysis

### Column Distribution by Type
- Text Columns: 39
- Numeric Columns: 3
- Boolean Columns: 2
- DateTime Columns: 0

### Key Columns Identified
**Core Event Fields**: event_id, summary, category, found_at, confidence, company_name

**Extended Fields**: 38 additional fields including location, amounts, products, etc.

## Event Categories Analysis
- **Total Categories**: 29
- **Most Common Event Types**:
  - launches: 1615 events
  - partners_with: 1024 events
  - hires: 563 events
  - recognized_as: 223 events
  - receives_award: 166 events

## Company Coverage Analysis
- **Total Companies**: 1,259
- **Companies with Multiple Events**: 305
- **Single Event Companies**: 954
- **Top Active Companies**:
  - Apple: 189 events
  - DHL Express: 161 events
  - Amazon.com, Inc.: 157 events
  - Marriott International, Inc.: 143 events
  - Ericsson: 137 events

## Confidence Score Analysis
- **Mean Confidence**: 0.6166
- **High Confidence Events** (≥0.9): 712
- **Medium Confidence Events** (0.7-0.9): 1,379
- **Low Confidence Events** (<0.7): 2,901

## Temporal Analysis
- **Date Range**: 2010-12-25T23:00:00+00:00 to 2025-07-02T17:30:34+00:00
- **Time Span**: 5,302 days
- **Recent Activity**:
  - Last 30 days: 0 events
  - Last 90 days: 38 events
  - Last year: 473 events

## Financial Events Analysis
- **Total Financial Events**: 283
- **Total Amount**: $393,019,460,000.00
- **Average Amount**: $1,388,761,342.76
- **Median Amount**: $61,000,000.00

## Data Quality Preview
### Missing Data Analysis
- **company_name**: 91 missing (1.82%)

### Duplicate Issues Preview
- **Duplicate Event IDs**: 1
- **Duplicate Summaries**: 17
- **Duplication Rate**: 0.02%
