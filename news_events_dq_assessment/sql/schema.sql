-- News Events Data Quality Assessment - Database Schema
-- Optimized for storing cleaned news events data with quality tracking

-- =====================================================
-- CORE ENTITIES TABLES
-- =====================================================

-- Companies table - stores unique company information
CREATE TABLE companies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id VARCHAR(50) UNIQUE NOT NULL,  -- From original data
    name VARCHAR(500) NOT NULL,
    domain VARCHAR(255),
    ticker_symbol VARCHAR(10),
    industry VARCHAR(100),
    headquarters_location VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- News articles table - stores source article information
CREATE TABLE news_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id VARCHAR(50) UNIQUE NOT NULL,  -- From original data
    title TEXT,
    url TEXT,
    publication_date TIMESTAMP,
    source VARCHAR(255),
    language VARCHAR(10) DEFAULT 'en',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Events table - main events data with quality tracking
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id VARCHAR(50) UNIQUE NOT NULL,
    summary TEXT NOT NULL,
    category VARCHAR(100) NOT NULL,
    confidence DECIMAL(3,2) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    found_at TIMESTAMP NOT NULL,
    effective_date DATE,
    detection_lag_days INTEGER,  -- Calculated: found_at - effective_date
    
    -- Location information (standardized during cleaning)
    location VARCHAR(255),
    country VARCHAR(100),
    region VARCHAR(100),
    
    -- Financial information
    amount DECIMAL(15,2),
    currency VARCHAR(3) DEFAULT 'USD',
    
    -- Additional structured data
    contact_info TEXT,  -- JSON string for contact details
    award_info TEXT,    -- JSON string for award details
    product_info TEXT,  -- JSON string for product details
    
    -- Quality tracking fields
    data_quality_score DECIMAL(5,2),
    is_cleaned BOOLEAN DEFAULT FALSE,
    cleaning_applied TEXT,  -- Comma-separated list of cleaning operations
    quality_flags TEXT,     -- JSON string for quality indicators
    
    -- Audit fields
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CHECK (detection_lag_days >= 0)
);

-- =====================================================
-- RELATIONSHIP TABLES (Many-to-Many)
-- =====================================================

-- Event-Company relationships
CREATE TABLE event_companies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id VARCHAR(50) NOT NULL,
    company_id VARCHAR(50) NOT NULL,
    relationship_type VARCHAR(50) DEFAULT 'primary',  -- primary, secondary, mentioned
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE,
    FOREIGN KEY (company_id) REFERENCES companies(company_id) ON DELETE CASCADE,
    UNIQUE(event_id, company_id, relationship_type)
);

-- Event-Article relationships
CREATE TABLE event_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id VARCHAR(50) NOT NULL,
    article_id VARCHAR(50) NOT NULL,
    extraction_confidence DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE,
    FOREIGN KEY (article_id) REFERENCES news_articles(article_id) ON DELETE CASCADE,
    UNIQUE(event_id, article_id)
);

-- =====================================================
-- DATA QUALITY TRACKING TABLES
-- =====================================================

-- Quality assessments history
CREATE TABLE quality_assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    assessment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_records INTEGER NOT NULL,
    sample_size INTEGER NOT NULL,
    
    -- Dimension scores
    completeness_score DECIMAL(5,2),
    uniqueness_score DECIMAL(5,2),
    validity_score DECIMAL(5,2),
    consistency_score DECIMAL(5,2),
    timeliness_score DECIMAL(5,2),
    overall_score DECIMAL(5,2),
    
    -- Assessment metadata
    config_version VARCHAR(20),
    assessment_type VARCHAR(50) DEFAULT 'full',  -- full, incremental, sample
    notes TEXT
);

-- Data cleaning operations log
CREATE TABLE cleaning_operations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    operation_type VARCHAR(100) NOT NULL,  -- deduplication, standardization, validation, etc.
    records_processed INTEGER,
    records_modified INTEGER,
    success_rate DECIMAL(5,2),
    execution_time_seconds DECIMAL(8,2),
    details TEXT,  -- JSON string with operation details
    
    -- Link to quality assessment
    assessment_id INTEGER,
    FOREIGN KEY (assessment_id) REFERENCES quality_assessments(id)
);

-- Data quality metrics by dimension
CREATE TABLE quality_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    assessment_id INTEGER NOT NULL,
    dimension VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,4),
    threshold_value DECIMAL(10,4),
    is_passing BOOLEAN,
    measured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (assessment_id) REFERENCES quality_assessments(id),
    UNIQUE(assessment_id, dimension, metric_name)
);

-- =====================================================
-- REFERENCE DATA TABLES
-- =====================================================

-- Standardized categories lookup
CREATE TABLE standard_categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    standard_name VARCHAR(100) UNIQUE NOT NULL,
    variations TEXT,  -- JSON array of variations/aliases
    description TEXT,
    business_impact VARCHAR(50),  -- high, medium, low
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Standardized locations lookup
CREATE TABLE standard_locations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    standard_name VARCHAR(255) UNIQUE NOT NULL,
    country VARCHAR(100),
    region VARCHAR(100),
    variations TEXT,  -- JSON array of variations/aliases
    latitude DECIMAL(10,8),
    longitude DECIMAL(11,8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================

-- Core business queries
CREATE INDEX idx_events_category ON events(category);
CREATE INDEX idx_events_found_at ON events(found_at);
CREATE INDEX idx_events_effective_date ON events(effective_date);
CREATE INDEX idx_events_confidence ON events(confidence);
CREATE INDEX idx_events_location ON events(location);
CREATE INDEX idx_events_quality_score ON events(data_quality_score);

-- Relationship queries
CREATE INDEX idx_event_companies_event ON event_companies(event_id);
CREATE INDEX idx_event_companies_company ON event_companies(company_id);
CREATE INDEX idx_event_articles_event ON event_articles(event_id);
CREATE INDEX idx_event_articles_article ON event_articles(article_id);

-- Company queries
CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_domain ON companies(domain);
CREATE INDEX idx_companies_ticker ON companies(ticker_symbol);

-- Quality monitoring queries
CREATE INDEX idx_quality_assessments_date ON quality_assessments(assessment_date);
CREATE INDEX idx_quality_metrics_dimension ON quality_metrics(dimension);
CREATE INDEX idx_cleaning_operations_type ON cleaning_operations(operation_type);
CREATE INDEX idx_cleaning_operations_date ON cleaning_operations(operation_date);

-- =====================================================
-- VIEWS FOR COMMON QUERIES
-- =====================================================

-- Events with company information
CREATE VIEW events_with_companies AS
SELECT 
    e.event_id,
    e.summary,
    e.category,
    e.confidence,
    e.found_at,
    e.effective_date,
    e.location,
    e.amount,
    e.currency,
    e.data_quality_score,
    c.name as company_name,
    c.domain as company_domain,
    c.ticker_symbol,
    ec.relationship_type
FROM events e
JOIN event_companies ec ON e.event_id = ec.event_id
JOIN companies c ON ec.company_id = c.company_id;

-- Latest quality assessment summary
CREATE VIEW latest_quality_summary AS
SELECT 
    qa.assessment_date,
    qa.total_records,
    qa.sample_size,
    qa.completeness_score,
    qa.uniqueness_score,
    qa.validity_score,
    qa.consistency_score,
    qa.timeliness_score,
    qa.overall_score,
    COUNT(co.id) as cleaning_operations_count
FROM quality_assessments qa
LEFT JOIN cleaning_operations co ON qa.id = co.assessment_id
WHERE qa.assessment_date = (SELECT MAX(assessment_date) FROM quality_assessments)
GROUP BY qa.id;

-- Data freshness monitoring
CREATE VIEW data_freshness_summary AS
SELECT 
    category,
    COUNT(*) as event_count,
    AVG(detection_lag_days) as avg_detection_lag,
    MAX(found_at) as latest_event,
    MIN(found_at) as oldest_event,
    COUNT(CASE WHEN detection_lag_days <= 7 THEN 1 END) as recent_events,
    ROUND(COUNT(CASE WHEN detection_lag_days <= 7 THEN 1 END) * 100.0 / COUNT(*), 2) as freshness_percentage
FROM events 
WHERE effective_date IS NOT NULL
GROUP BY category
ORDER BY freshness_percentage DESC;

-- Quality trends over time
CREATE VIEW quality_trends AS
SELECT 
    DATE(assessment_date) as assessment_date,
    completeness_score,
    uniqueness_score,
    validity_score,
    consistency_score,
    timeliness_score,
    overall_score,
    total_records
FROM quality_assessments
ORDER BY assessment_date DESC;

-- =====================================================
-- TRIGGERS FOR AUDIT TRAILS
-- =====================================================

-- Update timestamp trigger for events
CREATE TRIGGER events_update_timestamp 
    AFTER UPDATE ON events
    FOR EACH ROW
BEGIN
    UPDATE events SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Update timestamp trigger for companies
CREATE TRIGGER companies_update_timestamp 
    AFTER UPDATE ON companies
    FOR EACH ROW
BEGIN
    UPDATE companies SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- =====================================================
-- SAMPLE DATA INSERTION HELPERS
-- =====================================================

-- Insert standard categories (based on your data)
INSERT INTO standard_categories (standard_name, variations, description, business_impact) VALUES
('receives_award', '["receives_award", "awarded", "recognition"]', 'Company receives awards or recognition', 'medium'),
('launches', '["launches", "launch", "introduces", "releases"]', 'Product or service launches', 'high'),
('hires', '["hires", "hiring", "recruits", "appoints"]', 'Personnel changes and hiring', 'medium'),
('receives_financing', '["receives_financing", "funding", "investment", "raises"]', 'Funding and investment events', 'high'),
('is_developing', '["is_developing", "developing", "building", "creating"]', 'Product development activities', 'medium');

-- =====================================================
-- USEFUL QUERIES FOR YOUR ANALYSIS
-- =====================================================

-- Query 1: Events by category with quality metrics
-- SELECT category, COUNT(*) as count, AVG(confidence) as avg_confidence, 
--        AVG(data_quality_score) as avg_quality FROM events GROUP BY category;

-- Query 2: Top companies by event count
-- SELECT c.name, COUNT(e.event_id) as event_count, AVG(e.confidence) as avg_confidence
-- FROM companies c 
-- JOIN event_companies ec ON c.company_id = ec.company_id
-- JOIN events e ON ec.event_id = e.event_id
-- GROUP BY c.name ORDER BY event_count DESC LIMIT 10;

-- Query 3: Data quality issues by location
-- SELECT location, COUNT(*) as event_count, AVG(data_quality_score) as avg_quality
-- FROM events WHERE location IS NOT NULL 
-- GROUP BY location HAVING COUNT(*) > 5 ORDER BY avg_quality ASC;

-- Query 4: Recent high-confidence events
-- SELECT event_id, summary, category, confidence, found_at, company_name
-- FROM events_with_companies 
-- WHERE confidence > 0.8 AND found_at > datetime('now', '-30 days')
-- ORDER BY found_at DESC;