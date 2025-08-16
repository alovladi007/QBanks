-- =====================================================
-- sql/01_core_schema.sql
-- Core database setup and extensions
-- =====================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "ltree";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS qbank;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS audit;

-- Set search path
SET search_path TO qbank, public;

-- =====================================================
-- sql/02_content_ddl.sql
-- Content management tables with versioning
-- =====================================================

-- Topic hierarchy with ltree for efficient queries
CREATE TABLE topics (
    id BIGSERIAL PRIMARY KEY,
    tenant_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000001'::uuid,
    parent_id BIGINT REFERENCES topics(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    blueprint_code VARCHAR(50),
    description TEXT,
    weight DECIMAL(3,2) DEFAULT 1.0 CHECK (weight BETWEEN 0 AND 1),
    path LTREE,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT uq_topic_blueprint UNIQUE(tenant_id, blueprint_code)
);

CREATE INDEX idx_topics_tenant ON topics(tenant_id);
CREATE INDEX idx_topics_parent ON topics(parent_id);
CREATE INDEX idx_topics_blueprint ON topics(blueprint_code) WHERE blueprint_code IS NOT NULL;
CREATE INDEX idx_topics_path ON topics USING GIST(path);
CREATE INDEX idx_topics_metadata ON topics USING GIN(metadata);

-- Questions with soft delete and audit trail
CREATE TABLE questions (
    id BIGSERIAL PRIMARY KEY,
    tenant_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000001'::uuid,
    external_ref VARCHAR(100),
    created_by VARCHAR(255) NOT NULL,
    reviewed_by VARCHAR(255),
    is_deleted BOOLEAN DEFAULT FALSE,
    deleted_by VARCHAR(255),
    deleted_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT uq_question_external_ref UNIQUE(tenant_id, external_ref)
);

CREATE INDEX idx_questions_tenant ON questions(tenant_id);
CREATE INDEX idx_questions_external_ref ON questions(external_ref) WHERE external_ref IS NOT NULL;
CREATE INDEX idx_questions_created_by ON questions(created_by);
CREATE INDEX idx_questions_deleted ON questions(is_deleted, deleted_at) WHERE is_deleted = TRUE;

-- Question versions with full-text search and embeddings
CREATE TABLE question_versions (
    id BIGSERIAL PRIMARY KEY,
    question_id BIGINT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    version INT NOT NULL CHECK (version > 0),
    state VARCHAR(20) NOT NULL DEFAULT 'draft',
    stem_md TEXT NOT NULL,
    lead_in TEXT NOT NULL,
    rationale_md TEXT NOT NULL,
    difficulty_label VARCHAR(20),
    bloom_level INT CHECK (bloom_level BETWEEN 1 AND 6),
    topic_id BIGINT REFERENCES topics(id) ON DELETE SET NULL,
    tags TEXT[] DEFAULT '{}',
    assets JSONB DEFAULT '[]',
    references JSONB DEFAULT '[]',
    search_vector TSVECTOR,
    embedding vector(768),  -- For semantic search
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    reviewed_by VARCHAR(255),
    reviewed_at TIMESTAMPTZ,
    approved_by VARCHAR(255),
    approved_at TIMESTAMPTZ,
    
    CONSTRAINT uq_question_version UNIQUE(question_id, version),
    CONSTRAINT ck_question_state CHECK (state IN ('draft', 'in_review', 'approved', 'published', 'archived'))
);

CREATE INDEX idx_qv_question ON question_versions(question_id);
CREATE INDEX idx_qv_topic ON question_versions(topic_id);
CREATE INDEX idx_qv_state ON question_versions(state);
CREATE INDEX idx_qv_version ON question_versions(version);
CREATE INDEX idx_qv_tags ON question_versions USING GIN(tags);
CREATE INDEX idx_qv_search ON question_versions USING GIN(search_vector);
CREATE INDEX idx_qv_embedding ON question_versions USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_qv_metadata ON question_versions USING GIN(metadata);

-- Trigger to update search vector
CREATE OR REPLACE FUNCTION update_question_search_vector() RETURNS trigger AS $$
BEGIN
    NEW.search_vector := 
        setweight(to_tsvector('english', COALESCE(NEW.stem_md, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(NEW.lead_in, '')), 'B') ||
        setweight(to_tsvector('english', COALESCE(NEW.rationale_md, '')), 'C') ||
        setweight(to_tsvector('english', COALESCE(array_to_string(NEW.tags, ' '), '')), 'D');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_question_search_vector
    BEFORE INSERT OR UPDATE ON question_versions
    FOR EACH ROW EXECUTE FUNCTION update_question_search_vector();

-- Question options with distractor analysis support
CREATE TABLE question_options (
    id BIGSERIAL PRIMARY KEY,
    question_version_id BIGINT NOT NULL REFERENCES question_versions(id) ON DELETE CASCADE,
    option_label CHAR(1) NOT NULL CHECK (option_label IN ('A', 'B', 'C', 'D', 'E', 'F')),
    option_text_md TEXT NOT NULL,
    is_correct BOOLEAN NOT NULL DEFAULT FALSE,
    explanation_md TEXT,
    metadata JSONB DEFAULT '{}',
    
    CONSTRAINT uq_question_option UNIQUE(question_version_id, option_label)
);

CREATE INDEX idx_qo_question_version ON question_options(question_version_id);
CREATE INDEX idx_qo_correct ON question_options(is_correct) WHERE is_correct = TRUE;

-- Question publications for exam management
CREATE TABLE question_publications (
    id BIGSERIAL PRIMARY KEY,
    question_id BIGINT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    live_version INT NOT NULL,
    exam_code VARCHAR(50) NOT NULL,
    tenant_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000001'::uuid,
    published_at TIMESTAMPTZ DEFAULT NOW(),
    published_by VARCHAR(255) NOT NULL,
    expires_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB DEFAULT '{}',
    
    CONSTRAINT uq_question_publication UNIQUE(question_id, exam_code),
    CONSTRAINT fk_publication_version FOREIGN KEY (question_id, live_version) 
        REFERENCES question_versions(question_id, version)
);

CREATE INDEX idx_qp_question ON question_publications(question_id);
CREATE INDEX idx_qp_exam_code ON question_publications(exam_code);
CREATE INDEX idx_qp_tenant ON question_publications(tenant_id);
CREATE INDEX idx_qp_active ON question_publications(is_active, expires_at) WHERE is_active = TRUE;

-- =====================================================
-- sql/03_delivery_ddl.sql
-- Quiz delivery and response tracking
-- =====================================================

-- Quiz sessions with comprehensive tracking
CREATE TABLE quiz_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    tenant_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000001'::uuid,
    mode VARCHAR(20) NOT NULL DEFAULT 'practice',
    adaptive BOOLEAN DEFAULT FALSE,
    exam_code VARCHAR(50),
    config JSONB DEFAULT '{}',
    started_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    sealed_at TIMESTAMPTZ,
    score DECIMAL(5,2),
    percentile INT,
    time_taken_seconds INT,
    metadata JSONB DEFAULT '{}',
    
    CONSTRAINT ck_quiz_mode CHECK (mode IN ('tutor', 'exam', 'practice', 'diagnostic')),
    CONSTRAINT ck_quiz_expires CHECK (expires_at > started_at),
    CONSTRAINT ck_quiz_score CHECK (score BETWEEN 0 AND 100)
);

CREATE INDEX idx_qs_user ON quiz_sessions(user_id);
CREATE INDEX idx_qs_tenant ON quiz_sessions(tenant_id);
CREATE INDEX idx_qs_started ON quiz_sessions(started_at DESC);
CREATE INDEX idx_qs_mode ON quiz_sessions(mode);
CREATE INDEX idx_qs_exam_code ON quiz_sessions(exam_code) WHERE exam_code IS NOT NULL;
CREATE INDEX idx_qs_active ON quiz_sessions(expires_at) WHERE completed_at IS NULL;

-- Quiz items with position tracking
CREATE TABLE quiz_items (
    id BIGSERIAL PRIMARY KEY,
    quiz_id UUID NOT NULL REFERENCES quiz_sessions(id) ON DELETE CASCADE,
    question_id BIGINT NOT NULL,
    version INT NOT NULL,
    position INT NOT NULL CHECK (position > 0),
    served_at TIMESTAMPTZ DEFAULT NOW(),
    theta_before DECIMAL(4,3),  -- Ability estimate before
    theta_after DECIMAL(4,3),   -- Ability estimate after
    information DECIMAL(6,4),   -- Fisher information
    metadata JSONB DEFAULT '{}',
    
    CONSTRAINT uq_quiz_item_position UNIQUE(quiz_id, position),
    CONSTRAINT fk_quiz_item_question FOREIGN KEY (question_id, version)
        REFERENCES question_versions(question_id, version)
);

CREATE INDEX idx_qi_quiz ON quiz_items(quiz_id);
CREATE INDEX idx_qi_question ON quiz_items(question_id, version);
CREATE INDEX idx_qi_served ON quiz_items(served_at);

-- User responses with detailed tracking
CREATE TABLE user_responses (
    id BIGSERIAL PRIMARY KEY,
    quiz_id UUID NOT NULL REFERENCES quiz_sessions(id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL,
    question_id BIGINT NOT NULL,
    version INT NOT NULL,
    option_label CHAR(1) NOT NULL,
    is_correct BOOLEAN NOT NULL,
    time_taken_ms INT CHECK (time_taken_ms >= 0),
    confidence INT CHECK (confidence BETWEEN 1 AND 5),
    flagged BOOLEAN DEFAULT FALSE,
    note TEXT,
    client_info JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT uq_user_response UNIQUE(quiz_id, question_id),
    CONSTRAINT fk_response_question FOREIGN KEY (question_id, version)
        REFERENCES question_versions(question_id, version)
);

CREATE INDEX idx_ur_quiz ON user_responses(quiz_id);
CREATE INDEX idx_ur_user ON user_responses(user_id);
CREATE INDEX idx_ur_question ON user_responses(question_id, version);
CREATE INDEX idx_ur_created ON user_responses(created_at DESC);
CREATE INDEX idx_ur_correct ON user_responses(is_correct);
CREATE INDEX idx_ur_flagged ON user_responses(flagged) WHERE flagged = TRUE;

-- =====================================================
-- sql/04_analytics_ddl.sql
-- Analytics and psychometric tables
-- =====================================================

-- Item calibration with multiple IRT models
CREATE TABLE item_calibration (
    question_id BIGINT NOT NULL,
    version INT NOT NULL,
    model VARCHAR(10) NOT NULL,
    a DECIMAL(5,3),  -- Discrimination
    b DECIMAL(5,3),  -- Difficulty
    c DECIMAL(5,3),  -- Guessing
    se_a DECIMAL(5,3),  -- Standard errors
    se_b DECIMAL(5,3),
    se_c DECIMAL(5,3),
    n_respondents INT,
    fit_statistics JSONB DEFAULT '{}',
    calibrated_at TIMESTAMPTZ DEFAULT NOW(),
    
    PRIMARY KEY (question_id, version, model),
    CONSTRAINT ck_calibration_model CHECK (model IN ('CTT', 'Rasch', '1PL', '2PL', '3PL', 'GRM')),
    CONSTRAINT fk_calibration_question FOREIGN KEY (question_id, version)
        REFERENCES question_versions(question_id, version)
);

CREATE INDEX idx_ic_question ON item_calibration(question_id, version);
CREATE INDEX idx_ic_model ON item_calibration(model);
CREATE INDEX idx_ic_calibrated ON item_calibration(calibrated_at DESC);

-- Classical Test Theory metrics
CREATE TABLE item_statistics (
    question_id BIGINT NOT NULL,
    version INT NOT NULL,
    p_value DECIMAL(5,4) CHECK (p_value BETWEEN 0 AND 1),
    discrimination DECIMAL(5,4) CHECK (discrimination BETWEEN -1 AND 1),
    point_biserial DECIMAL(5,4) CHECK (point_biserial BETWEEN -1 AND 1),
    distractor_analysis JSONB DEFAULT '{}',
    response_time_mean INT,
    response_time_median INT,
    response_time_std INT,
    n_responses INT,
    calculated_at TIMESTAMPTZ DEFAULT NOW(),
    
    PRIMARY KEY (question_id, version),
    CONSTRAINT fk_statistics_question FOREIGN KEY (question_id, version)
        REFERENCES question_versions(question_id, version)
);

CREATE INDEX idx_is_question ON item_statistics(question_id, version);
CREATE INDEX idx_is_p_value ON item_statistics(p_value);
CREATE INDEX idx_is_discrimination ON item_statistics(discrimination);

-- Exposure control with Sympson-Hetter
CREATE TABLE item_exposure_control (
    question_id BIGINT NOT NULL,
    version INT NOT NULL,
    sh_p DECIMAL(5,4) DEFAULT 1.0 CHECK (sh_p BETWEEN 0 AND 1),
    exposure_count INT DEFAULT 0,
    exposure_rate DECIMAL(5,4),
    last_calibrated TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    PRIMARY KEY (question_id, version),
    CONSTRAINT fk_exposure_question FOREIGN KEY (question_id, version)
        REFERENCES question_versions(question_id, version)
);

CREATE INDEX idx_iec_question ON item_exposure_control(question_id, version);
CREATE INDEX idx_iec_sh_p ON item_exposure_control(sh_p);
CREATE INDEX idx_iec_updated ON item_exposure_control(updated_at DESC);

-- User ability tracking (theta estimates)
CREATE TABLE user_abilities (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    topic_id BIGINT REFERENCES topics(id) ON DELETE CASCADE,
    theta DECIMAL(5,3) DEFAULT 0.0,
    theta_se DECIMAL(5,3) DEFAULT 1.0,
    n_responses INT DEFAULT 0,
    last_quiz_id UUID REFERENCES quiz_sessions(id) ON DELETE SET NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT uq_user_ability UNIQUE(user_id, topic_id)
);

CREATE INDEX idx_ua_user ON user_abilities(user_id);
CREATE INDEX idx_ua_topic ON user_abilities(topic_id);
CREATE INDEX idx_ua_updated ON user_abilities(updated_at DESC);

-- DIF (Differential Item Functioning) analysis
CREATE TABLE item_dif_analysis (
    question_id BIGINT NOT NULL,
    version INT NOT NULL,
    group_type VARCHAR(50) NOT NULL,
    group_value VARCHAR(100) NOT NULL,
    dif_value DECIMAL(5,3),
    dif_type VARCHAR(20),
    n_focal INT,
    n_reference INT,
    significance DECIMAL(5,4),
    calculated_at TIMESTAMPTZ DEFAULT NOW(),
    
    PRIMARY KEY (question_id, version, group_type, group_value),
    CONSTRAINT fk_dif_question FOREIGN KEY (question_id, version)
        REFERENCES question_versions(question_id, version)
);

CREATE INDEX idx_dif_question ON item_dif_analysis(question_id, version);
CREATE INDEX idx_dif_group ON item_dif_analysis(group_type, group_value);

-- =====================================================
-- sql/05_calibration_ddl.sql
-- Calibration run management
-- =====================================================

-- Calibration runs with full tracking
CREATE TABLE calibration_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    exam_code VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'queued',
    run_type VARCHAR(20) NOT NULL DEFAULT 'full',
    params JSONB NOT NULL,
    history JSONB DEFAULT '[]',
    result JSONB,
    error TEXT,
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    job_id VARCHAR(100),
    
    CONSTRAINT ck_calibration_status CHECK (status IN ('queued', 'running', 'done', 'failed', 'cancelled')),
    CONSTRAINT ck_calibration_type CHECK (run_type IN ('full', 'incremental', 'exposure', 'dif'))
);

CREATE INDEX idx_cr_exam ON calibration_runs(exam_code);
CREATE INDEX idx_cr_status ON calibration_runs(status);
CREATE INDEX idx_cr_created ON calibration_runs(created_at DESC);
CREATE INDEX idx_cr_job ON calibration_runs(job_id) WHERE job_id IS NOT NULL;

-- =====================================================
-- sql/06_governance_ddl.sql
-- Security, audit, and multi-tenancy
-- =====================================================

-- Feature flags for gradual rollout
CREATE TABLE feature_flags (
    key VARCHAR(100) PRIMARY KEY,
    enabled BOOLEAN DEFAULT TRUE,
    value_json JSONB DEFAULT '{}',
    rollout_percentage INT DEFAULT 100 CHECK (rollout_percentage BETWEEN 0 AND 100),
    whitelist_users TEXT[] DEFAULT '{}',
    blacklist_users TEXT[] DEFAULT '{}',
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    updated_by VARCHAR(255)
);

CREATE INDEX idx_ff_enabled ON feature_flags(enabled) WHERE enabled = TRUE;

-- A/B testing cohort assignments
CREATE TABLE cohort_assignments (
    user_id VARCHAR(255) NOT NULL,
    cohort_key VARCHAR(100) NOT NULL,
    cohort_value VARCHAR(255) NOT NULL,
    assigned_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    
    PRIMARY KEY (user_id, cohort_key)
);

CREATE INDEX idx_ca_user ON cohort_assignments(user_id);
CREATE INDEX idx_ca_cohort ON cohort_assignments(cohort_key, cohort_value);

-- Audit log for compliance
CREATE TABLE audit.audit_logs (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    action VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(255) NOT NULL,
    changes JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_al_user ON audit.audit_logs(user_id);
CREATE INDEX idx_al_entity ON audit.audit_logs(entity_type, entity_id);
CREATE INDEX idx_al_created ON audit.audit_logs(created_at DESC);
CREATE INDEX idx_al_action ON audit.audit_logs(action);

-- Row-level security policies
ALTER TABLE questions ENABLE ROW LEVEL SECURITY;
ALTER TABLE question_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE quiz_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_responses ENABLE ROW LEVEL SECURITY;

-- Policy for tenant isolation
CREATE POLICY tenant_isolation_questions ON questions
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

CREATE POLICY tenant_isolation_quiz_sessions ON quiz_sessions
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =====================================================
-- sql/07_indexes.sql
-- Performance optimization indexes
-- =====================================================

-- Composite indexes for common queries
CREATE INDEX idx_qv_state_topic ON question_versions(state, topic_id) WHERE state = 'published';
CREATE INDEX idx_ur_user_correct ON user_responses(user_id, is_correct);
CREATE INDEX idx_ur_question_correct ON user_responses(question_id, version, is_correct);
CREATE INDEX idx_qs_user_mode_date ON quiz_sessions(user_id, mode, started_at DESC);

-- Partial indexes for performance
CREATE INDEX idx_questions_active ON questions(tenant_id, id) WHERE is_deleted = FALSE;
CREATE INDEX idx_qv_published ON question_versions(question_id, version) WHERE state = 'published';
CREATE INDEX idx_qs_incomplete ON quiz_sessions(user_id, expires_at) WHERE completed_at IS NULL;

-- BRIN indexes for time-series data
CREATE INDEX idx_ur_created_brin ON user_responses USING BRIN(created_at);
CREATE INDEX idx_qs_started_brin ON quiz_sessions USING BRIN(started_at);

-- =====================================================
-- sql/08_functions.sql
-- Stored procedures and functions
-- =====================================================

-- Function to calculate item statistics
CREATE OR REPLACE FUNCTION calculate_item_statistics(p_question_id BIGINT, p_version INT)
RETURNS TABLE (
    p_value DECIMAL,
    discrimination DECIMAL,
    point_biserial DECIMAL,
    n_responses INT
) AS $$
DECLARE
    v_p_value DECIMAL;
    v_discrimination DECIMAL;
    v_point_biserial DECIMAL;
    v_n_responses INT;
BEGIN
    -- Calculate p-value (difficulty)
    SELECT 
        AVG(CASE WHEN is_correct THEN 1 ELSE 0 END)::DECIMAL,
        COUNT(*)
    INTO v_p_value, v_n_responses
    FROM user_responses
    WHERE question_id = p_question_id AND version = p_version;
    
    -- Calculate discrimination (simplified)
    WITH user_scores AS (
        SELECT 
            ur.user_id,
            ur.is_correct AS item_correct,
            AVG(CASE WHEN ur2.is_correct THEN 1 ELSE 0 END) AS total_score
        FROM user_responses ur
        JOIN user_responses ur2 ON ur.quiz_id = ur2.quiz_id
        WHERE ur.question_id = p_question_id AND ur.version = p_version
        GROUP BY ur.user_id, ur.is_correct
    )
    SELECT 
        CORR(item_correct::INT, total_score)::DECIMAL
    INTO v_discrimination
    FROM user_scores;
    
    v_point_biserial := v_discrimination; -- Simplified
    
    RETURN QUERY SELECT v_p_value, v_discrimination, v_point_biserial, v_n_responses;
END;
$$ LANGUAGE plpgsql;

-- Function to update user ability after response
CREATE OR REPLACE FUNCTION update_user_ability(
    p_user_id VARCHAR(255),
    p_topic_id BIGINT,
    p_is_correct BOOLEAN,
    p_item_difficulty DECIMAL
) RETURNS VOID AS $$
DECLARE
    v_current_theta DECIMAL;
    v_current_se DECIMAL;
    v_new_theta DECIMAL;
    v_new_se DECIMAL;
    v_n_responses INT;
BEGIN
    -- Get current ability
    SELECT theta, theta_se, n_responses
    INTO v_current_theta, v_current_se, v_n_responses
    FROM user_abilities
    WHERE user_id = p_user_id AND topic_id = p_topic_id;
    
    IF NOT FOUND THEN
        v_current_theta := 0.0;
        v_current_se := 1.0;
        v_n_responses := 0;
    END IF;
    
    -- Simple EAP update (simplified Bayesian)
    v_new_theta := v_current_theta + 
        (CASE WHEN p_is_correct THEN 1 ELSE -1 END) * 
        (0.5 / (v_n_responses + 1));
    
    v_new_se := v_current_se * SQRT(v_n_responses / (v_n_responses + 1.0));
    
    -- Upsert ability
    INSERT INTO user_abilities (user_id, topic_id, theta, theta_se, n_responses)
    VALUES (p_user_id, p_topic_id, v_new_theta, v_new_se, v_n_responses + 1)
    ON CONFLICT (user_id, topic_id)
    DO UPDATE SET
        theta = v_new_theta,
        theta_se = v_new_se,
        n_responses = user_abilities.n_responses + 1,
        updated_at = NOW();
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- sql/09_seed_data.sql
-- Initial seed data for development
-- =====================================================

-- Insert default tenant
INSERT INTO feature_flags (key, enabled, value_json) VALUES
    ('adaptive_testing', true, '{"model": "3PL"}'),
    ('exposure_control', true, '{"method": "sympson_hetter"}'),
    ('semantic_search', true, '{"model": "all-MiniLM-L6-v2"}'),
    ('bulk_import', true, '{"formats": ["QTI", "GIFT", "CSV"]}'),
    ('advanced_analytics', true, '{"enabled_reports": ["item_analysis", "dif", "test_info"]}');

-- Insert sample topics
INSERT INTO topics (name, blueprint_code, description, weight) VALUES
    ('Mathematics', 'MATH', 'Mathematical concepts and problem solving', 0.25),
    ('Science', 'SCI', 'Scientific principles and methodology', 0.25),
    ('Reading', 'READ', 'Reading comprehension and analysis', 0.25),
    ('Writing', 'WRITE', 'Writing skills and composition', 0.25);

-- Insert subtopics
INSERT INTO topics (parent_id, name, blueprint_code, description, weight) VALUES
    (1, 'Algebra', 'MATH.ALG', 'Algebraic expressions and equations', 0.33),
    (1, 'Geometry', 'MATH.GEO', 'Geometric shapes and proofs', 0.33),
    (1, 'Statistics', 'MATH.STAT', 'Statistical analysis and probability', 0.34),
    (2, 'Biology', 'SCI.BIO', 'Life sciences and organisms', 0.33),
    (2, 'Chemistry', 'SCI.CHEM', 'Chemical reactions and properties', 0.33),
    (2, 'Physics', 'SCI.PHYS', 'Physical laws and mechanics', 0.34);

-- Update ltree paths
UPDATE topics SET path = id::text::ltree WHERE parent_id IS NULL;
UPDATE topics SET path = p.path || id::text::ltree 
FROM topics p WHERE topics.parent_id = p.id;

-- Create materialized view for analytics
CREATE MATERIALIZED VIEW analytics.item_performance AS
SELECT 
    q.id as question_id,
    qv.version,
    qv.topic_id,
    t.name as topic_name,
    qv.difficulty_label,
    COUNT(ur.id) as total_responses,
    AVG(CASE WHEN ur.is_correct THEN 1.0 ELSE 0.0 END) as p_value,
    STDDEV(CASE WHEN ur.is_correct THEN 1.0 ELSE 0.0 END) as discrimination,
    AVG(ur.time_taken_ms) as avg_time_ms,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ur.time_taken_ms) as median_time_ms
FROM questions q
JOIN question_versions qv ON q.id = qv.question_id
LEFT JOIN topics t ON qv.topic_id = t.id
LEFT JOIN user_responses ur ON ur.question_id = q.id AND ur.version = qv.version
WHERE qv.state = 'published'
GROUP BY q.id, qv.version, qv.topic_id, t.name, qv.difficulty_label;

CREATE UNIQUE INDEX idx_item_performance_pk ON analytics.item_performance(question_id, version);
CREATE INDEX idx_item_performance_topic ON analytics.item_performance(topic_id);
CREATE INDEX idx_item_performance_p_value ON analytics.item_performance(p_value);

-- Create update trigger for search vectors
CREATE OR REPLACE FUNCTION refresh_analytics_views() RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY analytics.item_performance;
END;
$$ LANGUAGE plpgsql;

-- Schedule periodic refresh (requires pg_cron extension)
-- SELECT cron.schedule('refresh-analytics', '*/10 * * * *', 'SELECT refresh_analytics_views();');