"""
User, Organization, and Authentication models.
"""
from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column, String, Boolean, DateTime, ForeignKey, 
    Integer, Float, Text, Enum, Table, UniqueConstraint
)
from sqlalchemy.orm import relationship, backref
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
import enum
import uuid

from .base import BaseModel, Base, TenantMixin, SoftDeleteMixin

# Association tables
user_organizations = Table(
    'user_organizations',
    Base.metadata,
    Column('user_id', UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE')),
    Column('organization_id', UUID(as_uuid=True), ForeignKey('organizations.id', ondelete='CASCADE')),
    Column('role', String(50), default='member'),
    Column('joined_at', DateTime(timezone=True), server_default='now()'),
    UniqueConstraint('user_id', 'organization_id', name='uq_user_org')
)

user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE')),
    Column('role_id', UUID(as_uuid=True), ForeignKey('roles.id', ondelete='CASCADE')),
    Column('assigned_at', DateTime(timezone=True), server_default='now()'),
    Column('assigned_by', String(255)),
    UniqueConstraint('user_id', 'role_id', name='uq_user_role')
)

role_permissions = Table(
    'role_permissions',
    Base.metadata,
    Column('role_id', UUID(as_uuid=True), ForeignKey('roles.id', ondelete='CASCADE')),
    Column('permission_id', UUID(as_uuid=True), ForeignKey('permissions.id', ondelete='CASCADE')),
    UniqueConstraint('role_id', 'permission_id', name='uq_role_permission')
)

class UserStatus(str, enum.Enum):
    """User account status."""
    PENDING = "pending"
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    BANNED = "banned"

class SubscriptionTier(str, enum.Enum):
    """Subscription tiers."""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"

class SubscriptionStatus(str, enum.Enum):
    """Subscription status."""
    TRIAL = "trial"
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

class Organization(BaseModel, TenantMixin, SoftDeleteMixin):
    """Organization model for multi-tenancy."""
    
    __tablename__ = 'organizations'
    
    # Basic Information
    name = Column(String(255), nullable=False)
    slug = Column(String(255), unique=True, nullable=False, index=True)
    domain = Column(String(255), unique=True, nullable=True)
    logo_url = Column(String(500), nullable=True)
    description = Column(Text, nullable=True)
    
    # Contact Information
    email = Column(String(255), nullable=False)
    phone = Column(String(50), nullable=True)
    address = Column(JSONB, nullable=True)
    
    # Settings
    settings = Column(JSONB, default={}, nullable=False)
    features = Column(ARRAY(String), default=[], nullable=False)
    
    # Subscription
    subscription_tier = Column(
        Enum(SubscriptionTier),
        default=SubscriptionTier.FREE,
        nullable=False
    )
    subscription_status = Column(
        Enum(SubscriptionStatus),
        default=SubscriptionStatus.TRIAL,
        nullable=False
    )
    subscription_expires_at = Column(DateTime(timezone=True), nullable=True)
    stripe_customer_id = Column(String(255), unique=True, nullable=True)
    stripe_subscription_id = Column(String(255), unique=True, nullable=True)
    
    # Limits
    max_users = Column(Integer, default=5, nullable=False)
    max_questions = Column(Integer, default=100, nullable=False)
    max_tests = Column(Integer, default=10, nullable=False)
    max_students = Column(Integer, default=50, nullable=False)
    storage_limit_gb = Column(Float, default=1.0, nullable=False)
    
    # Usage
    current_users = Column(Integer, default=0, nullable=False)
    current_questions = Column(Integer, default=0, nullable=False)
    current_tests = Column(Integer, default=0, nullable=False)
    current_students = Column(Integer, default=0, nullable=False)
    storage_used_gb = Column(Float, default=0.0, nullable=False)
    
    # Relationships
    users = relationship(
        'User',
        secondary=user_organizations,
        back_populates='organizations'
    )
    
    def is_limit_reached(self, resource: str) -> bool:
        """Check if resource limit is reached."""
        limits = {
            'users': (self.current_users, self.max_users),
            'questions': (self.current_questions, self.max_questions),
            'tests': (self.current_tests, self.max_tests),
            'students': (self.current_students, self.max_students),
            'storage': (self.storage_used_gb, self.storage_limit_gb)
        }
        if resource in limits:
            current, maximum = limits[resource]
            return current >= maximum if maximum > 0 else False
        return False

class User(BaseModel, SoftDeleteMixin):
    """User model with comprehensive features."""
    
    __tablename__ = 'users'
    
    # Authentication
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    
    # Profile
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    display_name = Column(String(200), nullable=True)
    avatar_url = Column(String(500), nullable=True)
    bio = Column(Text, nullable=True)
    phone = Column(String(50), nullable=True)
    timezone = Column(String(50), default='UTC', nullable=False)
    language = Column(String(10), default='en', nullable=False)
    
    # Status
    status = Column(
        Enum(UserStatus),
        default=UserStatus.PENDING,
        nullable=False,
        index=True
    )
    is_verified = Column(Boolean, default=False, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    
    # Security
    two_factor_enabled = Column(Boolean, default=False, nullable=False)
    two_factor_secret = Column(String(255), nullable=True)
    recovery_codes = Column(ARRAY(String), default=[], nullable=False)
    
    # OAuth
    google_id = Column(String(255), unique=True, nullable=True)
    microsoft_id = Column(String(255), unique=True, nullable=True)
    github_id = Column(String(255), unique=True, nullable=True)
    
    # Timestamps
    email_verified_at = Column(DateTime(timezone=True), nullable=True)
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    last_activity_at = Column(DateTime(timezone=True), nullable=True)
    password_changed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Settings
    preferences = Column(JSONB, default={}, nullable=False)
    notification_settings = Column(JSONB, default={}, nullable=False)
    
    # Security tracking
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime(timezone=True), nullable=True)
    login_history = Column(JSONB, default=[], nullable=False)
    
    # Relationships
    organizations = relationship(
        'Organization',
        secondary=user_organizations,
        back_populates='users'
    )
    roles = relationship(
        'Role',
        secondary=user_roles,
        back_populates='users'
    )
    sessions = relationship(
        'UserSession',
        back_populates='user',
        cascade='all, delete-orphan'
    )
    api_keys = relationship(
        'ApiKey',
        back_populates='user',
        cascade='all, delete-orphan'
    )
    
    @property
    def full_name(self) -> str:
        """Get user's full name."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.display_name or self.username
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        if self.is_superuser:
            return True
        for role in self.roles:
            if role.has_permission(permission):
                return True
        return False
    
    def is_locked(self) -> bool:
        """Check if account is locked."""
        if self.locked_until:
            return datetime.utcnow() < self.locked_until
        return False

class Role(BaseModel):
    """Role model for RBAC."""
    
    __tablename__ = 'roles'
    
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    is_system = Column(Boolean, default=False, nullable=False)
    priority = Column(Integer, default=0, nullable=False)
    
    # Relationships
    users = relationship(
        'User',
        secondary=user_roles,
        back_populates='roles'
    )
    permissions = relationship(
        'Permission',
        secondary=role_permissions,
        back_populates='roles'
    )
    
    def has_permission(self, permission: str) -> bool:
        """Check if role has specific permission."""
        return any(p.name == permission for p in self.permissions)

class Permission(BaseModel):
    """Permission model for fine-grained access control."""
    
    __tablename__ = 'permissions'
    
    name = Column(String(100), unique=True, nullable=False)
    resource = Column(String(100), nullable=False)
    action = Column(String(50), nullable=False)
    description = Column(Text, nullable=True)
    
    # Relationships
    roles = relationship(
        'Role',
        secondary=role_permissions,
        back_populates='permissions'
    )

class UserSession(BaseModel):
    """User session tracking."""
    
    __tablename__ = 'user_sessions'
    
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False
    )
    session_token = Column(String(255), unique=True, nullable=False, index=True)
    refresh_token = Column(String(255), unique=True, nullable=True)
    
    # Session info
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    device_info = Column(JSONB, nullable=True)
    location = Column(JSONB, nullable=True)
    
    # Timestamps
    expires_at = Column(DateTime(timezone=True), nullable=False)
    last_activity_at = Column(DateTime(timezone=True), nullable=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship('User', back_populates='sessions')
    
    def is_valid(self) -> bool:
        """Check if session is valid."""
        if self.revoked_at:
            return False
        return datetime.utcnow() < self.expires_at

class ApiKey(BaseModel):
    """API key for programmatic access."""
    
    __tablename__ = 'api_keys'
    
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False
    )
    name = Column(String(100), nullable=False)
    key_hash = Column(String(255), unique=True, nullable=False)
    prefix = Column(String(10), nullable=False)
    
    # Permissions
    scopes = Column(ARRAY(String), default=[], nullable=False)
    rate_limit = Column(Integer, default=1000, nullable=False)
    
    # Usage
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    usage_count = Column(Integer, default=0, nullable=False)
    
    # Validity
    expires_at = Column(DateTime(timezone=True), nullable=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship('User', back_populates='api_keys')
    
    def is_valid(self) -> bool:
        """Check if API key is valid."""
        if self.revoked_at:
            return False
        if self.expires_at:
            return datetime.utcnow() < self.expires_at
        return True