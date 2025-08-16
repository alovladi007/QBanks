"""
Base database models and mixins.
"""
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, DateTime, String, Boolean, JSON, UUID, Integer
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.sql import func
from sqlalchemy.orm import Session
import uuid

Base = declarative_base()

class TimestampMixin:
    """Mixin for automatic timestamp management."""
    
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        index=True
    )

class SoftDeleteMixin:
    """Mixin for soft delete functionality."""
    
    is_deleted = Column(Boolean, default=False, nullable=False, index=True)
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    deleted_by = Column(String(255), nullable=True)
    
    def soft_delete(self, user_id: str) -> None:
        """Soft delete the record."""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()
        self.deleted_by = user_id
    
    def restore(self) -> None:
        """Restore a soft deleted record."""
        self.is_deleted = False
        self.deleted_at = None
        self.deleted_by = None

class AuditMixin:
    """Mixin for audit trail."""
    
    created_by = Column(String(255), nullable=False, index=True)
    updated_by = Column(String(255), nullable=True)
    version = Column(Integer, default=1, nullable=False)
    
    def increment_version(self) -> None:
        """Increment the version number."""
        self.version += 1

class TenantMixin:
    """Mixin for multi-tenancy support."""
    
    @declared_attr
    def tenant_id(cls):
        return Column(
            UUID(as_uuid=True),
            nullable=False,
            default=uuid.UUID('00000000-0000-0000-0000-000000000001'),
            index=True
        )

class MetadataMixin:
    """Mixin for flexible metadata storage."""
    
    metadata = Column(JSON, default={}, nullable=False)
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key."""
        return self.metadata.get(key, default) if self.metadata else default
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value by key."""
        if not self.metadata:
            self.metadata = {}
        self.metadata[key] = value
    
    def update_metadata(self, data: Dict[str, Any]) -> None:
        """Update metadata with dictionary."""
        if not self.metadata:
            self.metadata = {}
        self.metadata.update(data)

class BaseModel(Base, TimestampMixin, AuditMixin, MetadataMixin):
    """Base model with common fields."""
    
    __abstract__ = True
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            if isinstance(value, datetime):
                value = value.isoformat()
            elif isinstance(value, uuid.UUID):
                value = str(value)
            result[column.name] = value
        return result
    
    def update_from_dict(self, data: Dict[str, Any], exclude: Optional[set] = None) -> None:
        """Update model from dictionary."""
        exclude = exclude or set()
        for key, value in data.items():
            if key not in exclude and hasattr(self, key):
                setattr(self, key, value)
    
    @classmethod
    def create(cls, db: Session, **kwargs) -> 'BaseModel':
        """Create a new record."""
        instance = cls(**kwargs)
        db.add(instance)
        db.commit()
        db.refresh(instance)
        return instance
    
    def save(self, db: Session) -> None:
        """Save changes to database."""
        db.add(self)
        db.commit()
        db.refresh(self)
    
    def delete(self, db: Session) -> None:
        """Delete record from database."""
        db.delete(self)
        db.commit()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<{self.__class__.__name__}(id={self.id})>"