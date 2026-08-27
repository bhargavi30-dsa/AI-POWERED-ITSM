"""
Database setup for Enterprise AI ITSM.

Defines the SQLAlchemy models and connection for users,
tickets, and extracted features (PostgreSQL).
"""

from sqlalchemy import (
    create_engine, Column, Integer, String, Boolean,
    DateTime, ForeignKey, Text
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import datetime, timezone

# =========================================================
# DATABASE CONNECTION
# =========================================================

DATABASE_URL = "postgresql://postgres:DataBase King@localhost:5432/itsm_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()


# =========================================================
# USER MODEL
# =========================================================

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    role = Column(String, nullable=False, default="employee")  # "employee" or "manager"
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    tickets = relationship("Ticket", back_populates="user", cascade="all, delete-orphan")


# =========================================================
# TICKET MODEL
# =========================================================

class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    incident_description = Column(Text, nullable=False)
    summary = Column(Text)

    predicted_category = Column(String)
    predicted_priority = Column(String)
    predicted_sla = Column(Boolean)

    predicted_assignment_group = Column(String)
    recommended_resolver = Column(String)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="tickets")
    features = relationship(
        "ExtractedFeature",
        back_populates="ticket",
        uselist=False,
        cascade="all, delete-orphan"
    )


# =========================================================
# EXTRACTED FEATURES MODEL
# =========================================================

class ExtractedFeature(Base):
    __tablename__ = "extracted_features"

    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(Integer, ForeignKey("tickets.id"), nullable=False)

    contact_type = Column(String)
    location = Column(String)
    u_symptom = Column(String)
    impact = Column(String)
    urgency = Column(String)
    knowledge = Column(Boolean)
    notify = Column(String)
    opened_hour = Column(Integer)
    opened_day_of_week = Column(Integer)
    opened_month = Column(Integer)
    is_weekend = Column(Boolean)

    ticket = relationship("Ticket", back_populates="features")


# =========================================================
# CREATE TABLES
# =========================================================

def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
    print("Database tables created successfully.")