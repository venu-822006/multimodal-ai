"""authentication users sessions login attempts

Revision ID: b3c7e1a9f204
Revises: 1abd75fd72b8
Create Date: 2026-10-12 22:30:00.000000
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'b3c7e1a9f204'
down_revision: Union[str, None] = '1abd75fd72b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('users',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('email', sa.String(length=320), nullable=False),
    sa.Column('email_normalized', sa.String(length=320), nullable=False),
    sa.Column('password_hash', sa.String(length=255), nullable=False),
    sa.Column('full_name', sa.String(length=200), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_users'))
    )
    op.create_index('ix_users_email_normalized', 'users', ['email_normalized'], unique=True)
    op.create_index('ix_users_created_at', 'users', ['created_at'], unique=False)

    op.create_table('auth_sessions',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('token_hash', sa.String(length=64), nullable=False),
    sa.Column('user_agent', sa.String(length=300), nullable=True),
    sa.Column('ip_address', sa.String(length=64), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_auth_sessions_user_id_users'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_auth_sessions'))
    )
    op.create_index('ix_auth_sessions_token_hash', 'auth_sessions', ['token_hash'], unique=True)
    op.create_index('ix_auth_sessions_user_id', 'auth_sessions', ['user_id'], unique=False)
    op.create_index('ix_auth_sessions_expires_at', 'auth_sessions', ['expires_at'], unique=False)

    op.create_table('auth_login_attempts',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('client_hash', sa.String(length=64), nullable=False),
    sa.Column('failed_attempts', sa.Integer(), nullable=False),
    sa.Column('last_failed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('locked_until', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint('failed_attempts >= 0', name=op.f('ck_auth_login_attempts_failed_attempts_non_negative')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_auth_login_attempts'))
    )
    op.create_index('ix_auth_login_attempts_client_hash', 'auth_login_attempts', ['client_hash'], unique=True)
    op.create_index('ix_auth_login_attempts_locked_until', 'auth_login_attempts', ['locked_until'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_auth_login_attempts_locked_until', table_name='auth_login_attempts')
    op.drop_index('ix_auth_login_attempts_client_hash', table_name='auth_login_attempts')
    op.drop_table('auth_login_attempts')
    op.drop_index('ix_auth_sessions_expires_at', table_name='auth_sessions')
    op.drop_index('ix_auth_sessions_user_id', table_name='auth_sessions')
    op.drop_index('ix_auth_sessions_token_hash', table_name='auth_sessions')
    op.drop_table('auth_sessions')
    op.drop_index('ix_users_created_at', table_name='users')
    op.drop_index('ix_users_email_normalized', table_name='users')
    op.drop_table('users')
