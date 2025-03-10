"""Add intermediate_value_type column to represent +inf and -inf

Revision ID: v3.0.0.c
Revises: v3.0.0.b
Create Date: 2022-05-16 17:17:28.810792

"""

from __future__ import annotations

import enum

import numpy as np
from alembic import op
import sqlalchemy as sa
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import orm

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    # TODO(c-bata): Remove this after dropping support for SQLAlchemy v1.3 or prior.
    from sqlalchemy.ext.declarative import declarative_base


# revision identifiers, used by Alembic.
revision = "v3.0.0.c"
down_revision = "v3.0.0.b"
branch_labels = None
depends_on = None


BaseModel = declarative_base()
RDB_MAX_FLOAT = np.finfo(np.float32).max
RDB_MIN_FLOAT = np.finfo(np.float32).min


FLOAT_PRECISION = 53


class IntermediateValueModel(BaseModel):
    class TrialIntermediateValueType(enum.Enum):
        FINITE = 1
        INF_POS = 2
        INF_NEG = 3
        NAN = 4

    __tablename__ = "trial_intermediate_values"
    trial_intermediate_value_id = sa.Column(sa.Integer, primary_key=True)
    intermediate_value = sa.Column(sa.Float(precision=FLOAT_PRECISION), nullable=True)
    intermediate_value_type = sa.Column(sa.Enum(TrialIntermediateValueType), nullable=False)

    @classmethod
    def intermediate_value_to_stored_repr(
        cls,
        value: float,
    ) -> tuple[float | None, TrialIntermediateValueType]:
        if np.isnan(value):
            return (None, cls.TrialIntermediateValueType.NAN)
        elif value == float("inf"):
            return (None, cls.TrialIntermediateValueType.INF_POS)
        elif value == float("-inf"):
            return (None, cls.TrialIntermediateValueType.INF_NEG)
        else:
            return (value, cls.TrialIntermediateValueType.FINITE)


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    column_names = [c["name"] for c in inspector.get_columns("trial_intermediate_values")]

    sa.Enum(IntermediateValueModel.TrialIntermediateValueType).create(bind, checkfirst=True)

    # MySQL and PostgreSQL supports DEFAULT clause like 'ALTER TABLE <tbl_name>
    # ADD COLUMN <col_name> ... DEFAULT "FINITE_OR_NAN"', but seemingly Alembic
    # does not support such a SQL statement. So first add a column with schema-level
    # default value setting, then remove it by `batch_op.alter_column()`.
    if "intermediate_value_type" not in column_names:
        with op.batch_alter_table("trial_intermediate_values") as batch_op:
            batch_op.add_column(
                sa.Column(
                    "intermediate_value_type",
                    sa.Enum(
                        "FINITE", "INF_POS", "INF_NEG", "NAN", name="trialintermediatevaluetype"
                    ),
                    nullable=False,
                    server_default="FINITE",
                ),
            )
    with op.batch_alter_table("trial_intermediate_values") as batch_op:
        batch_op.alter_column(
            "intermediate_value_type",
            existing_type=sa.Enum(
                "FINITE", "INF_POS", "INF_NEG", "NAN", name="trialintermediatevaluetype"
            ),
            existing_nullable=False,
            server_default=None,
        )

    session = orm.Session(bind=bind)
    try:
        records = (
            session.query(IntermediateValueModel)
            .filter(
                sa.or_(
                    IntermediateValueModel.intermediate_value > 1e16,
                    IntermediateValueModel.intermediate_value < -1e16,
                    IntermediateValueModel.intermediate_value.is_(None),
                )
            )
            .all()
        )
        mapping = []
        for r in records:
            value: float
            if r.intermediate_value is None or np.isnan(r.intermediate_value):
                value = float("nan")
            elif np.isclose(r.intermediate_value, RDB_MAX_FLOAT) or np.isposinf(
                r.intermediate_value
            ):
                value = float("inf")
            elif np.isclose(r.intermediate_value, RDB_MIN_FLOAT) or np.isneginf(
                r.intermediate_value
            ):
                value = float("-inf")
            else:
                value = r.intermediate_value
            (
                stored_value,
                float_type,
            ) = IntermediateValueModel.intermediate_value_to_stored_repr(value)
            mapping.append(
                {
                    "trial_intermediate_value_id": r.trial_intermediate_value_id,
                    "intermediate_value_type": float_type,
                    "intermediate_value": stored_value,
                }
            )
        session.bulk_update_mappings(IntermediateValueModel, mapping)
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        raise e
    finally:
        session.close()


def downgrade():
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    try:
        records = session.query(IntermediateValueModel).all()
        mapping = []
        for r in records:
            if (
                r.intermediate_value_type
                == IntermediateValueModel.TrialIntermediateValueType.FINITE
                or r.intermediate_value_type
                == IntermediateValueModel.TrialIntermediateValueType.NAN
            ):
                continue

            _intermediate_value = r.intermediate_value
            if (
                r.intermediate_value_type
                == IntermediateValueModel.TrialIntermediateValueType.INF_POS
            ):
                _intermediate_value = RDB_MAX_FLOAT
            else:
                _intermediate_value = RDB_MIN_FLOAT

            mapping.append(
                {
                    "trial_intermediate_value_id": r.trial_intermediate_value_id,
                    "intermediate_value": _intermediate_value,
                }
            )
        session.bulk_update_mappings(IntermediateValueModel, mapping)
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        raise e
    finally:
        session.close()

    with op.batch_alter_table("trial_intermediate_values", schema=None) as batch_op:
        batch_op.drop_column("intermediate_value_type")

    sa.Enum(IntermediateValueModel.FloatTypeEnum).drop(bind, checkfirst=True)
