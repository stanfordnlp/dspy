"""Change floating point precision and make intermediate_value nullable.

Revision ID: v3.0.0.b
Revises: v3.0.0.a
Create Date: 2022-04-27 16:31:42.012666

"""

import enum

from alembic import op
from sqlalchemy import and_
from sqlalchemy import Column
from sqlalchemy import Enum
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy.orm import Session

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    # TODO(c-bata): Remove this after dropping support for SQLAlchemy v1.3 or prior.
    from sqlalchemy.ext.declarative import declarative_base


# revision identifiers, used by Alembic.
revision = "v3.0.0.b"
down_revision = "v3.0.0.a"
branch_labels = None
depends_on = None

BaseModel = declarative_base()
FLOAT_PRECISION = 53


class TrialState(enum.Enum):
    RUNNING = 0
    COMPLETE = 1
    PRUNED = 2
    FAIL = 3
    WAITING = 4


class TrialModel(BaseModel):
    __tablename__ = "trials"
    trial_id = Column(Integer, primary_key=True)
    number = Column(Integer)
    state = Column(Enum(TrialState), nullable=False)


class TrialValueModel(BaseModel):
    __tablename__ = "trial_values"
    trial_value_id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, ForeignKey("trials.trial_id"), nullable=False)
    value = Column(Float, nullable=False)


def upgrade():
    bind = op.get_bind()
    session = Session(bind=bind)

    if (
        session.query(TrialValueModel)
        .join(TrialModel, TrialValueModel.trial_id == TrialModel.trial_id)
        .filter(and_(TrialModel.state == TrialState.COMPLETE, TrialValueModel.value.is_(None)))
        .count()
    ) != 0:
        raise ValueError("Found invalid trial_values records (value=None and state='COMPLETE')")
    session.query(TrialValueModel).filter(TrialValueModel.value.is_(None)).delete()

    with op.batch_alter_table("trial_intermediate_values") as batch_op:
        batch_op.alter_column(
            "intermediate_value",
            type_=Float(precision=FLOAT_PRECISION),
            nullable=True,
        )
    with op.batch_alter_table("trial_params") as batch_op:
        batch_op.alter_column(
            "param_value",
            type_=Float(precision=FLOAT_PRECISION),
            existing_nullable=True,
        )
    with op.batch_alter_table("trial_values") as batch_op:
        batch_op.alter_column(
            "value",
            type_=Float(precision=FLOAT_PRECISION),
            nullable=False,
        )


def downgrade():
    with op.batch_alter_table("trial_intermediate_values") as batch_op:
        batch_op.alter_column("intermediate_value", type_=Float, nullable=False)
    with op.batch_alter_table("trial_params") as batch_op:
        batch_op.alter_column("param_value", type_=Float, existing_nullable=True)
    with op.batch_alter_table("trial_values") as batch_op:
        batch_op.alter_column("value", type_=Float, existing_nullable=False)
