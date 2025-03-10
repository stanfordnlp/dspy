"""empty message

Revision ID: v2.4.0.a
Revises: v1.3.0.a
Create Date: 2020-11-17 02:16:16.536171

"""

from alembic import op
import sqlalchemy as sa
from typing import Any

from sqlalchemy import Column
from sqlalchemy import Enum
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import UniqueConstraint
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import orm

from optuna.study import StudyDirection

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    # TODO(c-bata): Remove this after dropping support for SQLAlchemy v1.3 or prior.
    from sqlalchemy.ext.declarative import declarative_base


# revision identifiers, used by Alembic.
revision = "v2.4.0.a"
down_revision = "v1.3.0.a"
branch_labels = None
depends_on = None

# Model definition
BaseModel = declarative_base()


class StudyModel(BaseModel):
    __tablename__ = "studies"
    study_id = Column(Integer, primary_key=True)
    direction = sa.Column(sa.Enum(StudyDirection))


class StudyDirectionModel(BaseModel):
    __tablename__ = "study_directions"
    __table_args__: Any = (UniqueConstraint("study_id", "objective"),)
    study_direction_id = Column(Integer, primary_key=True)
    direction = Column(Enum(StudyDirection), nullable=False)
    study_id = Column(Integer, ForeignKey("studies.study_id"), nullable=False)
    objective = Column(Integer, nullable=False)


class TrialModel(BaseModel):
    __tablename__ = "trials"
    trial_id = Column(Integer, primary_key=True)
    number = Column(Integer)
    study_id = Column(Integer, ForeignKey("studies.study_id"))
    value = sa.Column(sa.Float)


class TrialValueModel(BaseModel):
    __tablename__ = "trial_values"
    __table_args__: Any = (UniqueConstraint("trial_id", "objective"),)
    trial_value_id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, ForeignKey("trials.trial_id"), nullable=False)
    objective = Column(Integer, nullable=False)
    value = Column(Float, nullable=False)
    step = sa.Column(sa.Integer)


class TrialIntermediateValueModel(BaseModel):
    __tablename__ = "trial_intermediate_values"
    __table_args__: Any = (UniqueConstraint("trial_id", "step"),)
    trial_intermediate_value_id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, ForeignKey("trials.trial_id"), nullable=False)
    step = Column(Integer, nullable=False)
    intermediate_value = Column(Float, nullable=False)


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()

    if "study_directions" not in tables:
        op.create_table(
            "study_directions",
            sa.Column("study_direction_id", sa.Integer(), nullable=False),
            sa.Column(
                "direction",
                sa.Enum("NOT_SET", "MINIMIZE", "MAXIMIZE", name="studydirection"),
                nullable=False,
            ),
            sa.Column("study_id", sa.Integer(), nullable=False),
            sa.Column("objective", sa.Integer(), nullable=False),
            sa.ForeignKeyConstraint(
                ["study_id"],
                ["studies.study_id"],
            ),
            sa.PrimaryKeyConstraint("study_direction_id"),
            sa.UniqueConstraint("study_id", "objective"),
        )

    if "trial_intermediate_values" not in tables:
        op.create_table(
            "trial_intermediate_values",
            sa.Column("trial_intermediate_value_id", sa.Integer(), nullable=False),
            sa.Column("trial_id", sa.Integer(), nullable=False),
            sa.Column("step", sa.Integer(), nullable=False),
            sa.Column("intermediate_value", sa.Float(), nullable=False),
            sa.ForeignKeyConstraint(
                ["trial_id"],
                ["trials.trial_id"],
            ),
            sa.PrimaryKeyConstraint("trial_intermediate_value_id"),
            sa.UniqueConstraint("trial_id", "step"),
        )

    session = orm.Session(bind=bind)
    try:
        studies_records = session.query(StudyModel).all()
        objects = [
            StudyDirectionModel(study_id=r.study_id, direction=r.direction, objective=0)
            for r in studies_records
        ]
        session.bulk_save_objects(objects)

        intermediate_values_records = session.query(
            TrialValueModel.trial_id, TrialValueModel.value, TrialValueModel.step
        ).all()
        objects = [
            TrialIntermediateValueModel(
                trial_id=r.trial_id, intermediate_value=r.value, step=r.step
            )
            for r in intermediate_values_records
        ]
        session.bulk_save_objects(objects)

        session.query(TrialValueModel).delete()
        session.commit()

        with op.batch_alter_table("trial_values", schema=None) as batch_op:
            batch_op.add_column(sa.Column("objective", sa.Integer(), nullable=False))
            # The name of this constraint is manually determined.
            # In the future, the naming convention may be determined based on
            # https://alembic.sqlalchemy.org/en/latest/naming.html
            batch_op.create_unique_constraint(
                "uq_trial_values_trial_id_objective", ["trial_id", "objective"]
            )

        trials_records = session.query(TrialModel).all()
        objects = [
            TrialValueModel(trial_id=r.trial_id, value=r.value, objective=0)
            for r in trials_records
        ]
        session.bulk_save_objects(objects)

        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        raise e
    finally:
        session.close()

    with op.batch_alter_table("studies", schema=None) as batch_op:
        batch_op.drop_column("direction")

    with op.batch_alter_table("trial_values", schema=None) as batch_op:
        batch_op.drop_column("step")

    with op.batch_alter_table("trials", schema=None) as batch_op:
        batch_op.drop_column("value")

    for c in inspector.get_unique_constraints("trial_values"):
        # MySQL changes the uniq constraint of (trial_id, step) to that of trial_id.
        if c["column_names"] == ["trial_id"]:
            with op.batch_alter_table("trial_values", schema=None) as batch_op:
                batch_op.drop_constraint(c["name"], type_="unique")
            break


# TODO(imamura): Implement downgrade
def downgrade():
    pass
