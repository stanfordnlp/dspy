"""empty message

Revision ID: v1.2.0.a
Revises: v0.9.0.a
Create Date: 2020-02-05 15:17:41.458947

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "v1.2.0.a"
down_revision = "v0.9.0.a"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("trials") as batch_op:
        batch_op.alter_column(
            "state",
            type_=sa.Enum("RUNNING", "COMPLETE", "PRUNED", "FAIL", "WAITING", name="trialstate"),
            existing_type=sa.Enum("RUNNING", "COMPLETE", "PRUNED", "FAIL", name="trialstate"),
        )


def downgrade():
    with op.batch_alter_table("trials") as batch_op:
        batch_op.alter_column(
            "state",
            type_=sa.Enum("RUNNING", "COMPLETE", "PRUNED", "FAIL", name="trialstate"),
            existing_type=sa.Enum(
                "RUNNING", "COMPLETE", "PRUNED", "FAIL", "WAITING", name="trialstate"
            ),
        )
