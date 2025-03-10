"""empty message

Revision ID: v2.6.0.a
Revises: v2.4.0.a
Create Date: 2021-03-01 11:30:32.214196

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "v2.6.0.a"
down_revision = "v2.4.0.a"
branch_labels = None
depends_on = None

MAX_STRING_LENGTH = 2048


def upgrade():
    with op.batch_alter_table("study_user_attributes") as batch_op:
        batch_op.alter_column("value_json", type_=sa.TEXT)
    with op.batch_alter_table("study_system_attributes") as batch_op:
        batch_op.alter_column("value_json", type_=sa.TEXT)
    with op.batch_alter_table("trial_user_attributes") as batch_op:
        batch_op.alter_column("value_json", type_=sa.TEXT)
    with op.batch_alter_table("trial_system_attributes") as batch_op:
        batch_op.alter_column("value_json", type_=sa.TEXT)
    with op.batch_alter_table("trial_params") as batch_op:
        batch_op.alter_column("distribution_json", type_=sa.TEXT)


def downgrade():
    with op.batch_alter_table("study_user_attributes") as batch_op:
        batch_op.alter_column("value_json", type_=sa.String(MAX_STRING_LENGTH))
    with op.batch_alter_table("study_system_attributes") as batch_op:
        batch_op.alter_column("value_json", type_=sa.String(MAX_STRING_LENGTH))
    with op.batch_alter_table("trial_user_attributes") as batch_op:
        batch_op.alter_column("value_json", type_=sa.String(MAX_STRING_LENGTH))
    with op.batch_alter_table("trial_system_attributes") as batch_op:
        batch_op.alter_column("value_json", type_=sa.String(MAX_STRING_LENGTH))
    with op.batch_alter_table("trial_params") as batch_op:
        batch_op.alter_column("distribution_json", type_=sa.String(MAX_STRING_LENGTH))
