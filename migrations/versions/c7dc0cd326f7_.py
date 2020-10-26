"""empty message

Revision ID: c7dc0cd326f7
Revises: 
Create Date: 2020-10-23 10:13:29.412526

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c7dc0cd326f7'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('exames',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(), nullable=True),
    sa.Column('cpf', sa.String(), nullable=True),
    sa.Column('tipoExame', sa.String(), nullable=True),
    sa.Column('response', sa.String(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('exames')
    # ### end Alembic commands ###
