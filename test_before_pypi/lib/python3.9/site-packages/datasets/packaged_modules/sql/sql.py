import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import pandas as pd
import pyarrow as pa

import datasets
import datasets.config
from datasets.features.features import require_storage_cast
from datasets.table import table_cast


if TYPE_CHECKING:
    import sqlite3

    import sqlalchemy


logger = datasets.utils.logging.get_logger(__name__)


@dataclass
class SqlConfig(datasets.BuilderConfig):
    """BuilderConfig for SQL."""

    sql: Union[str, "sqlalchemy.sql.Selectable"] = None
    con: Union[str, "sqlalchemy.engine.Connection", "sqlalchemy.engine.Engine", "sqlite3.Connection"] = None
    index_col: Optional[Union[str, List[str]]] = None
    coerce_float: bool = True
    params: Optional[Union[List, Tuple, Dict]] = None
    parse_dates: Optional[Union[List, Dict]] = None
    columns: Optional[List[str]] = None
    chunksize: Optional[int] = 10_000
    features: Optional[datasets.Features] = None

    def __post_init__(self):
        super().__post_init__()
        if self.sql is None:
            raise ValueError("sql must be specified")
        if self.con is None:
            raise ValueError("con must be specified")

    def create_config_id(
        self,
        config_kwargs: dict,
        custom_features: Optional[datasets.Features] = None,
    ) -> str:
        config_kwargs = config_kwargs.copy()
        # We need to stringify the Selectable object to make its hash deterministic

        # The process of stringifying is explained here: http://docs.sqlalchemy.org/en/latest/faq/sqlexpressions.html
        sql = config_kwargs["sql"]
        if not isinstance(sql, str):
            if datasets.config.SQLALCHEMY_AVAILABLE and "sqlalchemy" in sys.modules:
                import sqlalchemy

                if isinstance(sql, sqlalchemy.sql.Selectable):
                    engine = sqlalchemy.create_engine(config_kwargs["con"].split("://")[0] + "://")
                    sql_str = str(sql.compile(dialect=engine.dialect))
                    config_kwargs["sql"] = sql_str
                else:
                    raise TypeError(
                        f"Supported types for 'sql' are string and sqlalchemy.sql.Selectable but got {type(sql)}: {sql}"
                    )
            else:
                raise TypeError(
                    f"Supported types for 'sql' are string and sqlalchemy.sql.Selectable but got {type(sql)}: {sql}"
                )
        con = config_kwargs["con"]
        if not isinstance(con, str):
            config_kwargs["con"] = id(con)
            logger.info(
                f"SQL connection 'con' of type {type(con)} couldn't be hashed properly. To enable hashing, specify 'con' as URI string instead."
            )

        return super().create_config_id(config_kwargs, custom_features=custom_features)

    @property
    def pd_read_sql_kwargs(self):
        pd_read_sql_kwargs = {
            "index_col": self.index_col,
            "columns": self.columns,
            "params": self.params,
            "coerce_float": self.coerce_float,
            "parse_dates": self.parse_dates,
        }
        return pd_read_sql_kwargs


class Sql(datasets.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = SqlConfig

    def _info(self):
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={})]

    def _cast_table(self, pa_table: pa.Table) -> pa.Table:
        if self.config.features is not None:
            schema = self.config.features.arrow_schema
            if all(not require_storage_cast(feature) for feature in self.config.features.values()):
                # cheaper cast
                pa_table = pa.Table.from_arrays([pa_table[field.name] for field in schema], schema=schema)
            else:
                # more expensive cast; allows str <-> int/float or str to Audio for example
                pa_table = table_cast(pa_table, schema)
        return pa_table

    def _generate_tables(self):
        chunksize = self.config.chunksize
        sql_reader = pd.read_sql(
            self.config.sql, self.config.con, chunksize=chunksize, **self.config.pd_read_sql_kwargs
        )
        sql_reader = [sql_reader] if chunksize is None else sql_reader
        for chunk_idx, df in enumerate(sql_reader):
            pa_table = pa.Table.from_pandas(df)
            yield chunk_idx, self._cast_table(pa_table)
