from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, String

iris_source = FileSource(
    path="/home/jupyter/21f3000500_MLOPS_WEEKLY_ASSIGNMENT/iris_feature_repo/feature_repo/data/iris_data.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

iris_entity = Entity(
    name="iris_id",
    join_keys=["iris_id"],
)

iris_feature_view = FeatureView(
    name="iris_features",
    entities=[iris_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
        Field(name="species", dtype=String),
    ],
    source=iris_source,
)
