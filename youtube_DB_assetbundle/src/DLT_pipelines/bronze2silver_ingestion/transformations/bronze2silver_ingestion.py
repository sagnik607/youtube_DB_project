import dlt
from pyspark.sql.functions import *
from pyspark.sql.types import *

# variables
Catelog_name = spark.conf.get("Catelog_name")
volume_path = f"/Volumes/{Catelog_name}/bronze/earthquake_data"
primary_key = "id"

# schema
properties_schema = StructType(
    [
        StructField("mag", StringType()),
        StructField("place", StringType()),
        StructField("time", StringType()),
        StructField("tsunami", StringType()),
        StructField("type", StringType()),
        StructField("url", StringType()),
        StructField("detail", StringType()),
        StructField("felt", StringType()),
        StructField("mmi", StringType()),
        StructField("alert", StringType()),
        StructField("sig", StringType()),
        StructField("net", StringType()),
        StructField("code", StringType()),
        StructField("ids", StringType()),
        StructField("sources", StringType()),
        StructField("types", StringType()),
        StructField("nst", StringType()),
        StructField("dmin", StringType()),
        StructField("rms", StringType()),
        StructField("gap", StringType()),
        StructField("magType", StringType()),
        StructField("title", StringType()),
    ]
)

geomatric_schema = StructType([StructField("coordinates", ArrayType(DoubleType()))])

feature_schema = StructType(
    [
        StructField("id", StringType()),
        StructField("properties", properties_schema),
        StructField("geometry", geomatric_schema),
    ]
)

schema = ArrayType(feature_schema)


# view
@dlt.view()
def silver_earthquake_view():
    df = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        .load(volume_path)
        .withColumn("_load_timestamp", current_timestamp())
    )
    df = df.withColumn("parsed_data", from_json(col("features"), schema))
    df = df.select(explode(col("parsed_data")).alias("features"), "_load_timestamp")
    df2 = df.select(
        "features.properties.*",
        "features.id",
        col("features.geometry.coordinates")[0].alias("longitude"),
        col("features.geometry.coordinates")[1].alias("latitude"),
        col("features.geometry.coordinates")[2].alias("depth"),
        "_load_timestamp",
    )
    df2 = (
        df2.withColumn("time", from_unixtime(col("time") / 1000).cast("timestamp"))
        .withColumn("mag", col("mag").cast("float"))
        .withColumn("nst", col("nst").cast("int"))
        .withColumn("tsunami", col("tsunami").cast("int"))
    )
    return df2


dlt.create_streaming_table(name="earthquake_silver")

dlt.apply_changes(
    target="earthquake_silver",
    source="silver_earthquake_view",
    keys=[primary_key],
    sequence_by="_load_timestamp",
    stored_as_scd_type="1",
)
