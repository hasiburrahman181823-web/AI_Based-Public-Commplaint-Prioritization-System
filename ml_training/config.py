"""Edit COLUMN_MAPPING to match the headers in your source CSV.

Keys are the standardized output names; values are the source CSV headers.
Only text, category and severity must exist in the source CSV. Optional
columns are copied when present and filled according to docs/data_schema.md.
"""

SCHEMA_COLUMNS = (
    "id", "original_id", "text", "lang", "category", "severity",
    "source_type", "location", "timestamp",
)
REQUIRED_COLUMNS = ("text", "category", "severity")

COLUMN_MAPPING = {
    "id": "id",
    "original_id": "original_id",
    "text": "text",
    "lang": "lang",
    "category": "category",
    "severity": "severity",
    "source_type": "source_type",
    "location": "location",
    "timestamp": "timestamp",
}

SPLIT_RATIOS = (0.70, 0.15, 0.15)
