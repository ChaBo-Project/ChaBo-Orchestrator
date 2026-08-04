# filters.py — valid values for each filterable field declared in params.cfg.
#
# Rules:
# - Keys must match field names in params.cfg [metadata_filters] filterable_fields exactly.
# - Every field in filterable_fields MUST have an entry here — missing entries raise a
#   ValueError at startup.
# - Values must match what is stored in Qdrant payload metadata for that field.
# - Extra keys here that are not in filterable_fields are silently ignored.
#
# Real values are instance-specific content — never committed to this repo. Supplied via
# INSTANCE_CONFIG_DIR/instance.yaml's `filters:` key (see components.utils.load_instance_yaml);
# falls back to {} (no filterable fields resolvable) when unset.

from components.utils import load_instance_yaml

FILTER_VALUES: dict[str, list] = load_instance_yaml().get("filters", {})