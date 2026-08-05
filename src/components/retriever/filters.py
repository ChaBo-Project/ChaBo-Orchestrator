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

from typing import Dict

from components.utils import load_instance_yaml

FILTER_VALUES: dict[str, list] = load_instance_yaml().get("filters", {})


def validate_filterable_fields(filterable_fields: Dict[str, str]) -> None:
    """
    Raise ValueError if any field declared in params.cfg [metadata_filters]
    filterable_fields has no entry in FILTER_VALUES.
    """
    missing = [f for f in filterable_fields if f not in FILTER_VALUES]
    if missing:
        raise ValueError(
            f"Fields declared in params.cfg [metadata_filters] are missing valid values: {missing}. "
            "Add them under the `filters` key in INSTANCE_CONFIG_DIR/instance.yaml "
            "or remove them from filterable_fields in params.cfg."
        )