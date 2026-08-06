# filters.py — valid values for filterable metadata fields.
#
# params.cfg's [metadata_filters] filterable_fields only declares *which* fields are
# filterable, by name. The valid *values* for each field are instance content and live
# here — resolved below from INSTANCE_CONFIG_DIR/instance.yaml's `filters:` key (see
# components.utils.load_instance_yaml); falls back to {} (no filterable fields
# resolvable) when unset.
#
# Rules:
# - Keys must match the field names declared in filterable_fields exactly.
# - Every declared field MUST have an entry here — missing entries raise a ValueError at
#   startup.
# - Values must match what is stored in Qdrant payload metadata for that field.
# - Extra keys here that are not declared as filterable fields are silently ignored.

from typing import Dict

from components.utils import load_instance_yaml

FILTER_VALUES: dict[str, list] = load_instance_yaml().get("filters", {})


def validate_filterable_fields(filterable_fields: Dict[str, str]) -> None:
    """
    Raise ValueError if any field declared as filterable has no entry in FILTER_VALUES.
    """
    missing = [f for f in filterable_fields if f not in FILTER_VALUES]
    if missing:
        raise ValueError(
            f"Fields declared as filterable have no valid values: {missing}. "
            "Add them under the `filters` key in INSTANCE_CONFIG_DIR/instance.yaml, "
            "or remove them from params.cfg's [metadata_filters] filterable_fields."
        )