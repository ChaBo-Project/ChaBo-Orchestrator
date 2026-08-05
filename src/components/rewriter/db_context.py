"""
DBContext loader for the query rewriter (+ any other future dependent modules)

Loads a per-deployment abstract describing the doc store and a glossary, from
INSTANCE_CONFIG_DIR/instance.yaml's `db_context` key.
"""
import logging
from typing import List, Dict, Any

from pydantic import BaseModel, Field

from components.utils import load_instance_yaml

logger = logging.getLogger(__name__)


class DBContext(BaseModel):
    """Per-deployment database-awareness object for the query rewriter."""
    abstract: str = ""
    glossary: List[Dict[str, Any]] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True if no meaningful context is configured."""
        return not self.abstract.strip() and not self.glossary


def load_db_context_from_instance() -> DBContext:
    """
    Load DBContext from INSTANCE_CONFIG_DIR/instance.yaml's `db_context` key.

    Returns an empty DBContext if INSTANCE_CONFIG_DIR is unset, instance.yaml is absent,
    or the `db_context` key is absent/empty.

    Filters out explicit `null` values before construction: a hand-edited YAML that
    leaves `abstract:`/`glossary:` blank parses to None in PyYAML, and DBContext's
    typed fields (str/list) reject None outright — this keeps that case falling back
    to the field's own default instead of raising at startup.
    """
    data = load_instance_yaml().get("db_context", {})
    data = {k: v for k, v in data.items() if v is not None}
    return DBContext(**data)
