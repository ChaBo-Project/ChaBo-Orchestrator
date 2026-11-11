"""
Renders an instance's real orchestrator config for Mode B, by shelling out to that instance
repo's own scripts/render-config.py + scripts/validate-config.py - never reimplemented here,
so this can't drift from what the real deploy pipeline actually produces (confirmed against
instance-example and instance-endev: both just run those two standalone scripts, substituting
deploy.env's values for ${VAR} placeholders in params.override.cfg.template).

Operates on an already-checked-out local instance repo (--instance-repo <path>) - no cloning
and no branch-switching happen here; whatever's currently checked out there is what gets used.

Rendered output goes to a disposable temp directory, never inside the instance repo's own
checkout - unlike the real deploy.yml (which clones a fresh, throwaway copy per CI run, so it
can write .rendered/ inside without consequence), --instance-repo points at a real, persistent
local checkout the user keeps using for other things. Writing generated output into it would
leave untracked files sitting there indefinitely. The long-term record of "what config produced
this run" is a separate, deliberate copy into that run's results folder (see run_eval.py) - not
this working copy, which only needs to exist for the duration of one run.
"""
import configparser
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple


class ConfigRenderError(RuntimeError):
    """Raised when render-config.py or validate-config.py fails."""


def _parse_env_file(path: Path) -> Dict[str, str]:
    """Parse deploy.env's simple KEY="value" / KEY=value shell-style format."""
    values: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def render_config(instance_repo_dir: Path) -> Tuple[Path, str]:
    """
    Run the instance repo's own render-config.py + validate-config.py against its real
    deploy.env + params.override.cfg.template. Returns (rendered_config_dir, chabo_tag) -
    rendered_config_dir is a fresh disposable temp directory (not inside instance_repo_dir)
    containing instance.yaml, prompt_overrides.md (if present), and the rendered
    params.override.cfg - the same three files the real deploy step produces, just written
    somewhere that isn't the user's persistent local checkout. Caller is responsible for
    cleaning this up once done with it (e.g. shutil.rmtree) - not deleted automatically here.
    """
    deploy_env = _parse_env_file(instance_repo_dir / "deploy.env")
    chabo_tag = deploy_env.get("CHABO_TAG")
    if not chabo_tag:
        raise ConfigRenderError("deploy.env has no CHABO_TAG - can't determine which orchestrator image to pull")

    instance_config_dir = instance_repo_dir / "orchestrator" / "instance_config"
    rendered_dir = Path(tempfile.mkdtemp(prefix="chabo-eval-"))

    shutil.copy(instance_config_dir / "instance.yaml", rendered_dir / "instance.yaml")
    prompt_overrides = instance_config_dir / "prompt_overrides.md"
    if prompt_overrides.exists():
        shutil.copy(prompt_overrides, rendered_dir / "prompt_overrides.md")

    output_path = rendered_dir / "params.override.cfg"
    env = {**os.environ, **deploy_env}

    for script, script_args in (
        ("render-config.py", [str(instance_config_dir / "params.override.cfg.template"), str(output_path)]),
        ("validate-config.py", [str(output_path)]),
    ):
        result = subprocess.run(
            ["python3", str(instance_repo_dir / "scripts" / script), *script_args],
            capture_output=True, text=True, env=env, cwd=instance_repo_dir,
        )
        if result.returncode != 0:
            raise ConfigRenderError(f"{script} failed:\n{result.stdout}\n{result.stderr}")

    return rendered_dir, chabo_tag


def apply_override(rendered_config_path: Path, override_path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    """
    Layer an override file's [section] values on top of the rendered params.override.cfg,
    in place - a plain configparser merge, independent of deploy.env's ${VAR} substitution
    (which is only for deploy-time constants like URLs, not eval-sweep parameters).

    Returns the swept fields actually applied ({section: {key: value}}), for manifest
    recording - empty dict when no override file is given.
    """
    if override_path is None:
        return {}

    config = configparser.ConfigParser()
    config.read(rendered_config_path)

    override = configparser.ConfigParser()
    override.read(override_path)

    swept: Dict[str, Dict[str, str]] = {}
    for section in override.sections():
        if not config.has_section(section):
            config.add_section(section)
        swept[section] = {}
        for key, value in override.items(section):
            config.set(section, key, value)
            swept[section][key] = value

    with open(rendered_config_path, "w", encoding="utf-8") as f:
        config.write(f)

    return swept
