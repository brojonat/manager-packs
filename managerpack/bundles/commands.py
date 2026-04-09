"""Bundle management commands."""

import json
from pathlib import Path

import click

BUNDLES_DIR = Path(__file__).resolve().parent.parent.parent / "bundles"


def _bundle_dir(name: str) -> Path:
    d = BUNDLES_DIR / name
    if not d.is_dir():
        raise click.ClickException(f"Bundle not found: {name} (looked in {BUNDLES_DIR})")
    return d


def load_manifest(name: str) -> dict:
    manifest_path = _bundle_dir(name) / "manifest.json"
    if not manifest_path.exists():
        raise click.ClickException(f"Missing manifest.json in {_bundle_dir(name)}")
    return json.loads(manifest_path.read_text())


def save_manifest(name: str, manifest: dict) -> None:
    manifest_path = _bundle_dir(name) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


@click.group()
def bundles():
    """Manage skill bundles."""


@bundles.command("list")
def list_bundles():
    """List all bundles and their publish status."""
    if not BUNDLES_DIR.is_dir():
        click.echo("No bundles/ directory found.")
        return
    for d in sorted(BUNDLES_DIR.iterdir()):
        if not d.is_dir():
            continue
        manifest_path = d / "manifest.json"
        if not manifest_path.exists():
            click.echo(f"  {d.name}  [no manifest]")
            continue
        m = json.loads(manifest_path.read_text())
        stripe_ok = "stripe" if m.get("stripe_product_id") else "      "
        reddit_ok = "reddit" if m.get("reddit_post_id") else "      "
        price = f"${m.get('price_cents', 0) / 100:.0f}"
        click.echo(f"  {d.name:<30} {price:>5}  [{stripe_ok}] [{reddit_ok}]")


@bundles.command()
@click.argument("name")
def validate(name):
    """Validate a bundle's manifest and files."""
    m = load_manifest(name)
    d = _bundle_dir(name)
    errors = []

    for field in ("name", "title", "description", "price_cents", "tags", "files"):
        if field not in m:
            errors.append(f"Missing required field: {field}")

    for f in m.get("files", []):
        if not (d / f).exists():
            errors.append(f"Listed file missing: {f}")

    if errors:
        for e in errors:
            click.echo(f"  ERROR: {e}", err=True)
        raise SystemExit(1)
    click.echo(f"  {name}: OK ({len(m['files'])} files)")


@bundles.command()
@click.argument("name", required=False)
@click.option("--all", "upload_all", is_flag=True, help="Upload all bundles.")
def upload(name, upload_all):
    """Upload bundle files to R2 under managerpacks/bundles/<name>/."""
    import os
    import subprocess

    bucket = "iybi-sites"
    prefix = "managerpacks/bundles"

    names = []
    if upload_all:
        names = [d.name for d in sorted(BUNDLES_DIR.iterdir()) if d.is_dir() and (d / "manifest.json").exists()]
    elif name:
        names = [name]
    else:
        raise click.ClickException("Provide a bundle name or --all.")

    for bundle_name in names:
        m = load_manifest(bundle_name)
        d = _bundle_dir(bundle_name)
        count = 0
        for f in m["files"]:
            fpath = d / f
            key = f"{bucket}/{prefix}/{bundle_name}/{f}"
            result = subprocess.run(
                ["npx", "wrangler", "r2", "object", "put", key, "--file", str(fpath), "--remote"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise click.ClickException(f"R2 upload failed for {f}: {result.stderr}")
            count += 1
        click.echo(f"  {bundle_name}: uploaded {count} files to r2://{prefix}/{bundle_name}/")
