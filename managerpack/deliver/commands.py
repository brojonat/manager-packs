"""Bundle delivery — email purchased bundles to buyers."""

import base64
import json
import os
from pathlib import Path

import click
import resend

from managerpack.bundles.commands import load_manifest, BUNDLES_DIR


def _ensure_resend():
    key = os.environ.get("RESEND_API_KEY")
    if not key:
        raise click.ClickException("Missing RESEND_API_KEY in environment.")
    resend.api_key = key


def _build_email_html(bundle_name: str, m: dict) -> str:
    files_list = "".join(f"<li><code>{f}</code></li>" for f in m["files"])
    return f"""\
<h2>Your ManagerPack bundle: {m['title']}</h2>
<p>Thanks for your purchase! Your skill files are attached to this email.</p>
<h3>What's included</h3>
<ul>{files_list}</ul>
<h3>How to use</h3>
<ol>
  <li>Save the attached <code>.md</code> files</li>
  <li>Drop them into your project's <code>.claude/skills/{bundle_name}/</code> directory</li>
  <li>Your AI agent will pick them up immediately</li>
</ol>
<p>Questions? Reply to this email.</p>
<p>&mdash; ManagerPack</p>
"""


@click.group()
def deliver():
    """Deliver bundles to buyers."""


@deliver.command("send")
@click.argument("email")
@click.argument("name")
def send(email, name):
    """Manually deliver a bundle to an email address."""
    _ensure_resend()
    m = load_manifest(name)
    d = BUNDLES_DIR / name

    # Build attachments from bundle files
    attachments = []
    for f in m["files"]:
        fpath = d / f
        if not fpath.exists():
            raise click.ClickException(f"Bundle file missing: {fpath}")
        attachments.append({
            "filename": f,
            "content": base64.b64encode(fpath.read_bytes()).decode(),
        })

    click.echo(f"  Delivering {name} ({len(attachments)} files) to {email}...")
    resp = resend.Emails.send({
        "from": "ManagerPack <delivery@iybi-twc.com>",
        "to": email,
        "subject": f"Your ManagerPack bundle: {m['title']}",
        "html": _build_email_html(name, m),
        "attachments": attachments,
    })
    click.echo(f"  Sent: {resp.get('id', resp)}")
