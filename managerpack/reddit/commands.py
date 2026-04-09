"""Reddit integration — post bundle listings to r/rayab."""

import os

import click
import praw

from managerpack.bundles.commands import load_manifest, save_manifest, BUNDLES_DIR

SUBREDDIT = "rayab"


def _get_reddit() -> praw.Reddit:
    required = [
        "REDDIT_CLIENT_ID",
        "REDDIT_CLIENT_SECRET",
        "REDDIT_USERNAME",
        "REDDIT_PASSWORD",
    ]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise click.ClickException(f"Missing env vars: {', '.join(missing)}")
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        username=os.environ["REDDIT_USERNAME"],
        password=os.environ["REDDIT_PASSWORD"],
        user_agent=os.environ.get("REDDIT_USER_AGENT", "managerpack/1.0"),
    )


def _format_post_body(m: dict) -> str:
    files_list = "\n".join(f"- `{f}`" for f in m["files"])
    tags_str = ", ".join(m.get("tags", []))
    return f"""## {m['title']}

{m['description']}

**What's included:**
{files_list}

**Tags:** {tags_str}

**Price:** ${m['price_cents'] / 100:.2f}

---

**[Buy now]({m['stripe_payment_link']})**

---

*Drop the files into `.claude/skills/` and your AI agent is immediately skilled in this domain. No training, no videos, no courses.*
"""


@click.group()
def reddit():
    """Reddit operations for r/rayab."""


@reddit.command("list")
def list_posts():
    """List our recent posts on r/rayab."""
    r = _get_reddit()
    sub = r.subreddit(SUBREDDIT)
    me = r.user.me().name
    found = 0
    for submission in sub.new(limit=50):
        if submission.author and submission.author.name == me:
            found += 1
            score = submission.score
            comments = submission.num_comments
            click.echo(f"  [{score:>3}] {submission.title}  ({comments} comments)")
            click.echo(f"        https://reddit.com{submission.permalink}")
    if not found:
        click.echo(f"  No posts by {me} found in r/{SUBREDDIT}.")


@reddit.command()
@click.argument("name")
@click.option("--dry-run", is_flag=True, help="Print post without submitting.")
def post(name, dry_run):
    """Post a bundle listing to r/rayab."""
    m = load_manifest(name)

    if not m.get("stripe_payment_link"):
        raise click.ClickException(f"{name}: no stripe_payment_link. Run `stripe create` first.")

    if m.get("reddit_post_id"):
        click.echo(f"  {name}: already posted (id={m['reddit_post_id']})")
        click.echo(f"  https://reddit.com/r/{SUBREDDIT}/comments/{m['reddit_post_id']}")
        return

    title = f"{m['title']} — ${m['price_cents'] / 100:.0f}"
    body = _format_post_body(m)

    if dry_run:
        click.echo(f"--- DRY RUN ---")
        click.echo(f"Subreddit: r/{SUBREDDIT}")
        click.echo(f"Title: {title}")
        click.echo()
        click.echo(body)
        return

    r = _get_reddit()
    sub = r.subreddit(SUBREDDIT)

    click.echo(f"  Posting to r/{SUBREDDIT}: {title}")
    submission = sub.submit(title=title, selftext=body)

    # Try to set flair
    if m.get("tags"):
        try:
            choices = list(submission.flair.choices())
            tag = m["tags"][0]
            match = next(
                (f for f in choices if f["flair_text"].lower() == tag.lower()),
                None,
            )
            if match:
                submission.flair.select(match["flair_template_id"])
        except Exception:
            pass

    m["reddit_post_id"] = submission.id
    save_manifest(name, m)

    click.echo(f"  Posted: https://reddit.com{submission.permalink}")


@reddit.command()
@click.argument("name")
def update(name):
    """Update an existing Reddit post body from the current manifest."""
    m = load_manifest(name)
    if not m.get("reddit_post_id"):
        raise click.ClickException(f"{name}: no reddit_post_id. Run `reddit post` first.")

    r = _get_reddit()
    submission = r.submission(id=m["reddit_post_id"])
    body = _format_post_body(m)
    submission.edit(body)
    click.echo(f"  {name}: post updated.")
