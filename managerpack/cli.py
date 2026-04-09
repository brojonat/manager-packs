"""ManagerPack CLI — publish skill bundles to r/rayab via Stripe."""

import click

from managerpack.bundles.commands import bundles
from managerpack.stripe.commands import stripe
from managerpack.reddit.commands import reddit
from managerpack.deliver.commands import deliver


@click.group()
@click.option(
    "--env",
    type=click.Choice(["dev", "prod"]),
    default="dev",
    help="Which .env file to load (default: dev).",
)
@click.pass_context
def main(ctx, env):
    """ManagerPack — curated skill bundles for AI agents."""
    from managerpack.env import load_env

    ctx.ensure_object(dict)
    ctx.obj["ENV"] = env
    load_env(env)


main.add_command(bundles)
main.add_command(stripe)
main.add_command(reddit)
main.add_command(deliver)


@main.command()
@click.argument("name")
@click.pass_context
def publish(ctx, name):
    """Create Stripe product and post to r/rayab in one step."""
    ctx.invoke(stripe.get_command(ctx, "create"), name=name)
    ctx.invoke(reddit.get_command(ctx, "post"), name=name)
