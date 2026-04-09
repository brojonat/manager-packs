"""Stripe integration — create products, prices, and payment links for bundles."""

import os
import time

import click
import stripe as stripe_lib

from managerpack.bundles.commands import load_manifest, save_manifest, BUNDLES_DIR

STRIPE_DELAY = 0.3  # seconds between API calls


def _ensure_stripe():
    key = os.environ.get("STRIPE_SECRET_KEY")
    if not key:
        raise click.ClickException("Missing STRIPE_SECRET_KEY in environment.")
    stripe_lib.api_key = key


@click.group()
def stripe():
    """Stripe product management."""


@stripe.command("list")
def list_products():
    """List all ManagerPack products in Stripe."""
    _ensure_stripe()
    products = stripe_lib.Product.list(active=True, limit=100)
    found = 0
    for p in products.data:
        meta = p.metadata.to_dict() if p.metadata else {}
        if meta.get("platform") != "managerpack":
            continue
        found += 1
        prices = stripe_lib.Price.list(product=p.id, active=True, limit=1)
        price_str = ""
        if prices.data:
            price_str = f"${prices.data[0].unit_amount / 100:.2f}"
        payment_url = meta.get("payment_url", "")
        click.echo(f"  {p.name:<35} {price_str:>7}  bundle={meta.get('bundle', '?')}")
        if payment_url:
            click.echo(f"    {payment_url}")
        time.sleep(STRIPE_DELAY)
    if not found:
        click.echo("  No ManagerPack products found. Run `managerpack stripe create <name>` to create one.")


@stripe.command()
@click.argument("name")
def create(name):
    """Create a Stripe product + price + payment link for a bundle."""
    _ensure_stripe()
    m = load_manifest(name)

    if m.get("stripe_product_id"):
        click.echo(f"  {name}: already has Stripe product {m['stripe_product_id']}")
        click.echo("  Use `stripe sync` to update, or clear stripe_product_id in manifest to recreate.")
        return

    # 1. Create product
    click.echo(f"  Creating product: {m['title']}...")
    product = stripe_lib.Product.create(
        name=m["title"],
        description=m["description"],
        metadata={
            "platform": "managerpack",
            "bundle": m["name"],
            "tags": ",".join(m.get("tags", [])),
        },
    )
    time.sleep(STRIPE_DELAY)

    # 2. Create price
    click.echo(f"  Creating price: ${m['price_cents'] / 100:.2f}...")
    price = stripe_lib.Price.create(
        product=product.id,
        unit_amount=m["price_cents"],
        currency="usd",
    )
    time.sleep(STRIPE_DELAY)

    # 3. Create payment link
    click.echo("  Creating payment link...")
    payment_link = stripe_lib.PaymentLink.create(
        line_items=[{"price": price.id, "quantity": 1}],
        metadata={
            "platform": "managerpack",
            "bundle": m["name"],
        },
        after_completion={
            "type": "redirect",
            "redirect": {"url": "https://reddit.com/r/rayab"},
        },
    )

    # 4. Update manifest
    m["stripe_product_id"] = product.id
    m["stripe_price_id"] = price.id
    m["stripe_payment_link"] = payment_link.url
    save_manifest(name, m)

    click.echo(f"  Product: {product.id}")
    click.echo(f"  Price:   {price.id}")
    click.echo(f"  Link:    {payment_link.url}")


@stripe.command()
@click.argument("name")
def sync(name):
    """Sync price/metadata from manifest to existing Stripe product."""
    _ensure_stripe()
    m = load_manifest(name)

    if not m.get("stripe_product_id"):
        raise click.ClickException(f"{name}: no stripe_product_id. Run `stripe create` first.")

    # Update product metadata
    stripe_lib.Product.modify(
        m["stripe_product_id"],
        name=m["title"],
        description=m["description"],
        metadata={
            "platform": "managerpack",
            "bundle": m["name"],
            "tags": ",".join(m.get("tags", [])),
        },
    )
    click.echo(f"  {name}: product metadata updated.")
