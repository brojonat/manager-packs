# ManagerPack

Sell curated AI agent skill bundles on Reddit with Stripe checkout and
automated email delivery.

Skill bundles are collections of markdown files that plug into AI coding
agents (Claude Code, Cursor, etc.) and make them competent in a specific
domain. Each bundle is listed as a self-post on a subreddit with a Stripe
payment link. When a buyer completes checkout, a Cloudflare Worker
receives the Stripe webhook and emails the skill files to the buyer
automatically.

Reddit handles discovery, ranking, and reviews. Stripe handles payments.
We build content, not infrastructure.

## Architecture

```
                                ┌──────────────┐
┌──────────┐   click "Buy"     │   Stripe     │
│ r/rayab  │ ────────────────▶ │   Checkout   │
│  post    │                   └──────┬───────┘
└──────────┘                          │ checkout.session.completed
      ▲                               ▼
      │ post                   ┌──────────────────┐
┌─────┴──────┐                 │  CF Worker        │
│ managerpack│                 │  (webhook handler)│
│ CLI        │                 └──────┬───────────┘
└─────┬──────┘                        │
      │ upload                        │ read bundle from R2
      ▼                               │ email via Resend
┌──────────┐                          ▼
│  R2      │                   ┌──────────────┐
│  bucket  │ ◀────────────────│  Buyer inbox  │
└──────────┘                   └──────────────┘
```

**Flow:**
1. Operator creates a bundle (markdown skill files + manifest)
2. CLI creates a Stripe product/price/payment link
3. CLI posts a listing to the subreddit with the payment link
4. Buyer clicks "Buy now", completes Stripe checkout
5. Stripe fires `checkout.session.completed` webhook
6. Cloudflare Worker reads bundle files from R2, emails them to the buyer

## Quick start

```bash
pip install -e .
```

### 1. Set up credentials

Create `.env.dev` (sandbox) and `.env.prod` (live) in the project root.
Both files are gitignored. See [Environment variables](#environment-variables)
for the required keys.

### 2. Create a bundle

```bash
mkdir -p bundles/my-skill
```

Add your skill files (markdown) and a `manifest.json`:

```json
{
  "name": "my-skill",
  "title": "My Skill — Human-Readable Title",
  "description": "What this skill teaches an AI agent to do.",
  "price_cents": 500,
  "tags": ["backend", "python"],
  "files": ["SKILL.md"],
  "stripe_product_id": null,
  "stripe_price_id": null,
  "stripe_payment_link": null,
  "reddit_post_id": null
}
```

The `stripe_*` and `reddit_*` fields start as `null` and get populated
automatically when you publish.

### 3. Validate, publish, and upload

```bash
# Validate the manifest and files
managerpack bundles validate my-skill

# Create Stripe product + payment link, then post to Reddit
managerpack publish my-skill

# Upload bundle files to R2 so the webhook worker can deliver them
managerpack bundles upload my-skill
```

Or do each step individually:

```bash
managerpack stripe create my-skill    # create Stripe product
managerpack reddit post my-skill      # post to subreddit
managerpack bundles upload my-skill   # upload to R2
```

## CLI reference

All commands accept `--env dev|prod` (default: `dev`) to select which
`.env` file to load.

```bash
managerpack --env prod <command>      # use production credentials
```

### Bundle management

```bash
managerpack bundles list              # list bundles and publish status
managerpack bundles validate <name>   # check manifest + files
managerpack bundles upload <name>     # upload to R2
managerpack bundles upload --all      # upload all bundles to R2
```

### Stripe

```bash
managerpack stripe create <name>      # create product + price + payment link
managerpack stripe list               # list ManagerPack products in Stripe
managerpack stripe sync <name>        # update Stripe product from manifest
```

### Reddit

```bash
managerpack reddit post <name>        # post listing to subreddit
managerpack reddit post <name> --dry-run  # preview without posting
managerpack reddit update <name>      # edit existing post from manifest
managerpack reddit list               # list our posts on the subreddit
```

### Delivery

```bash
managerpack deliver send <email> <name>   # manually email a bundle
```

### Publish (shortcut)

```bash
managerpack publish <name>            # stripe create + reddit post
```

## Bundle format

Each bundle lives in `bundles/<name>/` with a `manifest.json` and one or
more markdown skill files:

```
bundles/
├── scikit-learn/
│   ├── manifest.json
│   └── SKILL.md
├── fastapi-service/
│   ├── manifest.json
│   └── SKILL.md
└── temporal-python/
    ├── manifest.json
    ├── SKILL.md
    ├── EXAMPLES.md
    └── REFERENCE.md
```

### manifest.json fields

| Field | Type | Description |
|---|---|---|
| `name` | string | Bundle identifier (matches directory name) |
| `title` | string | Human-readable title shown in Stripe and Reddit |
| `description` | string | What the skill teaches an agent |
| `price_cents` | int | Price in cents (500 = $5.00) |
| `tags` | string[] | Category tags; first tag used for Reddit flair |
| `files` | string[] | List of files included in the bundle |
| `stripe_product_id` | string\|null | Populated by `stripe create` |
| `stripe_price_id` | string\|null | Populated by `stripe create` |
| `stripe_payment_link` | string\|null | Populated by `stripe create` |
| `reddit_post_id` | string\|null | Populated by `reddit post` |

### Skill file format

Skill files are markdown documents with YAML frontmatter:

```markdown
---
name: my-skill
description: When the agent should use this skill and what it does.
---

# Skill Title

Instructions, conventions, code examples, and patterns.
```

## Webhook worker

The Cloudflare Worker at `worker/index.js` handles automated delivery.
It receives Stripe webhooks, reads bundle files from R2, and emails
them to the buyer via Resend.

### How it works

1. Stripe sends `checkout.session.completed` to the Worker
2. Worker verifies the webhook signature (HMAC-SHA256)
3. Worker checks `metadata.platform === "managerpack"` (ignores other events)
4. Worker reads bundle files from R2 at `managerpacks/bundles/<name>/`
5. Worker emails the files as attachments via Resend
6. Worker writes a delivery receipt to R2 at `managerpacks/deliveries/`

### Deploying the worker

```bash
cd worker
CLOUDFLARE_API_TOKEN=<token> npx wrangler deploy
```

### Setting worker secrets

```bash
cd worker
echo "<value>" | CLOUDFLARE_API_TOKEN=<token> npx wrangler secret put STRIPE_WEBHOOK_SECRET
echo "<value>" | CLOUDFLARE_API_TOKEN=<token> npx wrangler secret put RESEND_API_KEY
```

### Registering the Stripe webhook endpoint

Create a webhook endpoint in Stripe (dashboard or API) pointing to:

```
https://<worker-name>.<subdomain>.workers.dev/webhooks/stripe
```

Subscribe to the `checkout.session.completed` event. Save the webhook
signing secret and set it as the `STRIPE_WEBHOOK_SECRET` worker secret.

### R2 layout

All ManagerPack data lives under the `managerpacks/` prefix in the
shared R2 bucket:

```
<bucket>/
└── managerpacks/
    ├── bundles/
    │   ├── scikit-learn/
    │   │   └── SKILL.md
    │   └── fastapi-service/
    │       └── SKILL.md
    └── deliveries/
        ├── 1712345678-scikit-learn.json
        └── 1712345679-fastapi-service.json
```

## Stripe product metadata

All ManagerPack products are tagged with `metadata.platform = "managerpack"`
so they can be filtered from other products in the same Stripe account.

| Key | Value | Purpose |
|---|---|---|
| `platform` | `"managerpack"` | Filter ManagerPack products from others |
| `bundle` | `"scikit-learn"` | Maps checkout events to bundle names |
| `tags` | `"data-science,ml,python"` | Comma-separated category tags |

Payment links also carry `platform` and `bundle` in their metadata so
the webhook worker knows which bundle to deliver.

## Reddit post format

Posts follow this template:

```
Title: {title} — ${price}

Body:
## {title}

{description}

**What's included:**
- `SKILL.md`
- `EXAMPLES.md`

**Tags:** data-science, ml, python

**Price:** $5.00

---

**[Buy now]({stripe_payment_link})**

---

*Drop the files into `.claude/skills/` and your AI agent is immediately
skilled in this domain. No training, no videos, no courses.*
```

Upvotes and comments on the subreddit serve as the review/ranking system.

## Environment variables

Create `.env.dev` and `.env.prod` in the project root. These are
gitignored and must never be committed.

```bash
# .env.dev (or .env.prod)

# Stripe
STRIPE_SECRET_KEY=sk_test_...       # sk_live_... in prod
STRIPE_WEBHOOK_SECRET=whsec_...     # from Stripe webhook endpoint

# Cloudflare
CLOUDFLARE_API_TOKEN=...            # needs Workers, R2 permissions
CLOUDFLARE_ACCOUNT_ID=...           # your Cloudflare account ID

# Email delivery
RESEND_API_KEY=re_...

# Reddit (PRAW)
REDDIT_CLIENT_ID=...                # from reddit.com/prefs/apps
REDDIT_CLIENT_SECRET=...
REDDIT_USERNAME=...
REDDIT_PASSWORD=...
REDDIT_USER_AGENT=linux:managerpack:v0.1.0 (by /u/yourname)
```

### Stripe setup

1. Create a Stripe account (or use an existing one)
2. Get your API keys from the Stripe dashboard
3. The CLI creates products with `metadata.platform = "managerpack"` so
   they don't conflict with other products in the same account

### Reddit setup

1. Go to https://www.reddit.com/prefs/apps/
2. Create a "script" type application
3. Use the client ID and secret in your `.env` file

### Cloudflare setup

1. Create an API token with Workers Scripts, R2, and Account read
   permissions
2. The Worker uses the same R2 bucket as other projects but under the
   `managerpacks/` prefix

### Resend setup

1. Create a Resend account and verify your sending domain
2. Use the API key in both `.env` files and as a Worker secret

## Project structure

```
manager-pack/
├── bundles/                      # skill bundle content
│   ├── scikit-learn/
│   │   ├── manifest.json
│   │   └── SKILL.md
│   ├── fastapi-service/
│   ├── go-service/
│   ├── ibis-data/
│   ├── k8s-deployment/
│   └── temporal-python/
├── managerpack/                  # CLI source
│   ├── cli.py                    # root Click CLI
│   ├── env.py                    # .env.dev / .env.prod loader
│   ├── bundles/commands.py       # list, validate, upload
│   ├── stripe/commands.py        # create, list, sync
│   ├── reddit/commands.py        # post, list, update
│   └── deliver/commands.py       # manual email delivery
├── worker/
│   ├── index.js                  # Cloudflare Worker (webhook + delivery)
│   └── wrangler.toml             # Worker config (R2 binding, vars)
├── .env.dev                      # dev/sandbox credentials (gitignored)
├── .env.prod                     # production credentials (gitignored)
├── .gitignore
├── pyproject.toml
├── NOTES.md                      # project history / ideation
└── SPEC.md                       # implementation spec
```

## Cost

| Component | Free tier | At scale |
|---|---|---|
| R2 storage | 10 GB free | Markdown bundles = free |
| R2 reads | 10M/month free | Free |
| Workers | 100k req/day free | Free |
| Resend | 100 emails/day free | $20/mo at 5k emails/mo |
| Reddit | Free | Free |
| Stripe | 2.9% + $0.30/txn | $0.45 per $5 sale |
| **Net per $5 sale** | | **$4.55** |
