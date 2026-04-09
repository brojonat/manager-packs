# ManagerPack Implementation Spec

This document specifies the platform that powers ManagerPack: listing skill
bundles on r/rayab, handling Stripe checkout, and delivering purchased bundles
to buyers. The architecture borrows heavily from iybi-twc's proven patterns for
Stripe integration, Cloudflare Workers, and Reddit automation via PRAW.

## Overview

```
┌────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌────────────┐
│  r/rayab   │───▶│ Stripe       │───▶│ Cloudflare      │───▶│ Buyer gets │
│  post      │    │ Checkout     │    │ Worker webhook   │    │ bundle     │
└────────────┘    └──────────────┘    └─────────────────┘    └────────────┘
      ▲                                       │
      │                                       ▼
┌─────┴──────┐                        ┌───────────────┐
│  CLI:      │                        │ Resend email   │
│  publish   │                        │ w/ repo invite │
│  command   │                        │ or zip link    │
└────────────┘                        └───────────────┘
```

The operator (us) runs a CLI command that:
1. Creates a Stripe product + price + payment link for a skill bundle
2. Posts the listing to r/rayab with the payment link
3. When a buyer completes checkout, a Stripe webhook fires
4. Our Cloudflare Worker receives the webhook, and triggers delivery (email
   with a download link or private repo invite)

## 1. Skill Bundle Structure

A "bundle" is a directory of markdown files that lives in this repo under
`bundles/`. Each bundle has a `manifest.json` that describes it:

```
bundles/
├── scikit-learn/
│   ├── manifest.json
│   ├── SKILL.md
│   ├── EXAMPLES.md          # optional
│   └── REFERENCE.md         # optional
├── fastapi-service/
│   ├── manifest.json
│   └── SKILL.md
└── temporal-python/
    ├── manifest.json
    └── SKILL.md
```

### manifest.json

```json
{
  "name": "scikit-learn",
  "title": "scikit-learn ML Pipelines",
  "description": "Build reproducible ML workflows with scikit-learn Pipelines, ColumnTransformers, cross-validation, and MLflow tracking.",
  "price_cents": 500,
  "tags": ["ml", "data-science", "python"],
  "files": ["SKILL.md", "EXAMPLES.md", "REFERENCE.md"],
  "stripe_product_id": null,
  "stripe_price_id": null,
  "stripe_payment_link": null,
  "reddit_post_id": null
}
```

The `stripe_*` and `reddit_*` fields start as null and get populated by the
CLI when the bundle is published. This makes `manifest.json` the local source
of truth for what has been published where.

## 2. CLI

Built with Click (matching iybi-twc's CLI pattern). Entry point: `managerpack`.

### Commands

```bash
# Bundle management
managerpack bundles list                    # list all bundles and their publish status
managerpack bundles validate <name>         # validate manifest + files exist

# Stripe operations
managerpack stripe create <name>            # create product + price + payment link
managerpack stripe sync <name>              # update price or metadata from manifest
managerpack stripe list                     # list all products in Stripe

# Reddit operations
managerpack reddit post <name>              # post bundle listing to r/rayab
managerpack reddit update <name>            # edit existing post (e.g. update flair)
managerpack reddit list                     # list our posts on r/rayab

# Full publish pipeline
managerpack publish <name>                  # create stripe product → post to reddit
managerpack publish all                     # publish all unpublished bundles

# Delivery (manual, for debugging)
managerpack deliver <email> <name>          # manually deliver a bundle to an email
```

### `managerpack stripe create <name>`

Mirrors iybi-twc's `consulting/stripe/commands.py` pattern:

```python
import stripe as stripe_lib

def create(bundle_name: str):
    manifest = load_manifest(bundle_name)

    # 1. Create product
    product = stripe_lib.Product.create(
        name=manifest["title"],
        description=manifest["description"],
        metadata={
            "bundle": manifest["name"],       # key for filtering
            "tags": ",".join(manifest["tags"]),
        },
    )

    # 2. Create price
    price = stripe_lib.Price.create(
        product=product.id,
        unit_amount=manifest["price_cents"],  # 500 = $5.00
        currency="usd",
    )

    # 3. Create payment link
    payment_link = stripe_lib.PaymentLink.create(
        line_items=[{"price": price.id, "quantity": 1}],
        # Collect email so we know where to deliver
        custom_fields=[],
        after_completion={
            "type": "redirect",
            "redirect": {"url": "https://managerpacks.com/thank-you"},
        },
    )

    # 4. Update manifest
    manifest["stripe_product_id"] = product.id
    manifest["stripe_price_id"] = price.id
    manifest["stripe_payment_link"] = payment_link.url
    save_manifest(bundle_name, manifest)
```

### `managerpack reddit post <name>`

Uses PRAW, same auth pattern as iybi-twc's `consulting/customers/commands.py`:

```python
import praw

def _get_reddit_client() -> praw.Reddit:
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        username=os.environ["REDDIT_USERNAME"],
        password=os.environ["REDDIT_PASSWORD"],
        user_agent=os.environ.get("REDDIT_USER_AGENT", "managerpack/1.0"),
    )

SUBREDDIT = "rayab"

def post(bundle_name: str):
    manifest = load_manifest(bundle_name)
    assert manifest["stripe_payment_link"], "Run `stripe create` first"

    reddit = _get_reddit_client()
    sub = reddit.subreddit(SUBREDDIT)

    title = f"{manifest['title']} — ${manifest['price_cents'] / 100:.0f}"

    body = f"""## {manifest['title']}

{manifest['description']}

**What's included:**
{chr(10).join(f'- `{f}`' for f in manifest['files'])}

**Tags:** {', '.join(manifest['tags'])}

**Price:** ${manifest['price_cents'] / 100:.2f}

---

[Buy now]({manifest['stripe_payment_link']})
"""

    submission = sub.submit(title=title, selftext=body)

    # Flair the post with the primary tag
    if manifest["tags"]:
        try:
            flair_choices = list(submission.flair.choices())
            match = next(
                (f for f in flair_choices if f["flair_text"].lower() == manifest["tags"][0]),
                None,
            )
            if match:
                submission.flair.select(match["flair_template_id"])
        except Exception:
            pass  # flair is nice-to-have

    manifest["reddit_post_id"] = submission.id
    save_manifest(bundle_name, manifest)
```

### `managerpack publish <name>`

Orchestrates the full pipeline:

```python
def publish(bundle_name: str):
    manifest = load_manifest(bundle_name)

    if not manifest.get("stripe_product_id"):
        stripe_create(bundle_name)

    if not manifest.get("reddit_post_id"):
        reddit_post(bundle_name)
```

## 3. Stripe Webhook Handler

A Cloudflare Worker that receives `checkout.session.completed` events and
triggers bundle delivery. Follows iybi-twc's `worker/index.js` webhook pattern
exactly.

### Worker: `worker/index.js`

```javascript
export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === "/webhooks/stripe" && request.method === "POST") {
      return handleStripeWebhook(request, env);
    }

    // Health check
    if (url.pathname === "/health") {
      return new Response("ok");
    }

    return new Response("not found", { status: 404 });
  },
};

async function handleStripeWebhook(request, env) {
  const body = await request.text();
  const sig = request.headers.get("stripe-signature");

  // Verify signature (same HMAC-SHA256 pattern as iybi-twc)
  const verified = await verifyStripeSignature(body, sig, env.STRIPE_WEBHOOK_SECRET);
  if (!verified) {
    return new Response("invalid signature", { status: 400 });
  }

  const event = JSON.parse(body);

  if (event.type === "checkout.session.completed") {
    const session = event.data.object;
    const email = session.customer_details?.email;
    const bundle = session.metadata?.bundle;

    if (!email || !bundle) {
      return new Response("missing metadata", { status: 400 });
    }

    // Queue delivery
    const deliveryKey = `deliveries/${Date.now()}-${bundle}.json`;
    await env.BUCKET.put(deliveryKey, JSON.stringify({
      email,
      bundle,
      stripe_session_id: session.id,
      purchased_at: new Date().toISOString(),
    }));

    // Fire delivery immediately via service binding or queue
    // (or let a cron poll deliveries/)
    await deliverBundle(env, email, bundle);

    return new Response("ok");
  }

  return new Response("unhandled event", { status: 200 });
}
```

### Webhook Signature Verification

Lifted directly from iybi-twc's `worker/index.js` (lines 179-209):

```javascript
async function verifyStripeSignature(payload, sigHeader, secret) {
  const parts = Object.fromEntries(
    sigHeader.split(",").map((p) => {
      const [k, v] = p.split("=");
      return [k, v];
    })
  );
  const timestamp = parts["t"];
  const signature = parts["v1"];

  const signedPayload = `${timestamp}.${payload}`;
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const mac = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(signedPayload));
  const expected = Array.from(new Uint8Array(mac))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");

  // Timing-safe comparison
  if (expected.length !== signature.length) return false;
  let result = 0;
  for (let i = 0; i < expected.length; i++) {
    result |= expected.charCodeAt(i) ^ signature.charCodeAt(i);
  }
  return result === 0;
}
```

### Bundle Delivery

Two delivery options. Start with email (simplest), graduate to repo access
if we see repeat buyers or need versioning.

#### Option A: Email with download link (MVP)

1. Worker zips the bundle files from R2
2. Uploads the zip to R2 with a time-limited key
3. Sends email via Resend with the download link

```javascript
async function deliverBundle(env, email, bundleName) {
  // Bundle files are pre-uploaded to R2 at bundles/<name>/
  const zipKey = `downloads/${bundleName}-${Date.now()}.zip`;

  // Generate zip (using a lightweight zip library or pre-built zips)
  const files = await listBundleFiles(env, bundleName);
  const zip = await createZip(files);
  await env.BUCKET.put(zipKey, zip, {
    httpMetadata: { contentType: "application/zip" },
    customMetadata: { expires: new Date(Date.now() + 7 * 86400000).toISOString() },
  });

  const downloadUrl = `https://files.managerpacks.com/${zipKey}`;

  // Send via Resend
  await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.RESEND_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      from: "ManagerPack <delivery@managerpacks.com>",
      to: email,
      subject: `Your ManagerPack bundle: ${bundleName}`,
      html: `
        <h2>Thanks for your purchase!</h2>
        <p>Download your skill bundle here:</p>
        <p><a href="${downloadUrl}">${bundleName}.zip</a></p>
        <p>This link expires in 7 days. Drop the .md files into your
        project's <code>.claude/skills/</code> directory and your agents
        will pick them up immediately.</p>
        <p>Questions? Reply to this email.</p>
      `,
    }),
  });
}
```

#### Option B: Private repo invite (future)

For buyers who want updates, grant read access to a private GitHub repo
containing all purchased bundles. Use GitHub API to add the buyer as a
collaborator.

```javascript
async function grantRepoAccess(email, githubUsername) {
  await fetch(
    `https://api.github.com/repos/managerpack/bundles-private/collaborators/${githubUsername}`,
    {
      method: "PUT",
      headers: {
        Authorization: `Bearer ${GITHUB_TOKEN}`,
        Accept: "application/vnd.github+json",
      },
      body: JSON.stringify({ permission: "pull" }),
    }
  );
}
```

This requires collecting the buyer's GitHub username at checkout (via Stripe
custom fields) and is more complex. Save for later.

## 4. Cloudflare Infrastructure

Minimal setup, matching iybi-twc's pattern but much simpler (no site hosting,
no email worker, no KV customer registry needed for MVP).

### wrangler.toml

```toml
name = "managerpack-webhook"
main = "worker/index.js"
compatibility_date = "2024-01-01"

[vars]
ENVIRONMENT = "production"

[[r2_buckets]]
binding = "BUCKET"
bucket_name = "managerpack"
```

### R2 Bucket Layout

```
managerpack/
├── bundles/                      # pre-uploaded bundle files
│   ├── scikit-learn/
│   │   ├── SKILL.md
│   │   ├── EXAMPLES.md
│   │   └── REFERENCE.md
│   └── fastapi-service/
│       └── SKILL.md
├── downloads/                    # generated zips for buyers
│   └── scikit-learn-1712345678.zip
└── deliveries/                   # webhook delivery receipts
    └── 1712345678-scikit-learn.json
```

### CLI: Upload bundles to R2

```bash
managerpack bundles upload <name>     # upload bundle files to R2
managerpack bundles upload --all      # upload all bundles
```

Uses `npx wrangler r2 object put` per file, same as iybi-twc's
`_r2_upload()` in `consulting/deploy/commands.py`.

## 5. Subreddit Setup: r/rayab

### Subreddit Configuration

- **Type:** Public
- **Post types:** Self-posts only (link posts disabled — all content is in
  the post body with Stripe payment links)
- **Post flairs:** One per domain category:
  - `data-science`, `ml`, `data-engineering`, `backend`, `frontend`,
    `infrastructure`, `devops`, `notebooks`, `workflows`
- **Sidebar/wiki:** README content (what ManagerPack is, how skills work,
  how to install)
- **Automod:** Remove posts not from our account (initially — open to
  community submissions later)

### Post Format

Each post follows a consistent template:

```
Title: scikit-learn ML Pipelines — $5

Body:
## scikit-learn ML Pipelines

Build reproducible ML workflows with scikit-learn Pipelines,
ColumnTransformers, cross-validation, and MLflow tracking.

### What's included

- `SKILL.md` — Core pipeline patterns, preprocessing, model selection
- `EXAMPLES.md` — Worked examples with real datasets
- `REFERENCE.md` — Quick-reference for common transforms and metrics

### Tags

data-science, ml, python

---

**$5** — [Buy now](https://buy.stripe.com/xxx)

---

*Drop the files into `.claude/skills/` and your agent knows how to
build production ML pipelines. No training, no videos, no courses.*
```

### Reddit as Discovery + Reviews

- **Upvotes** = organic ranking of best bundles
- **Comments** = reviews, questions, feature requests
- **Flairs** = category filtering
- **Search** = Reddit's built-in search indexes our posts
- **Zero infrastructure** for discovery, ranking, or reviews

## 6. Environment Variables

```bash
# Stripe
STRIPE_SECRET_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...

# Reddit (PRAW)
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
REDDIT_USERNAME=...
REDDIT_PASSWORD=...
REDDIT_USER_AGENT=managerpack/1.0

# Cloudflare
CLOUDFLARE_API_TOKEN=...
CLOUDFLARE_ACCOUNT_ID=...

# Email delivery
RESEND_API_KEY=...
```

## 7. Implementation Order

### Phase 1: MVP (manual delivery)

1. Set up `bundles/` directory structure with manifests
2. Build CLI: `bundles list`, `bundles validate`
3. Build CLI: `stripe create` (product + price + payment link)
4. Build CLI: `reddit post` (PRAW submission to r/rayab)
5. Build CLI: `publish` (orchestrates stripe + reddit)
6. Manual delivery: buyer emails us, we send the files

### Phase 2: Automated delivery

7. Deploy Cloudflare Worker with Stripe webhook handler
8. Build CLI: `bundles upload` (push files to R2)
9. Worker zips and emails bundle on `checkout.session.completed`
10. Build CLI: `deliver` for manual re-delivery / debugging

### Phase 3: Polish

11. Add download link expiry + cleanup cron
12. Purchase tracking / analytics (deliveries stored in R2)
13. Open r/rayab to community skill submissions
14. Private repo access option for repeat buyers

## 8. Cost

| Component | Free tier | At scale |
|---|---|---|
| R2 storage | 10 GB free | Markdown bundles ≈ free |
| R2 reads | 10M/month free | Free |
| Workers | 100k req/day free | Free |
| Resend | 100 emails/day free | $20/mo at 5k/mo |
| Reddit | Free | Free |
| Stripe | 2.9% + $0.30/txn | $0.45 per $5 sale |
| **Total** | **$0** | **~$0.45/sale** |

Net per sale: $5.00 - $0.45 (Stripe) = **$4.55**
