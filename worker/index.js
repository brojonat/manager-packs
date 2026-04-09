/**
 * ManagerPack webhook worker.
 *
 * Receives Stripe checkout.session.completed events for bundle purchases
 * and delivers the bundle files to the buyer via email (Resend).
 *
 * R2 layout:
 *   managerpacks/bundles/<name>/SKILL.md
 *   managerpacks/bundles/<name>/EXAMPLES.md
 *   managerpacks/deliveries/<timestamp>-<bundle>.json
 *
 * This worker is deployed alongside the existing iybi-twc worker on the
 * same Stripe account. Both receive all webhook events. We filter to only
 * handle events with metadata.platform === "managerpack". The iybi-twc
 * worker already ignores unknown metadata types (logs + returns 200), so
 * there is no conflict.
 */

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === "/webhooks/stripe" && request.method === "POST") {
      return handleStripeWebhook(request, env);
    }

    if (url.pathname === "/health") {
      return new Response("ok");
    }

    return new Response("not found", { status: 404 });
  },
};

// --- Stripe webhook ---

async function handleStripeWebhook(request, env) {
  const body = await request.text();
  const sig = request.headers.get("stripe-signature");

  if (!sig) {
    return new Response("missing signature", { status: 400 });
  }

  const verified = await verifyStripeSignature(
    body,
    sig,
    env.STRIPE_WEBHOOK_SECRET
  );
  if (!verified) {
    return new Response("invalid signature", { status: 400 });
  }

  const event = JSON.parse(body);

  // Only handle checkout completions for our platform
  if (event.type !== "checkout.session.completed") {
    return new Response("ignored", { status: 200 });
  }

  const session = event.data.object;
  const meta = session.metadata || {};

  // Ignore events not meant for us
  if (meta.platform !== "managerpack") {
    return new Response("not managerpack", { status: 200 });
  }

  const email = session.customer_details?.email;
  const bundle = meta.bundle;

  if (!email || !bundle) {
    console.error("managerpack: missing email or bundle in session", session.id);
    return new Response("missing data", { status: 400 });
  }

  try {
    await deliverBundle(env, email, bundle, session.id);
  } catch (err) {
    console.error("managerpack: delivery failed", err);
    return new Response("delivery error", { status: 500 });
  }

  return new Response("ok");
}

// --- Delivery ---

async function deliverBundle(env, email, bundleName, sessionId) {
  const prefix = env.R2_PREFIX || "managerpacks";

  // 1. List bundle files from R2
  const listResult = await env.BUCKET.list({
    prefix: `${prefix}/bundles/${bundleName}/`,
  });

  if (!listResult.objects.length) {
    throw new Error(`No files found in R2 for bundle: ${bundleName}`);
  }

  // 2. Read each file and build attachments
  const attachments = [];
  for (const obj of listResult.objects) {
    const file = await env.BUCKET.get(obj.key);
    if (!file) continue;
    const content = await file.text();
    const filename = obj.key.split("/").pop();
    // Resend expects base64-encoded content for attachments
    attachments.push({
      filename,
      content: btoa(unescape(encodeURIComponent(content))),
    });
  }

  // 3. Build email
  const filesList = attachments
    .map((a) => `<li><code>${a.filename}</code></li>`)
    .join("");

  const html = `
<h2>Your ManagerPack bundle: ${bundleName}</h2>
<p>Thanks for your purchase! Your skill files are attached.</p>
<h3>What's included</h3>
<ul>${filesList}</ul>
<h3>How to use</h3>
<ol>
  <li>Save the attached <code>.md</code> files</li>
  <li>Drop them into your project's <code>.claude/skills/${bundleName}/</code> directory</li>
  <li>Your AI agent will pick them up immediately</li>
</ol>
<p>Questions? Reply to this email.</p>
<p>&mdash; ManagerPack</p>
`;

  // 4. Send via Resend
  const sendResp = await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.RESEND_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      from: "ManagerPack <delivery@iybi-twc.com>",
      to: email,
      subject: `Your ManagerPack bundle: ${bundleName}`,
      html,
      attachments,
    }),
  });

  if (!sendResp.ok) {
    const errText = await sendResp.text();
    throw new Error(`Resend failed (${sendResp.status}): ${errText}`);
  }

  // 5. Record delivery receipt
  const receipt = {
    email,
    bundle: bundleName,
    stripe_session_id: sessionId,
    files: attachments.map((a) => a.filename),
    delivered_at: new Date().toISOString(),
  };
  await env.BUCKET.put(
    `${prefix}/deliveries/${Date.now()}-${bundleName}.json`,
    JSON.stringify(receipt, null, 2)
  );
}

// --- Stripe signature verification ---
// (Same HMAC-SHA256 pattern as iybi-twc worker/index.js)

async function verifyStripeSignature(payload, sigHeader, secret) {
  const parts = {};
  for (const pair of sigHeader.split(",")) {
    const [k, v] = pair.split("=");
    parts[k.trim()] = v;
  }
  const timestamp = parts["t"];
  const signature = parts["v1"];

  if (!timestamp || !signature) return false;

  const signedPayload = `${timestamp}.${payload}`;
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const mac = await crypto.subtle.sign(
    "HMAC",
    key,
    new TextEncoder().encode(signedPayload)
  );
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
