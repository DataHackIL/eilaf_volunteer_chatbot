# WhatsApp chatbot — setup & run guide

This is the WhatsApp front-end for the Eilaf RAG pipeline. It runs a webhook
server that mirrors the Streamlit app as a multi-turn WhatsApp chat:

> question → answer (language auto-detected; event details optional, on request)

The language is guessed from the script of the user's first message (Hebrew,
Arabic, or English; Hebrew when there is no signal), so there is no language menu
to get through — the opening message is kept and prepended to the question. A
wrong guess is corrected with the two buttons on the welcome message, or by
typing `שפה` / `لغة` / `language` at any point.

A question is answered on the turn it arrives. Every answer carries an **Add
details** button: tapping it asks for a few words about the event and re-answers
the same question with that as the retrieval anchor. Setting
`WHATSAPP_COLLECT_FACTS=1` restores the older flow, which asked for the event
description, gender and age *before* answering — off by default because the
corpus carries almost no age/gender/locality tags, so those answers changed
nothing and only delayed the reply.

This guide takes you from a Meta developer account to a working bot on your own
machine, exposed to Meta through a tunnel. No paid hosting required.

---

## 0. How it fits together (read this first)

Your server never talks to the user's phone directly — everything goes through
Meta, in **two directions**:

```
  ① inbound   user's WhatsApp → Meta servers → HTTP POST → YOUR server (webhook)
  ② outbound  YOUR server → HTTP POST (Cloud API) → Meta servers → user's WhatsApp
```

Outbound (②) is a normal call from your machine and needs nothing special.
Inbound (①) is the catch: Meta must **call into your machine**, but a laptop on
home/office wifi has no public address. A **tunnel** (Cloudflare) gives your
local server a public HTTPS URL so Meta can reach it. The Cloud API is
webhook-only — there is no "poll for new messages" alternative.

---

## 1. Create the Meta app and get the credentials

You need four values, all injected via the repo-root `.env` (never committed):

| `.env` key                   | What it is                                  |
|------------------------------|---------------------------------------------|
| `WHATSAPP_ACCESS_TOKEN`      | The API token (the "API key")               |
| `WHATSAPP_PHONE_NUMBER_ID`   | The sending phone number's id               |
| `WHATSAPP_VERIFY_TOKEN`      | A string **you invent** (webhook handshake) |
| `WHATSAPP_APP_SECRET`        | App Secret (validates incoming webhooks)    |
| `WHATSAPP_APP_ID`            | App ID — *optional*, see below              |
| `WHATSAPP_COLLECT_FACTS`     | `1` restores the pre-answer fact questions  |

`WHATSAPP_APP_ID` is not a secret and nothing needs it to run. Set it and the
boot-time check and `scripts/deploy.sh` can report **how long the access token
has left** (it is the `<id>|<secret>` app access token that Meta's `debug_token`
requires); leave it unset and they can still tell you whether the token works,
just not for how long.

### 1.1 Create the app
1. Go to <https://developers.facebook.com> → **My Apps** → **Create App**.
2. Choose app type **Business**. Name it, finish creation.
3. On the app dashboard, find **WhatsApp** and click **Set up**. This creates a
   test **WhatsApp Business Account**, a **test phone number**, and a temporary
   access token.

### 1.2 Phone Number ID + a first token
On **WhatsApp → API Setup** (or *Getting Started*):
- Copy **Phone number ID** → `WHATSAPP_PHONE_NUMBER_ID`.
- Copy the **Temporary access token** (valid ~24h) → `WHATSAPP_ACCESS_TOKEN`.
  Fine for first tests; §1.4 makes it long-lived.

### 1.3 App Secret
**App → Settings → Basic → App Secret → Show** → `WHATSAPP_APP_SECRET`.
Used to validate the `X-Hub-Signature-256` header so only Meta can post to your
webhook.

### 1.4 A long-lived access token (do this once the temp token expires)
The temp token dies in 24h. For something that keeps working:
1. **business.facebook.com → Business Settings → Users → System Users**.
2. **Add** a system user (role *Admin* or *Employee*).
3. **Add Assets** → assign your app with full control.
4. **Generate new token** → pick the app → select scopes
   **`whatsapp_business_messaging`** and **`whatsapp_business_management`** →
   generate. Copy it → `WHATSAPP_ACCESS_TOKEN`. (System-user tokens can be made
   non-expiring.)

### 1.5 Add a test recipient
In **test mode you can only message numbers you register.** On
**WhatsApp → API Setup**, under *To*, **Add recipient phone number** and confirm
the code sent to that phone. Use that phone to chat with the bot.

> Going public (messaging anyone) later requires adding a **real** business
> phone number and completing Meta **business verification** — out of scope for
> this interim setup.

### 1.6 Fill in `.env`
In the repo root `.env` (alongside `GEMINI_API_KEY`):

```dotenv
WHATSAPP_ACCESS_TOKEN=EAAG...           # from 1.2 or 1.4
WHATSAPP_PHONE_NUMBER_ID=123456789012345
WHATSAPP_VERIFY_TOKEN=pick-any-string-here
WHATSAPP_APP_SECRET=your_app_secret
# GEMINI_API_KEY=...                     # already present — the answer engine
```

---

## 2. Start the server + tunnel (Docker, one command)

From the repo root:

```bash
cd app/whatsapp
docker compose up --build
```

This starts two containers:
- **chatbot** — the webhook server on port 8000 (loads the e5 model on first
  boot into a cache volume; later boots are fast).
- **tunnel** — a Cloudflare tunnel giving a public HTTPS URL.

Grab the public URL from the tunnel logs:

```bash
docker compose logs tunnel | grep trycloudflare
# → https://something-random.trycloudflare.com
```

> **Notes**
> - The **first** boot downloads the e5 weights (needs network, slower); after
>   that they're cached in the `hf-cache` volume.
> - Your machine and the stack must stay running to receive messages.
> - The free `trycloudflare.com` URL **changes each run**. For a URL that
>   survives restarts (register the webhook once), create a **named tunnel** in
>   the Cloudflare dashboard and use its `TUNNEL_TOKEN` — see the commented block
>   in `docker-compose.yml`.

### Rebuilding the corpus at boot (`SKIP_DATA_CURATION`)

The container's entrypoint can rebuild the corpus — scrape → enrich → embed, via
`scripts/curate_data.sh` — *before* uvicorn binds, so the webhook is never up
while answering from a corpus that is still being written. It is **off by
default** (`SKIP_DATA_CURATION=1` in `docker-compose.yml`): curation writes to
the bind-mounted `data/static/`, and an image built before a chunking or
embedder change would rewrite the host's corpus and embedding cache with its own
older idea of them. Turn it on deliberately, on a freshly built image:

```bash
SKIP_DATA_CURATION=0 docker compose up --build
```

Set it in the **shell**, as above (or in `app/whatsapp/.env`, which compose
interpolates) — not in the repo-root `.env`, which the `environment:` block
overrides.

Expect a long first boot: the enrichment pass is unbudgeted by default, so cap
it with `ENRICH_LIMIT=500` and run it again later — nothing is lost between
runs. `ENRICH_PROVIDER` (default `gemini`) and the `CURATE_*` variables from the
README's curation table all pass straight through. A failed stage is logged
loudly and the server starts anyway on the existing corpus; `CURATE_STRICT=1`
makes it fatal instead.

`EILAF_STATIC_DIR` moves the whole store off `/app/data/static` (an EC2 volume,
or S3 via a mount) — see the README section of the same name.

### Production mode (`EILAF_ENV`)

`WHATSAPP_APP_SECRET` is what stops anyone but Meta posting to your webhook, and
the check is *skipped* when the secret is empty — fine offline, an open webhook
on a public host. Set `EILAF_ENV=prod` on anything internet-reachable and the
secret becomes mandatory: `app/whatsapp/config.py` raises at import, before
uvicorn binds, so a missing or botched secret crash-loops the container visibly
instead of quietly serving unauthenticated traffic.

```bash
EILAF_ENV=prod docker compose up -d
```

Leave it unset locally — nothing changes and the offline flow keeps working.

> Rotating the secret takes a container **recreate**, not a restart:
> `config.py` reads the environment once at import, and `docker compose restart`
> reuses the old container's environment. Use
> `docker compose up -d --force-recreate chatbot`.

### Getting a stable public URL (the tunnel problem)

The default quick tunnel hands you a random `trycloudflare.com` hostname that
changes **every time the tunnel container restarts** — including an unattended
reboot. This is the single most disruptive thing about the current setup: the
new URL is never registered with Meta, so the bot keeps receiving nothing while
the dashboard still reports a healthy webhook. It happened three times in one
day during the first live run.

**A named Cloudflare tunnel needs a domain.** The stable hostname comes from a
DNS zone you control (`bot.example.org`), so this only works once you or Eilaf
have a domain added to Cloudflare. With one:

1. Cloudflare dashboard → **Zero Trust → Networks → Tunnels → Create a tunnel**
2. Choose **Cloudflared**, name it, copy the **token**
3. Add a **Public hostname**: your subdomain → `http://chatbot:8000`
4. Run with both variables set:

```bash
TUNNEL_TOKEN=<token> TUNNEL_RUN_ARGS=run docker compose up -d
```

Set them together — a token without `TUNNEL_RUN_ARGS=run` silently stays on a
quick tunnel, which is exactly the failure this is meant to end.

**Without a domain**, a named tunnel cannot help, and the options are:

| Option | Stable hostname | Notes |
|---|---|---|
| Register a domain | `bot.yourdomain.org` | ~$10/yr; also unlocks the direct-HTTPS route below |
| ngrok | `<name>.ngrok-free.app` | free tier includes one static domain |
| Tailscale Funnel | `<host>.<tailnet>.ts.net` | free, valid cert, no domain needed |

**Or skip tunnels entirely.** Once the bot lives on EC2 with an Elastic IP and
you have a domain, point an A record at it and terminate TLS on the box (Caddy
with Let's Encrypt, DNS-01 so only 443 is open). Meta requires HTTPS with a
CA-signed certificate, so a bare IP is never enough — but note that a domain is
the prerequisite for *both* paths. If you are going to need one anyway, the
direct route removes a moving part rather than stabilising it.

### When the access token dies (it fails silently)

An expired or under-scoped access token is the nastiest failure mode here,
because nothing looks broken: Meta keeps delivering webhooks, the dashboard
still shows a healthy callback URL, the container stays up — and every reply
fails. You find out from a user, not a log.

Two checks make it loud:

- **At boot** — `server.py` probes the token before loading the model. It logs
  the days remaining, warns under a week, and under `EILAF_ENV=prod` *refuses to
  start* on a token Meta rejects. A network failure counts as "unknown", not
  "dead": a Graph API blip must not keep the bot down.
- **At deploy** — `scripts/deploy.sh` pre-flights the same probe and refuses to
  ship a token that cannot send. `--no-token-check` skips it for offline work.

The dashboard only ever issues 24-hour tokens. Exchange one for 60 days:

```bash
source .env
curl -s "https://graph.facebook.com/v21.0/oauth/access_token?grant_type=fb_exchange_token&client_id=$WHATSAPP_APP_ID&client_secret=$WHATSAPP_APP_SECRET&fb_exchange_token=<24H_TOKEN>"
```

> **Regenerate the token after changing app↔WABA wiring.** Meta snapshots which
> assets a token may write to when it is issued, so a token minted before you
> ran `POST /<waba-id>/subscribed_apps` keeps *reading* fine while every send
> fails with `(#131005) Access denied`. Eliminate that before chasing
> permissions or allow-lists.

### Deploying to a remote host (`scripts/deploy.sh`)

Everything above runs the stack on the machine you are sitting at.
`scripts/deploy.sh` does the same thing on an EC2 box over SSH:

```bash
./scripts/deploy.sh --bootstrap -i ~/.aws/eilaf-chatbot.pem 16.171.9.191   # first time
./scripts/deploy.sh -i ~/.aws/eilaf-chatbot.pem 16.171.9.191               # every time after
```

It pre-flights locally (all four `WHATSAPP_*` keys present, corpus non-empty),
rsyncs the working tree, brings the stack up with `EILAF_ENV=prod`, and prints
the tunnel URL for the Meta webhook config. `--help` lists the rest
(`-u ec2-user` for Amazon Linux, `--no-build`, `--skip-corpus`).

Two things it exists to get right:

- **`data/static/` is not in git.** `.gitignore` excludes the corpus `.json`
  files and `.embeddings/`, so a `git clone` on the instance yields a bot that
  answers from nothing. The script rsyncs the store (~47 MB) alongside the code
  — including the embedding cache, without which the first query re-encodes the
  whole corpus on the instance's CPU.
- **Code ships by rsync, not `git pull`.** The instance needs no GitHub
  credentials, and what runs is exactly your working tree — which also means a
  dirty tree deploys dirty. The script warns and continues.

> **Sizing.** The retrieval core loads `multilingual-e5-large`, whose weights
> are a single 2.1 GB file, so the steady-state footprint is ~3 GB. A 1 GB
> `t3.micro` cannot load the model at all; `t3.medium` (4 GB) is the realistic
> floor, with 30 GB of disk for the image, weights and corpus.

### Running without Docker (alternative)
```bash
python -m app.whatsapp.server          # uvicorn on :8000
cloudflared tunnel --url http://localhost:8000   # separate terminal
```

---

## 3. Register the webhook with Meta

1. **App → WhatsApp → Configuration → Webhook → Edit**.
2. **Callback URL**: your public URL + `/webhook`
   (e.g. `https://something-random.trycloudflare.com/webhook`).
3. **Verify token**: the exact string you put in `WHATSAPP_VERIFY_TOKEN`.
4. Click **Verify and save**. Meta calls `GET /webhook`; the server echoes the
   challenge and the save succeeds.
5. Under **Webhook fields**, **Subscribe** to **`messages`**.

---

## 4. Smoke test

1. **Verification (no Meta needed):**
   ```bash
   curl "http://localhost:8000/webhook?hub.mode=subscribe&hub.verify_token=$WHATSAPP_VERIFY_TOKEN&hub.challenge=123"
   # → 123
   ```
2. **Round trip:** from your registered test phone, send any message to the test
   number. You should get the **language menu** (עברית / العربية / English).
3. Tap a language, type a question, then Skip or fill each detail. You should get
   an answer in the language you chose, plus the sources, then a prompt to ask
   another question.
4. If generation fails (e.g. Gemini free-tier daily quota), you'll see an error
   message but the flow keeps working — retry later or switch the generator in
   `app/whatsapp/pipeline_singleton.py`.

---

## Troubleshooting

- **"Verify and save" fails** — `WHATSAPP_VERIFY_TOKEN` in `.env` must exactly
  match the dashboard value; the tunnel must be up and the URL must end in
  `/webhook`.
- **Messages don't arrive** — confirm you **subscribed to `messages`**, the
  container is running, and the tunnel URL in the dashboard is current (it
  changes on restart unless you use a named tunnel).
- **`WHATSAPP_* is not set`** — the container isn't seeing `.env`; ensure it
  exists at the repo root (compose reads `../../.env`).
- **403 on POST** — signature mismatch: `WHATSAPP_APP_SECRET` is wrong. Leaving
  it unset disables the check (dev only) — see *Production mode* below.
- **Container exits with `WHATSAPP_APP_SECRET is not set`** — working as
  intended: `EILAF_ENV=prod` is set and the secret didn't reach the container.
  Fix the secret rather than clearing `EILAF_ENV`.
- **Sender not a test recipient** — in test mode add the number under
  *API Setup → recipient phone number* first.
