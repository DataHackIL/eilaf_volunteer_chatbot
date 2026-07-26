# WhatsApp chatbot — setup & run guide

This is the WhatsApp front-end for the Eilaf RAG pipeline. It runs a webhook
server that mirrors the Streamlit app as a multi-turn WhatsApp chat:

> language → question → violent-event description → gender → age → answer

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
  it unset disables the check (dev only) — set it for anything real.
- **Sender not a test recipient** — in test mode add the number under
  *API Setup → recipient phone number* first.
