#!/usr/bin/env bash
# Deploy the WhatsApp chatbot to a remote host (EC2) over SSH.
#
#   ./scripts/deploy.sh 16.171.9.191
#   ./scripts/deploy.sh --bootstrap -i ~/.aws/eilaf-chatbot.pem 16.171.9.191
#
# What it does, in order:
#   1. pre-flight the things that silently break a deploy (missing secret,
#      missing corpus) *locally*, before touching the remote;
#   2. rsync the working tree + the two things git does not carry — the
#      repo-root .env and data/static/ (corpus + embedding cache);
#   3. rebuild and restart the compose stack with EILAF_ENV=prod;
#   4. print the tunnel's public URL for the Meta webhook config.
#
# The code is rsync'd rather than pulled with git, deliberately: the instance
# needs no GitHub credentials, and what ships is exactly the tree you are
# looking at. That cuts both ways — a dirty tree deploys dirty, so it warns.
#
# Re-running is the normal way to update; only the changed bytes go over the
# wire, and the hf-cache volume keeps the e5 weights across restarts.
set -euo pipefail

SSH_USER="ubuntu"
SSH_KEY=""
REMOTE_DIR="eilaf_volunteer_chatbot"
BOOTSTRAP=0
BUILD="--build"
SKIP_CORPUS=0
CHECK_TOKEN=1

usage() {
    # The header block above, minus the shebang and the `set -e` line — so
    # editing the header can't silently desync the help text from it.
    awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "$0"
    cat <<'USAGE'

Options:
  -i, --key PATH     SSH private key (e.g. ~/.aws/eilaf-chatbot.pem)
  -u, --user USER    SSH user (default: ubuntu; use ec2-user on Amazon Linux)
  -d, --dir PATH     remote directory, relative to $HOME (default: eilaf_volunteer_chatbot)
      --bootstrap    install docker/compose/git/rsync first (fresh host, once)
      --no-build     restart without rebuilding the image (config-only changes)
      --skip-corpus  don't sync data/static (it is ~47 MB and rarely changes)
      --no-token-check  skip the live Graph API probe (offline deploys)
  -h, --help         this message
USAGE
    exit "${1:-0}"
}

die() { echo "deploy: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        -i|--key)      SSH_KEY="$2"; shift 2 ;;
        -u|--user)     SSH_USER="$2"; shift 2 ;;
        -d|--dir)      REMOTE_DIR="$2"; shift 2 ;;
        --bootstrap)   BOOTSTRAP=1; shift ;;
        --no-build)    BUILD=""; shift ;;
        --skip-corpus) SKIP_CORPUS=1; shift ;;
        --no-token-check) CHECK_TOKEN=0; shift ;;
        -h|--help)     usage 0 ;;
        -*)            die "unknown option $1 (try --help)" ;;
        *)             [ -n "${HOST:-}" ] && die "more than one host given"
                       HOST="$1"; shift ;;
    esac
done

[ -n "${HOST:-}" ] || usage 1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)
[ -n "$SSH_KEY" ] && SSH_OPTS+=(-i "$SSH_KEY")
TARGET="${SSH_USER}@${HOST}"
remote() { ssh "${SSH_OPTS[@]}" "$TARGET" "$@"; }

# --- 1. pre-flight ---------------------------------------------------------
# Every check here is for something that fails *late* and confusingly: a
# container that crash-loops on a missing secret, or a bot that answers from an
# empty corpus because .gitignore keeps data/static out of git.

echo "deploy: pre-flight"

[ -f .env ] || die ".env not found at the repo root — see app/whatsapp/SETUP.md §1.6"

for key in WHATSAPP_ACCESS_TOKEN WHATSAPP_PHONE_NUMBER_ID WHATSAPP_VERIFY_TOKEN WHATSAPP_APP_SECRET; do
    grep -qE "^${key}=.+" .env || die "$key is missing or empty in .env"
done
# EILAF_ENV=prod (set at launch below) makes the App Secret mandatory at import;
# catching it here beats watching the container crash-loop on the far end.

# A *present* token is not a working one, and a dead one fails silently: Meta
# keeps delivering webhooks and still reports a healthy callback URL while every
# reply fails. Ask Meta before shipping the credential, not after.
if [ "$CHECK_TOKEN" -eq 1 ]; then
    (
        set -a; . ./.env; set +a
        base="https://graph.facebook.com/${GRAPH_API_VERSION:-v21.0}"
        body=$(curl -s --max-time 20 "$base/$WHATSAPP_PHONE_NUMBER_ID" \
                    -H "Authorization: Bearer $WHATSAPP_ACCESS_TOKEN") || {
            echo "deploy: WARNING — could not reach the Graph API; token unverified" >&2
            exit 0
        }
        if ! printf '%s' "$body" | grep -q '"id"'; then
            echo "deploy: WHATSAPP_ACCESS_TOKEN was rejected by Meta:" >&2
            printf '%s\n' "$body" | head -c 400 >&2; echo >&2
            echo "deploy: refusing to deploy a token that cannot send." >&2
            exit 1
        fi
        # Expiry needs an app access token, so it is best-effort: no
        # WHATSAPP_APP_ID means we still know the token works, just not for
        # how long.
        if [ -n "${WHATSAPP_APP_ID:-}" ]; then
            exp=$(curl -s --max-time 20 "$base/debug_token" \
                       --get --data-urlencode "input_token=$WHATSAPP_ACCESS_TOKEN" \
                       --data-urlencode "access_token=$WHATSAPP_APP_ID|$WHATSAPP_APP_SECRET" \
                  | python3 -c "import json,sys; print(json.load(sys.stdin).get('data',{}).get('expires_at',''))" 2>/dev/null)
            if [ -n "$exp" ] && [ "$exp" != "0" ]; then
                days=$(( (exp - $(date +%s)) / 86400 ))
                if [ "$days" -lt 14 ]; then
                    echo "deploy: WARNING — access token expires in $days days; rotate before it does" >&2
                else
                    echo "deploy:   token ok (expires in $days days)"
                fi
            fi
        else
            echo "deploy:   token ok (set WHATSAPP_APP_ID in .env to see its expiry)"
        fi
    ) || exit 1
fi

if [ "$SKIP_CORPUS" -eq 0 ]; then
    [ -d data/static/enriched ] || die "data/static/enriched/ is missing — nothing to answer from"
    enriched=$(find data/static/enriched -name '*.json' | wc -l)
    [ "$enriched" -gt 0 ] || die "data/static/enriched/ has no .json — build the corpus first (scripts/curate_data.sh)"
    if [ ! -f data/static/.embeddings/embeddings.npy ]; then
        echo "deploy: WARNING — no embedding cache; the first query will re-encode" >&2
        echo "deploy:           the whole corpus on the instance's CPU (very slow)." >&2
    fi
    echo "deploy:   corpus ok ($enriched enriched documents)"
fi

if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    echo "deploy: WARNING — working tree is dirty; deploying it as-is:" >&2
    git status --short | sed 's/^/deploy:           /' >&2
fi

echo "deploy:   branch $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"

# --- 2. bootstrap (fresh host only) ----------------------------------------
if [ "$BOOTSTRAP" -eq 1 ]; then
    echo "deploy: bootstrapping $TARGET"
    remote 'bash -s' <<'BOOTSTRAP_EOF'
set -euo pipefail
if command -v apt-get >/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq docker.io docker-compose-v2 rsync
else
    sudo dnf install -y -q docker rsync
    # Amazon Linux ships neither compose v2 nor buildx as packages, and compose
    # refuses to build without buildx ("requires buildx 0.17.0 or later"), so
    # both CLI plugins are installed by hand.
    mkdir -p ~/.docker/cli-plugins
    curl -sSL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m)" \
        -o ~/.docker/cli-plugins/docker-compose
    chmod +x ~/.docker/cli-plugins/docker-compose

    # buildx release assets carry the version in the filename, so /latest/download
    # can't be used directly — ask the API, and fall back to a pin if that fails
    # (rate limit, no network to api.github.com).
    case "$(uname -m)" in
        x86_64)  arch=amd64 ;;
        aarch64) arch=arm64 ;;
        *)       arch="$(uname -m)" ;;
    esac
    bx=$(curl -sSL https://api.github.com/repos/docker/buildx/releases/latest \
         | grep -o "\"browser_download_url\": *\"[^\"]*linux-${arch}\"" \
         | head -1 | cut -d'"' -f4)
    [ -n "$bx" ] || bx="https://github.com/docker/buildx/releases/download/v0.17.1/buildx-v0.17.1.linux-${arch}"
    curl -sSL "$bx" -o ~/.docker/cli-plugins/docker-buildx
    chmod +x ~/.docker/cli-plugins/docker-buildx
fi
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"

# Swap is margin, not a substitute for RAM: the e5-large weights alone are
# ~2.1 GB resident, so this only covers build spikes on a 4 GB instance.
if ! swapon --show | grep -q .; then
    # No -q on mkswap: util-linux on Amazon Linux 2023 doesn't have it, and with
    # `set -e` the unknown flag aborts the whole bootstrap.
    sudo rm -f /swapfile                       # re-runnable after a failed attempt
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile >/dev/null
    sudo swapon /swapfile
    grep -q '^/swapfile ' /etc/fstab \
        || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi
BOOTSTRAP_EOF
    echo "deploy:   bootstrap done (docker group membership needs a new session —"
    echo "deploy:   this script opens one per step, so it takes effect already)"
fi

# --- 3. sync ---------------------------------------------------------------
# rsync carries .env and data/static/ even though both are gitignored: that is
# the whole point — they are exactly what a `git clone` on the box would miss.

echo "deploy: syncing to $TARGET:~/$REMOTE_DIR"
remote "mkdir -p ~/$REMOTE_DIR"

RSYNC_EXCLUDES=(
    --exclude '.git/'
    --exclude '__pycache__/'
    --exclude '*.py[cod]'
    --exclude '*.egg-info/'
    --exclude '.pytest_cache/'
    --exclude '.ruff_cache/'
    --exclude '.mypy_cache/'
    --exclude '.venv/'
    --exclude '.claude/'
    --exclude '.agents/'
    --exclude '.idea/'
    --exclude '.vscode/'
    # Per-machine compose tweaks. .gitignore keeps these out of git, which says
    # nothing to rsync — and shipping one is actively harmful: the dev override
    # mounts a local HF cache and sets HF_HUB_OFFLINE=1, which on a remote host
    # means an empty cache the container is forbidden to populate, so it
    # crash-loops on a model it could have downloaded in a minute.
    --exclude 'docker-compose.override.yml'
)

RSH="ssh ${SSH_OPTS[*]}"

# Code: --delete, so a file removed here is removed there (a stale module left
# behind can otherwise shadow the real one). data/static is excluded from this
# pass and handled separately below.
rsync -az --delete-after --info=stats1 \
    "${RSYNC_EXCLUDES[@]}" --exclude 'data/static/' \
    -e "$RSH" \
    ./ "$TARGET:~/$REMOTE_DIR/"

# Corpus: additive, deliberately NO --delete. Boot-time curation
# (SKIP_DATA_CURATION=0) writes into this directory on the instance, and those
# files legitimately have no local counterpart — deleting them would throw away
# work the box did. Stale entries there are rare and cheap; lost ones are not.
if [ "$SKIP_CORPUS" -eq 0 ]; then
    rsync -az --info=stats1 \
        -e "$RSH" \
        data/static/ "$TARGET:~/$REMOTE_DIR/data/static/"
fi

remote "chmod 600 ~/$REMOTE_DIR/.env"
# Excluding the override above stops us *sending* one, but rsync also protects
# excluded paths from --delete, so a copy shipped by an earlier version of this
# script would linger forever. A remote host must never have one.
remote "rm -f ~/$REMOTE_DIR/app/whatsapp/docker-compose.override.yml"

# --- 4. build + launch -----------------------------------------------------
# EILAF_ENV=prod: an internet-reachable host must validate the App-Secret HMAC,
# so config.py refuses to import without WHATSAPP_APP_SECRET. --force-recreate
# because a plain restart reuses the old container's environment, which would
# silently keep a rotated secret's predecessor.

echo "deploy: building and starting the stack"
remote "cd ~/$REMOTE_DIR/app/whatsapp && EILAF_ENV=prod docker compose up -d $BUILD --force-recreate"

# --- 5. verify + report the webhook URL ------------------------------------
echo "deploy: waiting for the server"
for _ in $(seq 1 60); do
    if remote "cd ~/$REMOTE_DIR/app/whatsapp && docker compose ps --status running --quiet chatbot" | grep -q .; then
        break
    fi
    sleep 2
done

if ! remote "cd ~/$REMOTE_DIR/app/whatsapp && docker compose ps --status running --quiet chatbot" | grep -q .; then
    echo "deploy: chatbot is not running — last 40 log lines:" >&2
    remote "cd ~/$REMOTE_DIR/app/whatsapp && docker compose logs --tail=40 chatbot" >&2
    exit 1
fi

# The tunnel prints its hostname a few seconds after the container starts; a
# named tunnel (TUNNEL_TOKEN) prints nothing here because its URL is fixed.
URL=""
for _ in $(seq 1 30); do
    URL=$(remote "cd ~/$REMOTE_DIR/app/whatsapp && docker compose logs tunnel 2>/dev/null" \
          | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' | tail -1 || true)
    [ -n "$URL" ] && break
    sleep 2
done

echo
echo "deploy: done."
if [ -n "$URL" ]; then
    cat <<EOF
deploy: webhook URL — put this in Meta → App → WhatsApp → Configuration:
deploy:
deploy:     $URL/webhook
deploy:
deploy: The trycloudflare hostname changes on every recreate. For a URL that
deploy: survives a redeploy, switch to a named tunnel (TUNNEL_TOKEN) — see the
deploy: commented block in app/whatsapp/docker-compose.yml.
EOF
else
    echo "deploy: no trycloudflare URL in the tunnel logs — expected if you run a"
    echo "deploy: named tunnel; otherwise check: docker compose logs tunnel"
fi
echo
echo "deploy: logs —  ssh ${SSH_KEY:+-i $SSH_KEY }$TARGET 'cd $REMOTE_DIR/app/whatsapp && docker compose logs -f chatbot'"
