# eilaf_volunteer_chatbot

RAG-based volunteer chatbot for Eilaf, with scraping, embedding, and
speech-to-text units.

## Install / create the env

The project is installable and locked with [uv](https://docs.astral.sh/uv/).
Create the environment from `uv.lock` with a single command.

The env is placed under `$ENVS_DIR` if that variable is set, otherwise in the
project directory (`./eilaf`):

```bash
UV_PROJECT_ENVIRONMENT="${ENVS_DIR:-.}/eilaf" uv sync --all-extras
```

`uv` reads the venv location from the `UV_PROJECT_ENVIRONMENT` environment
variable (it can't be a conditional path in `pyproject.toml`), and
`${ENVS_DIR:-.}` expands to `$ENVS_DIR` when set and `.` otherwise.

**`--all-extras` matters on a dev machine.** The `stt` extra (faster-whisper,
for `transcription/`) is kept out of the WhatsApp image; a bare `uv sync` gives
you the scraping, retrieval and WhatsApp stack without it.

> **GPU note:** the lock pins the default PyPI `torch`. On a CUDA machine,
> add the appropriate `[tool.uv.sources]` / index for the `cuXXX` wheels
> before syncing so you don't overwrite an existing GPU stack.

## Build the corpus: scraping → enrichment → embedding

The RAG pipeline reads `data/static/enriched/`. That directory is produced by
the stages below, run in order (all re-runnable and incremental) — or by
`scripts/curate_data.sh`, which chains them:

```text
web / .docx  ──scrape──▶  data/static/raw/*.json  ──enrich──▶  data/static/enriched/*.json
                                                                  │
                                                            embed ▼
                                                  data/static/enriched/.embeddings/
```

> **Run this before the app — a fresh clone has no corpus.** The scraped and
> enriched documents are deliberately kept out of git (`.gitignore` excludes
> `data/static/**/*.json` and friends), so `data/static/raw/` and
> `data/static/enriched/` start out empty. The app still *starts* on an empty
> corpus — it fails quietly, returning blank answers to every question rather
> than erroring — so if the bot comes up but answers nothing, that's the cause.
> The `.docx` sources are local/private files, so a clone can only rebuild the
> web-scraped part of the corpus.

Run everything from the repo root, with the env active (or prefixed with
`uv run`).

### 1. Scrape

Each scraper module has a `__main__` block holding the source URLs it was last
run with; edit that list to add sources.

```bash
python -m data.scraping.nevo         # Israeli law texts (modern + legacy templates)
python -m data.scraping.kol_zchut    # KolZchut rights pages, via the MediaWiki API
python -m data.scraping.gov_il       # gov.il guides and information pages, via its content API
python -m data.scraping.docx_source  # local .docx drafts → the same section-tree JSON
```

Output is one JSON section tree per source in `data/static/raw/`, named after
the URL. `docx_source` globs a local folder — point it at your own path before
running.

> **gov.il:** the site is a single-page app behind Cloudflare bot management —
> its HTML is an empty shell and the request is refused on its TLS fingerprint,
> so there is nothing to scrape. `gov_il.py` instead calls the content API the
> site's own front end uses, on a host that is not behind Cloudflare. Requests
> need both an `x-client-id` and an `Origin: https://www.gov.il` header; without
> `Origin` the gateway answers `500 RF-OriginError`, which looks like a server
> fault rather than a rejected request. gov.il publishes no sitemap, so
> `list_pages()` in that module discovers page URLs from a topic or office
> landing page.

> **Time:** one HTTP request per page, so this is fast — measured **0.4–1.2 s
> per page** (the largest legacy nevo law is the slow end), i.e. a few seconds
> for the whole current source list. Adding many pages scales linearly.

### 2. Enrich

Every text segment gets a `useful` flag plus age/gender/track tags, via an LLM.
Re-runs reuse the previous enriched copy and reject obvious junk by rule, so
only new or changed segments are ever billed.

`track` marks content belonging to a separate compensation track — hostile
acts, military bereavement, road or work accidents — which Israeli law routes
through its own authorities and eligibility tests. That material reads like
ordinary victim support, so without the tag retrieval offers it for the
neighbourhood-violence questions this bot exists to answer, and the entitlements
do not carry over. It is the one axis that penalises on a *missing* fact: a
tagged chunk is off-track until a user fact affirms it. A source devoted
entirely to one track can also stamp `track` on its tree root (see
`_HOSTILE_ACTS` in `data/scraping/gov_il.py`), which the loader inherits down to
every segment — per-segment inference cannot catch a lone sentence like "the
allowance is paid monthly", but the page it came from knows.

```bash
python -m data.enrich --dry-run                    # count what would be sent; calls nothing
python -m data.enrich --no-llm                     # rules-only pass, no API key needed
python -m data.enrich --provider claude --limit 500  # budgeted LLM run
```

`--provider` picks the back-end: `claude` (default, Message Batches API),
`gemini` (Google AI free tier), or `cerebras`. Neither back-end has a spend
cap, hence `--dry-run` / `--limit N` — always dry-run first.

`gemini` is the working free back-end. `cerebras` was the bulk option on its
free tier, but as of 2026-08-18 the key returns `402 payment_required` on every
call, so it needs billing (or a swap to Groq/OpenRouter — a one-line change in
`openai_enricher.py`) before it can carry the ~10k nevo backlog.

A segment the back-end fails on is left un-annotated and stays pending, so a
later run retries it; the run ends with a loud `WARNING: N segments failed` if
any did. Watch for that — a run can otherwise exit 0 having answered nothing,
which is exactly what a retired model name does. Google retires Gemini models
often, so if `--provider gemini` starts reporting model errors, check
`ROTATION_MODELS` in `data/enrich/gemini_enricher.py` against
`client.models.list()`.

> **Time:** `--dry-run` and `--no-llm` are seconds (pure local work). The LLM
> pass is the long one: the current raw corpus has **~10,000 segments still
> pending**. `claude` submits them as a batch and polls every 30 s — typically
> minutes to a couple of hours (Anthropic's batch SLA is 24 h). `gemini` and
> `cerebras` send **one sequential request per segment** (≈1 s each), so a full
> 10k pass is hours, and Gemini's free-tier daily quotas will spread it over
> days. Budget it with `--limit` and run it repeatedly; nothing is lost between
> runs.

### 3. Embed

The corpus is embedded once and cached on disk
(`data/static/enriched/.embeddings/`), fingerprinted by corpus *and* embedder.
The pipeline does this lazily on its first query — which is no good for a
webhook, where the first user would wait out the whole encode. Warm it
ahead of time:

```bash
python -m chatbot.rag.pipeline.warm_cache
```

A no-op (a sub-second load) when neither the corpus nor the embedder changed, so
it is safe to run on every boot; a full re-encode — around an hour on CPU — when
either did.

### All three stages at once

`scripts/curate_data.sh` chains scrape → enrich → embed in the only order they
work in. Everything is incremental, so re-running it is cheap once the corpus
exists:

```bash
./scripts/curate_data.sh
```

Stages are non-fatal by default: a refused scrape or an exhausted LLM quota is
reported and the run continues, so the embed stage still publishes whatever
corpus exists. The script exits non-zero if anything failed. Environment
variables tune it:

| Variable | Default | What it does |
| --- | --- | --- |
| `CURATE_SKIP_SCRAPE` / `_ENRICH` / `_EMBED` | `0` | `1` skips that one stage |
| `CURATE_SCRAPERS` | `nevo kol_zchut gov_il` | which `data.scraping` modules to run (`docx_source` is out: it globs a private local folder) |
| `ENRICH_PROVIDER` | `gemini` | `claude` / `gemini` / `cerebras` |
| `ENRICH_LIMIT` | *(unset)* | cap segments sent to the LLM this run; unset means the whole pending backlog, which can take hours |
| `ENRICH_ARGS` | *(unset)* | extra flags for `python -m data.enrich`, e.g. `--no-llm` |
| `CURATE_STRICT` | `0` | `1` aborts on the first stage failure |
| `PYTHON` | `python` | interpreter to run the stages with |

### Where the corpus lives: `EILAF_STATIC_DIR`

Everything above hangs off one root, `data/static/` by default. Set
`EILAF_STATIC_DIR` to move the whole store — `raw/`, `enriched/` and
`.embeddings/` travel together, and scrapers, enrichment, the bot and the cache
all follow it:

```bash
EILAF_STATIC_DIR=/mnt/eilaf-static ./scripts/curate_data.sh
```

Read **once, at import**, so it has to be a real environment variable (docker
compose's `env_file` makes one; a value set from Python later will not be seen).
An empty value falls back to the repo-local default.

For a deployment (EC2, S3) this is the hook to use, but note it is a
*filesystem* path — the corpus store does not speak the S3 API. (One thing now
does: `app/whatsapp/reporting` writes reported conversations to a bucket with
boto3. The two are unrelated — a report never lands in the corpus.) Point it at S3 through a
mount (`mountpoint-s3`, `s3fs`) or sync the bucket to local disk before start
and point it at that copy. A synced local copy is the safer of the two: the
embedding cache is a 20 MB `.npy` the pipeline memory-maps at boot and the
loader re-reads every JSON in the corpus, so per-object latency on a FUSE mount
lands directly on startup time.

The WhatsApp container does the sync for you: set `EILAF_CORPUS_S3_URI` and it
pulls the store with `python -m data.corpus_sync` before starting, downloading
only files whose content differs (see `app/whatsapp/SETUP.md`).

## Answer-engine keys

The answer heads read their keys from the repo-root `.env` (gitignored):

```bash
GEMINI_API_KEY=...     # gemini engine (the WhatsApp bot's default)
CEREBRAS_API_KEY=...   # cerebras engine
ANTHROPIC_API_KEY=...  # claude engine
```

Keys are resolved lazily, per engine — you only need the one for the engine
you actually use.

The `bedrock` engine is the exception: it runs the same Claude model as
`claude`, but through Amazon Bedrock on the project's own AWS account, so it
takes **no API key**. It needs a region instead, and AWS credentials from the
ambient chain (`AWS_PROFILE` locally, the EC2 instance role in the deployment —
the same mechanism the report store uses):

```bash
EILAF_BEDROCK_REGION=eu-north-1   # required; deliberately has no default
EILAF_BEDROCK_MODEL=...           # optional: pin the inference-profile id
```

Unset, the engine stays inert and the others are unaffected. The region has no
default on purpose — botocore would silently fall back to `us-east-1` and
generate answers outside the EU. See `app/whatsapp/SETUP.md` for the IAM policy
and the model-access step, both of which are needed before the engine will
answer.


## Track AWS spend

`scripts/aws_costs.py` prints the **actual cost** of the infrastructure per
service — the current month, the running total to date, and the change since the
last time you ran it — and shouts if the month is over budget.

```bash
./scripts/aws_costs.py                     # table + append to the CSV
./scripts/aws_costs.py --threshold 25      # tighter budget for this run
./scripts/aws_costs.py --month 2026-08     # look back at a closed month
./scripts/aws_costs.py --since 2024-01     # widen the to-date window
./scripts/aws_costs.py --with-credits      # what is billed today instead
./scripts/aws_costs.py --all --no-write    # zero-cost services too, look only
```

```
AWS spend by service (UnblendedCost, USD -- actual cost, credits excluded; whole account)
                                        2026-10 month-to-date      since 2025-10-01
Service                                    chatbot      legacy     chatbot      legacy     change
-------------------------------------------------------------------------------------------------
Amazon Elastic Compute Cloud - Compute       10.00           -       40.00      120.00       0.00
Amazon Virtual Private Cloud                  1.00           -        4.00       12.00       0.00
EC2 - Other                                   1.00           -        4.00       10.00       0.00
AWS Budgets                                      -        0.10           -        1.00       0.00

nothing in 2026-10 -- to date only         chatbot      legacy   $20.00 in total
  Amazon Registrar                               -       16.00
  Tax                                            -        3.00
  Amazon Simple Storage Service                  -        1.00
-------------------------------------------------------------------------------------------------
TOTAL                                        12.00        0.10       48.00      163.00       0.00

Both halves together: 2026-10 $12.10, since 2025-10-01 $211.00.
```

*(Figures above are illustrative. Real ones live in the git-ignored CSV.)*


**Every run splits the cost in two**, so each service is one row rather than the
same name appearing in two tables for the reader to join. `chatbot` is
`eu-north-1` from `2026-09`; `legacy` is everything else in scope.

Services that **stopped billing** move to a second block with only the two
columns that can say anything about them. Most of this account's services are
dormant history, so keeping them in the main table meant four money columns that
were mostly blank. A hard zero prints as `-` for the same reason: a column of
`0.00`s competes with the figures that matter. `--all` adds the services that
have never cost anything at all.


**Every run splits the cost in two**, so each service is one row rather than the
same name appearing in two tables for the reader to join. `chatbot` is
`eu-north-1` from `2026-09`; `legacy` is everything else in scope. A hard zero
prints as `-`, because a column of `0.00`s competes with the figures that matter.


**Credits and refunds are excluded**, which on this account is the difference
between a useful number and a useless one: it runs on a promotional credit
balance, so every `Usage` line is cancelled by an equal `Credit` line and the
netted total reads $0.00 while the infrastructure still costs real money. A
budget alarm on the netted figure would be unreachable until the credits ran dry
— exactly when it is needed. `--with-credits` asks the other question, what the
card is charged today; being a look rather than a reading, it records nothing.

Dropping credits is also what makes a service-grouped table honest. AWS files a
credit against a *different service* than the charge it cancels — here the
transfer usage books under `EC2 - Other` and its credit under `AWS Data
Transfer` — so a netted table shows one service as purely positive and another
as purely negative, neither of which is a thing that happened.

The exclusion works by **dropping record types** (`Credit`, `Refund`) rather than
allowlisting `Usage` and `Tax`: AWS books real charges under several other types
too (`DiscountedUsage`, `SavingsPlanCoveredUsage`, `Support`), and an allowlist
would discard them silently — the one failure mode a cost guard must not have.
`Discount` records stay for the same reason: a negotiated rate is part of the
real price, where a promotional credit is a balance draining.

Everything comes from **one** Cost Explorer request — monthly granularity over a
wide window, so the current month is the last bucket and the to-date total is the
sum of them all; grouped by `RECORD_TYPE` as well as by service, which is what
separates the credit records from the charges. Neither the extra column nor the
second group-by costs anything, which matters because the request is the
expensive part. It does page when the two group-bys produce more groups than one
response holds, and each extra page is another $0.01 — a truncated answer being
worse than a dearer one.

"To date" means the queried window, not the account's whole life: Cost Explorer
serves **12 months** by default, which is what this asks for. Widen it with
`--since` if the account is older *and* has historical data enabled.

Each run appends a snapshot to `data/costs/aws_costs.csv` (git-ignored — see
`.gitignore`) as
`snapshot,scope,month,service,chatbot_month_usd,legacy_month_usd,`
`chatbot_to_date_usd,legacy_to_date_usd,delta_usd`. That
file *is* the "change since" column, and the diff is taken on the **to-date**
figure rather than the month's — a to-date total only ever grows, so the delta
stays meaningful across a month boundary instead of resetting to a misleading
negative on the 1st. A file written by an older column layout is migrated in
place on the next write, with the actual-cost cells left **blank** on its rows
rather than back-filled: those figures were netted against credits, and copying
them across would read as "no credits that month", inventing history. The first
run after such a migration therefore shows `-` for the change rather than a
jump the size of the credit balance.

The budget defaults to **$40**, overridable per-run with `--threshold` or
per-machine with `EILAF_COST_ALERT_USD`. Over it, the script prints a banner to
stderr and exits **2** (1 = the query failed, 0 = under budget), so a cron job or
`&&` chain can act on it. The alert fires on the **month**, not on the to-date
total, deliberately: a lifetime figure crosses any fixed line exactly once and
then screams forever, whereas a monthly budget re-arms itself. It watches the
figure the table shows — the actual cost — so it cannot sit silent behind a
credit balance.

### How the chatbot is told apart from everything else

This AWS account carries workloads older than this project, and they dominate
the history — most of the twelve-month total is theirs. The split is a
**region-and-month rule** (`CHATBOT_REGION`, `CHATBOT_SINCE` at the top of the script), which is
exact here by luck of timing: the chatbot's instance launched 2026-09-12 into
`eu-north-1`, and the earlier workload in that region had already stopped — by
the month before the launch it had dwindled to pennies.

The rule is a heuristic, and it is worth knowing where it frays: `global` and
`NoRegion` costs (a domain, Route 53, tax) always land in `legacy` whether or not
they serve this project, and a stray $0.01 of RDS in `eu-north-1` falls inside
the window without being the chatbot's.

For an exact split, tag the resources and filter on the tag:

1. Tag the chatbot's resources — instance, its EBS volume, the reports bucket —
   with something like `Project=eilaf-chatbot`.
2. Activate `Project` in **Billing > Cost allocation tags** (takes up to 24 h).
3. `./scripts/aws_costs.py --tag Project=eilaf-chatbot`

**Tags are not retroactive**: they cover only usage incurred after activation, so
the region-and-month rule stays the way to read the history. An empty tagged
result says so rather than implying the resources were free.

`--region` and `--tag` narrow what is counted at all; the chatbot/legacy split
then applies *within* whatever is left. Each scope keeps its **own** history in
the CSV's `scope` column, and a snapshot is only ever diffed against its own
scope — a narrowed reading measured against a whole-account one would otherwise
look like a huge saving. An explicit `--since` is part of the scope for the same
reason; the default rolling window is not, so the main series stays continuous.

`--month` only chooses which bucket the per-month column reports; the to-date
column always runs to now. A back-dated `--month` prints but records nothing, so
a look backwards can't become the baseline the next run diffs against.

Three caveats, all of them load-bearing:

- **The query costs $0.01.** Cost Explorer bills per `GetCostAndUsage` request
  and this makes one per run — fine daily (~$0.30/month), absurd every five
  minutes ($86/month, more than the budget it guards).
- **Cost Explorer needs a permission the report user lacks by default.** The
  `eilaf` profile in `~/.aws` is `eilaf-dev-reports`, whose policy otherwise
  reaches S3 and only S3 (the `conversations/` prefix of the reports bucket, see
  `app/whatsapp/SETUP.md`). Add a `ce:GetCostAndUsage` statement — the script
  prints it when denied. Two one-time account settings also gate billing data:
  Cost Explorer has to be **enabled** (Billing console; first activation takes up
  to 24 h to backfill), and **IAM access to Billing** has to be on (Account
  settings), without which an IAM user is denied even holding the action.
- **The numbers lag.** Cost Explorer refreshes a few times a day, so the current
  month is always incomplete and is flagged `Estimated`. It is a trend gauge; the
  bill is the bill.

A **negative** net line is normal, not an error: a credit need not post against
the service whose charge it cancels. On this account the always-free 100 GB/month
of outbound transfer shows as an `EC2 - Other` charge and an equal negative `AWS
Data Transfer` credit, which is why net can show per-service activity and still
total nothing. Gross keeps the charge and drops the credit, so it does not cancel
out the same way.

Note `--profile` defaults to `eilaf` (else `$AWS_PROFILE`) because `~/.aws` here
holds no default profile. Cost Explorer is queried in `us-east-1` regardless of
`EILAF_REPORT_REGION` — it is a global service with one endpoint.
