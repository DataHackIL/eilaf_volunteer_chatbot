#!/usr/bin/env python3
"""AWS spend per service: this month so far, and the running total to date.

    ./scripts/aws_costs.py                     # table + append to the CSV
    ./scripts/aws_costs.py --threshold 25      # tighter budget for this run
    ./scripts/aws_costs.py --month 2026-08     # look back at a closed month
    ./scripts/aws_costs.py --since 2024-01     # widen the to-date window
    ./scripts/aws_costs.py --with-credits      # what is billed today instead
    ./scripts/aws_costs.py --all --no-write    # zero-cost services too, look only

Every figure is the **actual cost** of the infrastructure: the ``Credit`` and
``Refund`` records are dropped, so what prints is what the account consumed
rather than what it was charged after its promotional balance was applied.

That distinction is the whole point here. This account runs on credits -- every
``Usage`` line is matched by an equal, opposite ``Credit`` line -- so the netted
total reads $0.00 while the infrastructure still costs real money. A budget
alarm on the netted figure is unreachable until the credits run dry, which is
exactly the moment it would be needed. ``--with-credits`` asks the other
question (what the card is charged today) and, being a look rather than a
reading, records nothing.

Dropping credits is also what makes a service-grouped table honest. AWS files a
credit against a *different service* than the charge it cancels -- on this
account the transfer usage books under "EC2 - Other" and its credit under "AWS
Data Transfer" -- so a netted table shows one service as purely positive and
another as purely negative, neither of which is a thing that happened.

Credits are excluded by *dropping record types*, not by allowlisting ``Usage``
and ``Tax``: AWS books real charges under several other types too
(``DiscountedUsage`` for a reservation, ``SavingsPlanCoveredUsage``, ``Support``)
and an allowlist would discard them silently -- the one failure mode a cost guard
must not have. ``Discount`` records stay for the same reason: a negotiated rate
is part of the real price, where a promotional credit is a balance draining.

Both columns come from **one** Cost Explorer request: monthly granularity over a
wide window, so the current month is the last bucket and the to-date total is the
sum of them all. The record-type split is a second group-by, which CE does not
charge for -- so this stays one request however many columns it feeds.

Three things worth knowing before running it:

1. **The query costs money.** Cost Explorer charges $0.01 per
   ``GetCostAndUsage`` request, and this script makes one per run (two or more
   only if the answer pages). Daily is ~$0.30/month; a cron every five minutes
   is $86/month, i.e. more than the budget it would be guarding.
2. **The numbers lag.** Cost Explorer refreshes a few times a day, so the current
   month is always incomplete and the last hours of spend are simply not there
   yet (it flags that month ``Estimated``). It is a trend gauge, not a ledger.
3. **It needs a permission the report user does not have by default.** The
   credentials in ``~/.aws`` belong to ``eilaf-dev-reports``, whose policy
   otherwise reaches S3 and only S3 -- the conversation prefix of the reports
   bucket (see ``app/whatsapp/SETUP.md``). Without ``ce:GetCostAndUsage`` the
   script says so, with the statement to add, rather than failing opaquely.

"To date" means the queried window, not the account's whole life: Cost Explorer
serves 12 months by default, which is what this asks for. If the account is
older *and* has historical data enabled, widen it with ``--since``.

``UnblendedCost`` is the metric -- the per-account on-demand rate, which for a
single non-consolidated account is what actually gets charged.

The alert fires on the **month**, not on the to-date total, deliberately: a
lifetime figure crosses any fixed line exactly once and then screams forever,
which is a broken alarm. A monthly budget re-arms itself every month. And it
watches the figure the table shows -- the actual cost -- so it cannot sit silent
behind a credit balance.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
from pathlib import Path

# Defaults live here rather than in the arg parser so the two knobs most likely
# to be changed per machine can be set once, in the environment.
ALERT_THRESHOLD_USD = float(os.environ.get("EILAF_COST_ALERT_USD", "40"))
DEFAULT_CSV = Path(__file__).resolve().parents[1] / "data" / "costs" / "aws_costs.csv"
# Cost Explorer is a global service with a single endpoint in us-east-1. This is
# unrelated to EILAF_REPORT_REGION (eu-north-1, where the bucket and box live) --
# asking for costs in eu-north-1 fails outright.
CE_REGION = "us-east-1"
# How far back "to date" reaches when --since is not given. 12 months is what
# Cost Explorer serves without opting into historical data; asking for more on an
# account that has not enabled it is rejected, so this is the safe default.
DEFAULT_LOOKBACK_MONTHS = 12
# Below this a line is noise, not spend: CE reports every service ever touched,
# most of them at exactly 0.00, plus sub-cent rounding dust.
CENT = 0.005
# Named for what they hold. The rename from the earlier net-based columns is
# deliberate and load-bearing: those figures were netted against credits, and a
# delta taken across the two bases would invent a ~$23 jump nobody spent. On the
# next write the old rows migrate to blanks, which is the honest record of a
# history kept on another basis.
# What counts as this project rather than the older work sharing the account.
# A region-and-month rule, which is exact here by luck of timing: the chatbot's
# instance launched 2026-09-12 into eu-north-1, and the earlier workload in that
# region had already dwindled to pennies the month before. Cost allocation
# tags are the principled answer, but they are not retroactive -- see --tag.
CHATBOT_REGION = "eu-north-1"
CHATBOT_SINCE = "2026-09"
CSV_COLUMNS = [
    "snapshot",
    "scope",
    "month",
    "service",
    "chatbot_month_usd",
    "legacy_month_usd",
    "chatbot_to_date_usd",
    "legacy_to_date_usd",
    "delta_usd",
]
# What a run counted: "account" for everything, else the filter that narrowed
# it. Recorded per row and diffed only against its own scope -- a chatbot-only
# snapshot measured against a whole-account one would read as a huge saving.
SCOPE_ALL = "account"
# Record types that reduce a bill without being a charge, and so are what "gross"
# ignores. Discount record types (Savings Plans, EDP, bundled discounts) are
# deliberately *not* here: a negotiated rate is part of the real price, whereas a
# credit is a balance being drawn down. Neither appears on this account today.
CREDIT_RECORDS = {"Credit", "Refund"}
# One table line and the TOTAL under it: the service, then this month split
# chatbot/legacy, then to-date split the same way, then the change in the
# combined to-date. The change is None when there is no snapshot to compare to.
Row = tuple[str, float, float, float, float, float | None]
Totals = tuple[float, float, float, float, float | None]
# Written as a service row so the CSV is readable on its own; filtered out when
# the file is read back, or it would be diffed as if it were a service.
TOTAL_ROW = "TOTAL"

CE_POLICY = """    {
      "Sid": "ReadCostExplorer",
      "Effect": "Allow",
      "Action": "ce:GetCostAndUsage",
      "Resource": "*"
    }"""


def clamp(amount: float) -> float:
    """Zero out sub-cent dust, so nothing prints as a signed "-0.00".

    Not cosmetic only: a credit posts as a line of its own, equal and opposite
    to the charge it cancels, and the two rarely cancel to exactly 0.0 in
    floating point. Without this a fully-credited month reports -0.00. It still
    matters with credits excluded: sub-cent usage dust prints the same way.
    """
    return 0.0 if abs(amount) < CENT else amount


def first_of_month(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y-%m").date()


def window_bounds(since: str | None, month: str | None) -> tuple[str, str, str]:
    """Return (start, end, focus month) for the single request.

    ``end`` is exclusive in the Cost Explorer API, and it is *tomorrow* rather
    than today: passing today would cut today's partial spend out of the answer
    entirely. The window always runs to now, whatever ``--month`` says -- the
    to-date column means to-date, and ``--month`` only chooses which bucket the
    per-month column reports.
    """
    today = dt.date.today()
    current = today.replace(day=1)
    focus = first_of_month(month) if month else current
    if focus > current:
        raise SystemExit(f"aws_costs: {month} is in the future, nothing to report")

    if since:
        start = first_of_month(since)
    else:
        # Step back whole months without a calendar library: 12 months back is
        # the same month number in an earlier year once normalised through
        # month arithmetic.
        total = (current.year * 12 + current.month - 1) - DEFAULT_LOOKBACK_MONTHS
        start = dt.date(total // 12, total % 12 + 1, 1)
    # A --month older than the window would report an empty bucket as $0.00,
    # which reads as "nothing spent" rather than "not asked for".
    start = min(start, focus)
    return (
        start.isoformat(),
        (today + dt.timedelta(days=1)).isoformat(),
        focus.strftime("%Y-%m"),
    )


Costs = dict[str, dict[str, float]]


def fetch_costs(
    start: str,
    end: str,
    profile: str | None,
    with_credits: bool,
    scope_filters: list[dict] | None = None,
) -> tuple[Costs, Costs]:
    """One GetCostAndUsage call -> (chatbot, legacy), each {month: {service: cost}}.

    Costs $0.01. Credits are excluded by **filter** here rather than by grouping
    on ``RECORD_TYPE`` and discarding the credit groups afterwards. That matters
    for a mundane reason: Cost Explorer allows only **two** group-by dimensions,
    and the split below needs ``REGION`` as the second one. Filtering the credits
    out buys that slot back.

    Excluding them is stated as dropping ``Credit`` and ``Refund`` rather than
    keeping ``Usage`` and ``Tax``: AWS books real charges under several other
    types too (``DiscountedUsage`` for a reservation, ``SavingsPlanCoveredUsage``,
    ``Support``) and an allowlist would discard them silently -- the one failure
    mode a cost guard must not have. ``Discount`` records stay for the same
    reason: a negotiated rate is part of the real price, where a promotional
    credit is a balance draining.
    """
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError

    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    client = session.client("ce", region_name=CE_REGION)
    filters = list(scope_filters or [])
    if not with_credits:
        filters.append(
            {
                "Not": {
                    "Dimensions": {
                        "Key": "RECORD_TYPE",
                        "Values": sorted(CREDIT_RECORDS),
                    }
                }
            }
        )
    pages = []
    try:
        token: str | None = None
        while True:
            # MONTHLY over the whole window gives one bucket per month, so the
            # same response answers every column. The final bucket is the
            # current month, partial.
            request: dict = {
                "TimePeriod": {"Start": start, "End": end},
                "Granularity": "MONTHLY",
                "Metrics": ["UnblendedCost"],
                "GroupBy": [
                    {"Type": "DIMENSION", "Key": "SERVICE"},
                    {"Type": "DIMENSION", "Key": "REGION"},
                ],
            }
            # "And" is only legal with two or more expressions, hence the
            # single-element special case.
            if filters:
                request["Filter"] = (
                    filters[0] if len(filters) == 1 else {"And": filters}
                )
            if token:
                request["NextPageToken"] = token
            response = client.get_cost_and_usage(**request)
            pages.append(response)
            # Two group-bys multiply the number of groups, so paging is no
            # longer purely hypothetical. Each extra page is another $0.01, but
            # a truncated answer is a silently wrong one.
            token = response.get("NextPageToken")
            if not token:
                break
    except NoCredentialsError:
        raise SystemExit(
            "aws_costs: no AWS credentials found.\n"
            "  ~/.aws/credentials here holds a named profile, not a default, so "
            "pass it:\n    ./scripts/aws_costs.py --profile eilaf"
        ) from None
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code == "AccessDeniedException":
            raise SystemExit(
                "aws_costs: this IAM identity may not read Cost Explorer.\n"
                "  Add this statement to the user's policy (IAM console > Users >\n"
                "  Permissions > edit the inline policy), then retry:\n\n"
                f"{CE_POLICY}\n\n"
                "  Two account-level settings also gate it, both one-time:\n"
                "  * Cost Explorer must be enabled (Billing console). The first\n"
                "    activation takes up to 24h to return data.\n"
                "  * IAM access to Billing must be on (Account settings > IAM user\n"
                "    and role access to Billing information). Without it an IAM\n"
                "    user is denied even holding ce:GetCostAndUsage."
            ) from None
        if code == "DataUnavailableException":
            raise SystemExit(
                "aws_costs: Cost Explorer has no data for that period yet "
                "(newly enabled accounts wait up to 24h)."
            ) from None
        if code == "ValidationException" and "historical" in str(exc).lower():
            raise SystemExit(
                f"aws_costs: {start} is further back than this account serves.\n"
                "  Cost Explorer gives 12 months unless historical data is "
                "enabled. Drop --since, or enable it in the Billing console."
            ) from None
        raise

    chatbot: Costs = {}
    legacy: Costs = {}
    for response in pages:
        for period in response.get("ResultsByTime", []):
            label = period["TimePeriod"]["Start"][:7]
            for group in period.get("Groups", []):
                service, region = group["Keys"]
                amount = float(group["Metrics"]["UnblendedCost"]["Amount"])
                # YYYY-MM strings compare chronologically, so the month cutoff
                # needs no date parsing.
                mine = region == CHATBOT_REGION and label >= CHATBOT_SINCE
                bucket = (chatbot if mine else legacy).setdefault(label, {})
                bucket[service] = bucket.get(service, 0.0) + amount
    return chatbot, legacy


def previous_snapshot(path: Path, scope: str) -> tuple[str, dict[str, float]]:
    """The most recent snapshot in the file: (timestamp, per-service to-date).

    Diffed on the **to-date** figure, not the month's, so the delta stays
    meaningful across a month boundary -- a to-date total only ever grows, while
    the month's resets to zero on the 1st and would show every service as having
    gone sharply negative.
    """
    if not path.exists():
        return "", {}

    # Taken as the last *block* of rows rather than the highest timestamp:
    # snapshots are appended in order, and trusting the order cannot be fooled
    # by a clock that went backwards or by two runs a second apart.
    stamp: str = ""
    costs: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            # Rows from a different scope -- or from before the column existed,
            # which counted the whole account -- are not comparable.
            if (row.get("scope") or SCOPE_ALL) != scope:
                continue
            if row["snapshot"] != stamp:
                stamp, costs = row["snapshot"], {}
            # The TOTAL row still moves the timestamp along, and only then is it
            # dropped: a month covered entirely by credits records nothing
            # BUT a total, and forgetting that stamp would make the run after it
            # look like the first ever rather than "no change since".
            if row["service"] == TOTAL_ROW:
                continue
            # The two halves summed, which is what the change column tracks.
            # No fall back to older column names: those held figures on another
            # basis (netted against credits, or unsplit), and reviving them is
            # what would produce a delta spanning two different meanings.
            try:
                costs[row["service"]] = float(row["chatbot_to_date_usd"]) + float(
                    row["legacy_to_date_usd"]
                )
            except (KeyError, TypeError, ValueError):
                continue
    return stamp, costs


def migrate_header(path: Path) -> None:
    """Rewrite a CSV written by an older column layout, in place.

    Appending new-shaped rows under an old header would leave a file that no
    reader can parse as one table -- worse than either shape alone, and silent.
    """
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle), [])
    if header == CSV_COLUMNS:
        return
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # cost_usd was this column's name before a to-date column existed.
            if "month_usd" not in row and "cost_usd" in row:
                row["month_usd"] = row["cost_usd"]
            # Nothing is back-filled into the split columns: older rows were
            # netted against credits and unsplit, so copying a figure across
            # would assert both a basis and an attribution it never had.
            writer.writerow(row)
    print(f"note: migrated {path} to the current columns ({', '.join(CSV_COLUMNS)})")


def append_snapshot(
    path: Path,
    snapshot: str,
    scope: str,
    month: str,
    rows: list[Row],
    totals: Totals,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    migrate_header(path)
    is_new = not path.exists()
    bot_month, leg_month, bot_date, leg_date, total_delta = totals

    def cell(value: float | None) -> str:
        return "" if value is None else f"{value:.2f}"

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(CSV_COLUMNS)
        for service, bm, lm, bd, ld, delta in rows:
            writer.writerow(
                [
                    snapshot,
                    scope,
                    month,
                    service,
                    cell(bm),
                    cell(lm),
                    cell(bd),
                    cell(ld),
                    cell(delta),
                ]
            )
        writer.writerow(
            [
                snapshot,
                scope,
                month,
                TOTAL_ROW,
                cell(bot_month),
                cell(leg_month),
                cell(bot_date),
                cell(leg_date),
                cell(total_delta),
            ]
        )


def print_table(
    rows: list[Row],
    totals: Totals,
    month: str,
    window_start: str,
    since: str,
    with_credits: bool,
    scope: str,
) -> None:
    """One line per service, each cost split chatbot | legacy.

    The split lives in columns rather than in two tables so a service appears
    exactly once: EC2 compute is one row reading 26.62 and 300.89, not the same
    name twice under different headings with the reader doing the joining.
    """
    bot_month, leg_month, bot_date, leg_date, total_delta = totals
    empty = "(nothing at or above $0.01)"
    width = max([len("Service"), len(empty)] + [len(row[0]) for row in rows])
    col = 10
    # The timestamp goes in the footer, not the heading: as a column header it
    # was the widest thing in the table, pushing four money columns apart to
    # label something that does not change between rows.
    delta_head = "change"
    dwidth = 9

    def fmt_delta(delta: float | None) -> str:
        if delta is None:
            return "-"
        # An explicit + marks a rise; a service new since the last check would
        # otherwise read the same as one that merely stayed flat.
        return f"{delta:+.2f}" if abs(delta) >= CENT else "0.00"

    def money(value: float) -> str:
        # A hard zero prints as a dash: a column of 0.00s is visual noise that
        # competes with the figures that matter, and "this service is not in
        # this half" is better said than spelled as a number.
        return f"{value:>{col}.2f}" if abs(value) >= CENT else f"{'-':>{col}}"

    basis = "billed, after credits" if with_credits else "actual cost, credits excluded"
    counted = "whole account" if scope == SCOPE_ALL else scope
    print(f"\nAWS spend by service (UnblendedCost, USD -- {basis}; {counted})")
    # Two banded headers: the period on top, the halves under it. Without the
    # band, four similar numbers per row give the reader no way to tell which
    # two are the month and which two the window.
    print(
        f"{'':<{width}}  {month + ' month-to-date':^{col * 2 + 2}}  "
        f"{'since ' + window_start:^{col * 2 + 2}}"
    )
    print(
        f"{'Service':<{width}}  {'chatbot':>{col}}  {'legacy':>{col}}  "
        f"{'chatbot':>{col}}  {'legacy':>{col}}  {delta_head:>{dwidth}}"
    )
    rule = "-" * (width + col * 4 + dwidth + 10)
    print(rule)
    if not rows:
        print(f"{empty:<{width}}" + f"  {'-':>{col}}" * 4 + f"  {'-':>{dwidth}}")
    # Most services bill in only one half, and most stopped billing months ago --
    # so a single table of four money columns is mostly blank. The ones still
    # running get the full row; the dormant ones go below with the two columns
    # that can say anything about them, rather than two more empty cells each.
    live = [r for r in rows if r[1] or r[2]]
    dormant = [r for r in rows if not (r[1] or r[2])]
    for service, bm, lm, bd, ld, delta in live:
        print(
            f"{service:<{width}}  {money(bm)}  {money(lm)}  {money(bd)}  {money(ld)}  "
            f"{fmt_delta(delta):>{dwidth}}"
        )
    if dormant:
        total_dormant = sum(r[3] + r[4] for r in dormant)
        print(
            f"\n{f'nothing in {month} -- to date only':<{width}}  "
            f"{'chatbot':>{col}}  {'legacy':>{col}}   ${total_dormant:,.2f} in total"
        )
        for service, _bm, _lm, bd, ld, _delta in dormant:
            print(f"  {service:<{width - 2}}  {money(bd)}  {money(ld)}")

    # The TOTAL sits below both blocks, because it totals both: placed directly
    # under the live rows it would be a sum of figures the reader cannot see.
    print(rule)
    print(
        f"{'TOTAL':<{width}}  {bot_month:>{col}.2f}  {leg_month:>{col}.2f}  "
        f"{bot_date:>{col}.2f}  {leg_date:>{col}.2f}  {fmt_delta(total_delta):>{dwidth}}"
    )
    # Combined figures as prose rather than a fifth pseudo-row: a row whose
    # cells sit under "chatbot" while meaning "both" reads as a wrong number.
    print(
        f"\nBoth halves together: {month} ${bot_month + leg_month:,.2f}, "
        f"since {window_start} ${bot_date + leg_date:,.2f}."
    )
    if since:
        print(
            f"'change' is the move in that second figure since the snapshot of "
            f"{since[:16].replace('T', ' ')}."
        )
    print(
        f"'chatbot' is {CHATBOT_REGION} from {CHATBOT_SINCE} (its instance launched"
        f" 2026-09-12); 'legacy' is\neverything else in scope."
    )
    if with_credits:
        print(
            "Credits and refunds are netted in: this is what is charged, not what it costs."
        )
    else:
        print(
            "Credit and Refund records are excluded; --with-credits nets them back in."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="AWS spend per service: the month so far, the total to date, and the change since the last run.",
        epilog="Exit code 2 means the month exceeded the budget; 1 means the query failed.",
    )
    parser.add_argument(
        "--profile",
        default=os.environ.get("AWS_PROFILE") or "eilaf",
        help="AWS profile (default: $AWS_PROFILE, else 'eilaf' — the only one in ~/.aws)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=ALERT_THRESHOLD_USD,
        help=f"alert above this month-to-date total in USD (default: {ALERT_THRESHOLD_USD:.2f}, or $EILAF_COST_ALERT_USD)",
    )
    parser.add_argument(
        "--month", help="YYYY-MM to report instead of the current month"
    )
    parser.add_argument(
        "--since",
        help=f"YYYY-MM the to-date column reaches back to (default: {DEFAULT_LOOKBACK_MONTHS} months)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"CSV to append to (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="print the table without recording a snapshot",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="include services costing nothing in either column",
    )
    parser.add_argument(
        "--region",
        help="count only this region, e.g. eu-north-1 (the chatbot's). Note that a domain, Route 53 and tax sit in 'global'/'NoRegion' and are excluded by any region filter",
    )
    parser.add_argument(
        "--tag",
        metavar="KEY=VALUE",
        help="count only resources carrying this cost allocation tag, e.g. Project=eilaf-chatbot. The tag must be activated in Billing > Cost allocation tags, and applies only to usage after activation",
    )
    parser.add_argument(
        "--with-credits",
        action="store_true",
        help="report what is billed after credits instead of what it costs; records nothing",
    )
    args = parser.parse_args(argv)

    start, end, month = window_bounds(args.since, args.month)
    scope_filters: list[dict] = []
    scope_parts: list[str] = []
    if args.region:
        scope_filters.append({"Dimensions": {"Key": "REGION", "Values": [args.region]}})
        scope_parts.append(f"region={args.region}")
    if args.tag:
        if "=" not in args.tag:
            raise SystemExit(
                "aws_costs: --tag wants KEY=VALUE, e.g. Project=eilaf-chatbot"
            )
        key, value = args.tag.split("=", 1)
        scope_filters.append({"Tags": {"Key": key, "Values": [value]}})
        scope_parts.append(f"tag:{key}={value}")
    # An explicit --since changes what "to date" counts, so it belongs in the
    # scope too: a window pinned to the chatbot's first month and the default
    # rolling one are different series, and diffing across them would invent the
    # months between. The default window is deliberately *not* labelled -- it
    # slides forward every month, and labelling it would reset the history each
    # time, which is the opposite of tracking spend as you go.
    if args.since:
        scope_parts.append(f"from={args.since}")
    scope = " ".join(scope_parts) if scope_parts else SCOPE_ALL

    chatbot_months, legacy_months = fetch_costs(
        start, end, args.profile, args.with_credits, scope_filters
    )
    # The history is kept on one basis only, so a --with-credits run has nothing
    # legitimate to compare against: diffing netted figures against recorded
    # actual ones reports a change the size of the credit balance.
    since, before = (
        ("", {}) if args.with_credits else previous_snapshot(args.csv, scope)
    )
    # A stamp with no per-service figures behind it is not a baseline either: a
    # file kept on older columns has exactly that shape, and diffing against
    # nothing would report the whole history as this run's change. Better to say
    # "no previous reading" for one run.
    if not before:
        since = ""

    def window_sum(months: Costs) -> dict[str, float]:
        out: dict[str, float] = {}
        for bucket in months.values():
            for service, cost in bucket.items():
                out[service] = out.get(service, 0.0) + cost
        return out

    bot_month_costs = chatbot_months.get(month, {})
    leg_month_costs = legacy_months.get(month, {})
    bot_to_date = window_sum(chatbot_months)
    leg_to_date = window_sum(legacy_months)

    rows: list[Row] = []
    for service in set(bot_to_date) | set(leg_to_date) | set(before):
        bm = clamp(bot_month_costs.get(service, 0.0))
        lm = clamp(leg_month_costs.get(service, 0.0))
        bd = clamp(bot_to_date.get(service, 0.0))
        ld = clamp(leg_to_date.get(service, 0.0))
        # Kept if *any* figure has substance: a service that billed once months
        # ago still belongs in a to-date table, even at nothing this month.
        if not args.all and not (bm or lm or bd or ld):
            continue
        delta = None if not since else clamp((bd + ld) - before.get(service, 0.0))
        rows.append((service, bm, lm, bd, ld, delta))
    # Ordered by what it costs over the window, chatbot first: that is the half
    # the reader came for, and this month's column is $0.00 for anything that
    # has stopped -- no order at all in a month where little has billed.
    rows.sort(key=lambda row: (-(row[3] + row[4]), -row[3], row[0]))

    bot_month = clamp(sum(bot_month_costs.values()))
    leg_month = clamp(sum(leg_month_costs.values()))
    bot_date = clamp(sum(bot_to_date.values()))
    leg_date = clamp(sum(leg_to_date.values()))
    month_total = clamp(bot_month + leg_month)
    totals = (
        bot_month,
        leg_month,
        bot_date,
        leg_date,
        None
        if not since
        else clamp((bot_date + leg_date) - clamp(sum(before.values()))),
    )
    print_table(rows, totals, month, start, since, args.with_credits, scope)
    # An empty narrowed answer is ambiguous in a way the whole-account one is
    # not: a tag that is not activated yet, or activated after the usage, looks
    # exactly like a resource that cost nothing.
    if scope != SCOPE_ALL and not rows:
        print(
            f"\nnothing matched {scope}. A cost allocation tag covers only usage\n"
            "incurred after it was activated in Billing, so this can mean the tag\n"
            "is right but the history predates it."
        )

    # Two kinds of run record nothing. A back-dated look is not the latest state,
    # and recording it would make the next run diff against a month that is no
    # longer the one being reported. A --with-credits run is a different question
    # answered on a different basis, and filing it in an actual-cost history is
    # what would make the next delta meaningless.
    backdated = bool(args.month) and month != dt.date.today().strftime("%Y-%m")
    if args.no_write or backdated or args.with_credits:
        why = (
            "--no-write"
            if args.no_write
            else "--with-credits is a look, not a reading"
            if args.with_credits
            else f"--month {month} is a look back"
        )
        print(f"({why}: snapshot not recorded)")
    else:
        append_snapshot(
            args.csv,
            dt.datetime.now().isoformat(timespec="milliseconds"),
            scope,
            month,
            rows,
            totals,
        )
        print(f"appended to {args.csv}")

    if month_total >= args.threshold:
        # The banner goes to stderr so a cron or a pipeline can separate it from
        # the table -- which means flushing stdout first, or a redirected run
        # prints the alert *above* the numbers it is about.
        sys.stdout.flush()
        banner = f"  ALERT: {month} AWS spend is ${month_total:.2f}, over the ${args.threshold:.2f} budget  "
        rule = "=" * len(banner)
        print(f"\n{rule}\n{rule}\n{banner}\n{rule}\n{rule}", file=sys.stderr)
        return 2
    # One quiet heads-up before the loud one, so the first warning is not also
    # the moment the budget is already blown.
    if month_total >= 0.8 * args.threshold:
        print(
            f"\nnote: at {month_total / args.threshold:.0%} of the ${args.threshold:.2f} budget "
            f"(${args.threshold - month_total:.2f} left).",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
