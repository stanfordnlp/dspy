"""Generate a realistic cohort analysis dataset (>5MB) for RLM + DataFrame testing.

Creates three DataFrames mimicking a SaaS product:
  - users:         10,000 users with signup info and acquisition channels
  - events:        ~500,000 feature usage events
  - subscriptions: 10,000 subscription records with churn patterns

Embedded signal (what the RLM should discover):
  - Worst channel: paid_campaign_x (highest churn).
  - Churn rates: paid_campaign_x ~45% (3x base); social ~22.5% (1.5x base);
    organic, referral, paid_campaign_y ~15% (base). Overall ~23–25%.
  - Behavioral: churned users from paid_campaign_x never use "advanced_reports";
    retained users do — so early advanced_reports usage is an activation signal.
"""

import random
from datetime import timedelta

import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)

NUM_USERS = 10_000
AVG_EVENTS_PER_USER = 50
START_DATE = pd.Timestamp("2024-01-01")
END_DATE = pd.Timestamp("2024-06-30")

CHANNELS = ["organic", "referral", "paid_campaign_x", "paid_campaign_y", "social"]
CHANNEL_WEIGHTS = [0.30, 0.15, 0.25, 0.15, 0.15]

PLANS = ["free", "starter", "pro", "enterprise"]
PLAN_WEIGHTS = [0.40, 0.30, 0.20, 0.10]
PLAN_MRR = {"free": 0, "starter": 29, "pro": 79, "enterprise": 249}

FEATURES = [
    "dashboard_view", "report_export", "advanced_reports",
    "team_invite", "api_call", "integration_setup",
    "settings_update", "search", "file_upload", "notification_click",
]

COUNTRIES = ["US", "UK", "DE", "FR", "CA", "AU", "BR", "IN", "JP", "KR"]
COUNTRY_WEIGHTS = [0.30, 0.12, 0.10, 0.08, 0.08, 0.06, 0.08, 0.08, 0.05, 0.05]

CANCELLATION_REASONS = [
    "too_expensive", "missing_features", "switched_competitor",
    "no_longer_needed", "poor_support", "other",
]


def generate_users() -> pd.DataFrame:
    rows = []
    for i in range(1, NUM_USERS + 1):
        signup = START_DATE + timedelta(days=random.randint(0, 150))
        channel = random.choices(CHANNELS, weights=CHANNEL_WEIGHTS, k=1)[0]
        plan = random.choices(PLANS, weights=PLAN_WEIGHTS, k=1)[0]
        country = random.choices(COUNTRIES, weights=COUNTRY_WEIGHTS, k=1)[0]
        rows.append({
            "user_id": i,
            "email": fake.email(),
            "name": fake.name(),
            "signup_date": signup,
            "acquisition_channel": channel,
            "plan_at_signup": plan,
            "country": country,
        })
    return pd.DataFrame(rows)


def generate_subscriptions(users: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, user in users.iterrows():
        uid = user["user_id"]
        channel = user["acquisition_channel"]
        plan = user["plan_at_signup"]
        start = user["signup_date"]
        mrr = PLAN_MRR[plan]

        # Churn probability by channel (expected: paid_campaign_x ~45%, social ~22.5%, others ~15%)
        base_churn = 0.15
        if channel == "paid_campaign_x":
            churn_prob = min(base_churn * 3, 0.80)  # ~45%
        elif channel == "social":
            churn_prob = base_churn * 1.5  # ~22.5%
        else:
            churn_prob = base_churn  # ~15%

        churned = random.random() < churn_prob
        if churned:
            days_active = random.randint(7, 90)
            end = start + timedelta(days=days_active)
            if end > END_DATE:
                end = END_DATE
                churned = False
            status = "cancelled"
            reason = random.choice(CANCELLATION_REASONS)
        else:
            end = None
            status = "active"
            reason = None

        rows.append({
            "user_id": uid,
            "subscription_start": start,
            "subscription_end": end,
            "plan": plan,
            "mrr": mrr,
            "status": status,
            "cancellation_reason": reason,
        })
    return pd.DataFrame(rows)


def generate_events(users: pd.DataFrame, subscriptions: pd.DataFrame) -> pd.DataFrame:
    sub_lookup = subscriptions.set_index("user_id")
    rows = []

    for _, user in users.iterrows():
        uid = user["user_id"]
        channel = user["acquisition_channel"]
        signup = user["signup_date"]
        sub = sub_lookup.loc[uid]
        end = sub["subscription_end"] if pd.notna(sub["subscription_end"]) else END_DATE

        # paid_campaign_x churners never use advanced_reports (embedded behavioral signal)
        churned = sub["status"] == "cancelled"
        if channel == "paid_campaign_x" and churned:
            available_features = [f for f in FEATURES if f != "advanced_reports"]
        else:
            available_features = FEATURES

        # Number of events scales with days active
        days_active = (end - signup).days
        n_events = max(5, int(AVG_EVENTS_PER_USER * (days_active / 180) * random.uniform(0.5, 1.5)))

        for _ in range(n_events):
            event_time = signup + timedelta(
                seconds=random.randint(0, max(1, days_active * 86400))
            )
            feature = random.choice(available_features)

            rows.append({
                "user_id": uid,
                "event_type": feature,
                "timestamp": event_time,
                "session_id": fake.uuid4()[:8],
                "duration_seconds": random.randint(1, 300),
            })

    return pd.DataFrame(rows)


def main():
    print("Generating users...")
    users = generate_users()

    print("Generating subscriptions...")
    subscriptions = generate_subscriptions(users)

    print("Generating events...")
    events = generate_events(users, subscriptions)

    # Sort events by timestamp
    events = events.sort_values("timestamp").reset_index(drop=True)

    # Save to parquet for efficient storage
    users.to_parquet("users.parquet", index=False)
    subscriptions.to_parquet("subscriptions.parquet", index=False)
    events.to_parquet("events.parquet", index=False)

    # Print stats
    total_bytes = (
        users.memory_usage(deep=True).sum()
        + subscriptions.memory_usage(deep=True).sum()
        + events.memory_usage(deep=True).sum()
    )
    print(f"\nDataset stats:")
    print(f"  Users:         {len(users):>10,} rows")
    print(f"  Subscriptions: {len(subscriptions):>10,} rows")
    print(f"  Events:        {len(events):>10,} rows")
    print(f"  Total memory:  {total_bytes / 1024 / 1024:>10.1f} MB")
    print(f"\nEmbedded signal:")
    print(f"  paid_campaign_x churn rate: {subscriptions[subscriptions['user_id'].isin(users[users['acquisition_channel']=='paid_campaign_x']['user_id'])]['status'].value_counts(normalize=True).get('cancelled', 0):.0%}")
    print(f"  organic churn rate:         {subscriptions[subscriptions['user_id'].isin(users[users['acquisition_channel']=='organic']['user_id'])]['status'].value_counts(normalize=True).get('cancelled', 0):.0%}")
    print(f"\nSaved to: users.parquet, subscriptions.parquet, events.parquet")


if __name__ == "__main__":
    main()
