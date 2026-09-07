"""Curated canonical-merchant list for alias resolution (design doc
§5.1). This is a small, hand-curated table sized for this demo's ~40
merchants -- a production system would maintain a much larger table,
typically mined from transaction volume rather than hand-written. The
point being demonstrated is the *technique* (fuzzy resolution against a
canonical list), not the size of the list.
"""

from __future__ import annotations

CANONICAL_MERCHANTS: list[str] = [
    "Acme Corp Payroll",
    "Ally Bank Interest",
    "Meridian Properties",
    "Georgia Power",
    "City Water Dept",
    "Xfinity Internet",
    "Verizon Wireless",
    "Kroger",
    "Whole Foods Market",
    "Blue Bottle Coffee",
    "Starbucks",
    "Chipotle",
    "Chick-fil-A",
    "The Local Tavern",
    "Olive Branch Bistro",
    "DoorDash",
    "Shell Oil",
    "Chevron",
    "Uber",
    "Lyft",
    "MARTA",
    "ParkMobile",
    "Amazon",
    "Target",
    "Best Buy",
    "Apple Store",
    "H&M",
    "IKEA",
    "CVS Pharmacy",
    "Piedmont Medical",
    "Anytime Fitness",
    "Netflix",
    "Spotify",
    "AMC Theatres",
    "Steam",
    "Delta Air Lines",
    "Marriott",
    "Hertz",
    "Chase Bank Fee",
    "Chase Interest Charge",
]

# A few merchants render as bare domain-style strings ("STEAMGAMES.COM",
# "BESTBUY.COM*4821") with no separable human-readable name for the
# generic cleaning cascade to recover -- fuzzy matching alone struggles
# on these, so they get literal substring aliases as a targeted fix.
DOMAIN_ALIASES: dict[str, str] = {
    "STEAMGAMES": "Steam",
    "BESTBUY": "Best Buy",
    "APPLE.COM": "Apple Store",
    "AMZN": "Amazon",
    "AMAZON.COM": "Amazon",
    "SBUX": "Starbucks",
    "WHOLEFDS": "Whole Foods Market",
}
