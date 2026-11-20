"""The loot rule: which bases are worth the army."""

from vision.loot import LootReading

# Shown gold + elixir a base must hold. One number, not one per resource: what a battle takes follows
# the total, and dark elixir is left out (it sits at the storage cap). Replay of the 84 star-stop
# battles of 2026-11-16 (docs/phase-5-audit/threshold_replay.py): 39% of offers clear 1.5M, and
# skipping the rest is worth about +20% loot per hour over attacking everything, at ~3 s per Next.
MIN_TOTAL = 1_500_000

# Offers drift with the hour and the league, and a threshold they have drifted below would search
# forever. After this many bases skipped in a row the next readable one is attacked whatever it holds:
# at 39% accepted, 15 skips in a row is a 1-in-1600 event, so hitting it means the threshold is stale.
MAX_NEXTS = 15

# A loot panel that stays unreadable base after base is a style the reader does not know, not a glitch:
# a 2x cupcake turns the digits yellow (frames 56-57) and every base reads "no glyphs". After this many
# unreadable bases in a row the next one is attacked unread. Under a cupcake any base pays double.
MAX_UNREADABLE = 3


def should_attack(reading: LootReading) -> bool:
    """Whether a base's shown loot clears the threshold. An unreadable panel never does."""

    return reading.ok and reading.gold + reading.elixir >= MIN_TOTAL
