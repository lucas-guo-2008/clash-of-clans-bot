"""Fixed manual loot threshold setting to decide whether to attack a base or not."""

from vision.loot import LootReading

MIN_GOLD = 400_000
MIN_ELIXIR = 400_000
MIN_DARK = 2_000

"""Returns whether a base is worth attacking"""
def should_attack(
    reading: LootReading,
    min_gold: int = MIN_GOLD,
    min_elixir: int = MIN_ELIXIR,
    min_dark: int = MIN_DARK,
) -> bool:

    if not reading.ok:
        return False

    return (
        reading.gold >= min_gold
        and reading.elixir >= min_elixir
        and reading.dark_elixir >= min_dark
    )
