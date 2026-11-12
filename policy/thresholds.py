"""Fixed manual loot threshold setting to decide whether to attack a base or not."""

from vision.loot import LootReading

MIN_GOLD = 0
MIN_ELIXIR = 0
MIN_DARK = 0


def should_attack(reading: LootReading) -> bool:
    """Returns whether a base is worth attacking."""

    if not reading.ok:
        return False

    return (
        reading.gold >= MIN_GOLD
        and reading.elixir >= MIN_ELIXIR
        and reading.dark_elixir >= MIN_DARK
    )
