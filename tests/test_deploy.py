"""Deploy points and the first card are found on scout frames and nowhere else."""

import cv2
import pytest

from vision.deploy import MAX_POINTS, UI_RECTS, deck_slots, deploy_points

WITH_OUTLINE = [5, 6, 9, *range(10, 18)]
NO_OUTLINE = [0, 1, 2, 3, 4, 7, 8]  # 4 is a scout frame with no outline drawn


@pytest.mark.parametrize("n", WITH_OUTLINE)
def test_outline_frame_gives_full_set_of_points(frame, n):
    points = deploy_points(frame(n))
    assert len(points) == MAX_POINTS
    for p in points:
        assert not any(x0 <= p.x < x1 and y0 <= p.y < y1 for x0, y0, x1, y1 in UI_RECTS)


@pytest.mark.parametrize("n", NO_OUTLINE)
def test_no_outline_gives_no_points(frame, n):
    assert deploy_points(frame(n)) == []


# Left edge of every deck card. Hero cards (current deck, slots 3-4) have dark art and the
# spell's is white, so these come from card outlines, not art.
CURRENT_DECK = [140, 302, 463, 608, 769]
DECK_LEFTS = {
    4: [148, 294, 453, 615],  # older 4-card deck, shifted 9 px
    5: [140, 285, 447, 607, 769],
    6: [140, 285, 447, 609, 771],
    7: [134, 299, 463, 606, 769],  # battle: the selected card is raised and wider
    **{n: CURRENT_DECK for n in [9, *range(10, 18)]},
}


@pytest.mark.parametrize("n, lefts", DECK_LEFTS.items())
def test_every_deck_card_is_found(frame, n, lefts):
    assert [s.x for s in deck_slots(frame(n))] == lefts


@pytest.mark.parametrize("n", [0, 1, 2, 3, 8, *range(18, 24)])  # no deck; 1 and 23 have card-width edges elsewhere
def test_no_deck_gives_no_slots(frame, n):
    assert deck_slots(frame(n)) == []


def test_resized_frame_is_refused(frame):
    with pytest.raises(ValueError):
        deploy_points(cv2.resize(frame(5), (960, 540)))
