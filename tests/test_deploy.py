"""Deploy points and the first card are found on scout frames and nowhere else."""

import cv2
import pytest

from vision.deploy import MAX_POINTS, UI_RECTS, deploy_points, first_card

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


@pytest.mark.parametrize("n, centre", [(4, (214, 980)), *[(n, (206, 980)) for n in WITH_OUTLINE]])
def test_first_card_is_found(frame, n, centre):
    card = first_card(frame(n))
    assert card is not None and card.centre == centre


@pytest.mark.parametrize("n", [1, 3])  # home, army: a bright run in the band that is not a card
def test_no_deck_gives_no_card(frame, n):
    assert first_card(frame(n)) is None


def test_resized_frame_is_refused(frame):
    with pytest.raises(ValueError):
        deploy_points(cv2.resize(frame(5), (960, 540)))
