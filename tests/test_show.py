"""The --show window draws what the bot found on its own frame over a different frame (the live video)."""

import numpy as np

import main
from policy.navigator import WAIT, Action
from vision.anchors import THRESHOLD, peak_scores
from vision.buttons import load_button_templates
from vision.deploy import deck_slots, deploy_points, outline_lines, outline_mask
from vision.overlay import MAGENTA
from vision.screens import Screen, classify


def test_findings_are_drawn_over_the_video_frame(frame):
    bot_frame = frame(47)  # scout: outline, deploy points and deck all found
    peaks = peak_scores(bot_frame, load_button_templates())
    screen = classify({k: m for k, m in peaks.items() if m.score >= THRESHOLD})
    assert screen is Screen.SCOUT
    lines = outline_lines(outline_mask(bot_frame))

    video = np.zeros_like(bot_frame)  # stands in for a video frame the bot never analysed
    findings = main.Findings(1, screen, Action(WAIT, why="test"), peaks, None, deck_slots(bot_frame), [],
                             deploy_points(bot_frame), [], None, 0.0, 0, lines, None, [], (0, 0, 0), 0.0)
    image = main.render(video, findings)

    game = image[-video.shape[0]:]  # under the text header
    outline = np.all(game == MAGENTA, axis=2)
    assert outline[lines > 0].mean() > 0.9  # the outline found on the bot's frame, drawn on the video
