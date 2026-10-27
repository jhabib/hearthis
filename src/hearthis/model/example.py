"""The working paper's explicitly synthetic two-listener calculation."""

import numpy as np

from .probability import probabilities
from .taste import Intent, Listener


def worked_example() -> dict:
    names = ("B", "G", "NC", "NL", "V", "P", "E")
    songs = np.array([[0.45, 0.80, 0.80, 0.55, 0.75, 0.95, 0.20],
                      [0.65, 0.85, 0.75, 0.75, 0.85, 0.25, 0.95]])
    centers = [[0.70, 0.90, 0.90, 0.80, 0.90, 0.90, 0.20],
               [0.70, 0.90, 0.90, 0.80, 0.90, 0.20, 0.95]]
    result = {"provenance": "synthetic numerical profiles from the working paper; not empirical song measurements",
              "songs": ["Diamonds", "After Love"], "features": list(names), "listeners": {}}
    for name, center in zip(["A", "B"], centers, strict=True):
        listener = Listener.from_centers(name, names, [center], provenance="paper synthetic example")
        utility = listener.utility(songs, Intent())
        distribution = probabilities(4 * utility)
        result["listeners"][name] = {"range_energies": (-utility).tolist(),
                                      "target_probabilities": distribution.tolist(),
                                      "preferred_song": result["songs"][int(np.argmax(distribution))],
                                      "preferred_odds": float(distribution.max() / distribution.min())}
    return result
