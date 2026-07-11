from dataclasses import dataclass
from typing import Optional


@dataclass
class PlaypenEnvConfig:
    seed: int = 42
    render_mode: str = "text"
    # Clembench games have no adversarial-bot concept: every role is played by the same
    # learner policy (self-play), so this is always "none". Kept only so EnvManager's
    # generic `built_in_opponent == "none"` self-play check works unchanged.
    built_in_opponent: str = "none"
    opponent_player: int = 1
    include_opponent_turn: str = "full"

    # Name of the clembench game to play, e.g. "wordle", "taboo". Must be discoverable by
    # clemcore.clemgame.registry.GameRegistry (a `clembench/<game_name>/clemgame.json` under
    # the process's current working directory).
    game_name: str = "wordle"
    # Optional playpen-data HF split ("train"/"validation") to restrict game instances to.
    # None (default) uses the clembench game's packaged default instances.json.
    game_instance_split: Optional[str] = None
    # Whether to cycle through game instances once (True) or infinitely (False, default;
    # appropriate for RL training where many episodes replay the same finite instance pool).
    single_pass: bool = False
