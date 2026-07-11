import logging
from typing import Any, Dict, Optional

from roll.agentic.env.base import BaseLanguageBasedEnv
from roll.agentic.env.playpen.config import PlaypenEnvConfig
from roll.agentic.utils import all_seed
from roll.utils.logging import get_logger

_MAX_LOGGED_COLLECTION_LEN = 20
_MAX_LOGGED_STRING_LEN = 300


def _summarize_for_log(value: Any) -> Any:
    """Recursively shrinks large values (arbitrarily nested) for logging.

    Some clembench games (e.g. wordle) keep large static resources (full word lists,
    localization string tables) on the same GameState object as the small, genuinely
    informative per-episode diagnostic fields (e.g. Taboo's clue_error, wordle's
    current_guess/guess_feedback) -- and those large resources aren't necessarily top-level
    fields (wordle's live in state.words["official_words_list"]). Summarizing recursively,
    rather than just capping the final string length, means the actually useful small
    fields still show up regardless of where the large ones sit in the structure.
    """
    if isinstance(value, dict):
        if len(value) > _MAX_LOGGED_COLLECTION_LEN:
            return f"<dict of {len(value)} keys>"
        return {k: _summarize_for_log(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        if len(value) > _MAX_LOGGED_COLLECTION_LEN:
            return f"<{type(value).__name__} of {len(value)} items>"
        return type(value)(_summarize_for_log(v) for v in value)
    if isinstance(value, str) and len(value) > _MAX_LOGGED_STRING_LEN:
        return value[:_MAX_LOGGED_STRING_LEN] + "...<truncated>"
    return value


class PlaypenEnv(BaseLanguageBasedEnv):
    """Generic wrapper exposing any clembench game (via Playpen's clemcore dependency) as a
    MARSHAL environment.

    Unlike the per-game OpenSpiel env modules (tictactoe/, hanabi/, ...), this class is
    game-agnostic: clembench games are entirely config/data-driven on the clemcore side (one
    GameMasterEnv + a game_name string), so a single wrapper parameterized by
    `config.game_name` covers both single-role games (e.g. wordle) and cooperative
    multi-role games (e.g. taboo) uniformly.

    Every role is always played by the learner policy (self-play; clembench has no built-in
    adversarial-bot concept), so `AgentControlWrapper` is configured with every role marked
    "learner". Actions are free text (there is no enumerable legal-actions set), so
    `legal_actions` is always an empty dict here -- EnvManager uses that as the sentinel to
    switch into free-text validation/prompting (see roll/agentic/rollout/env_manager.py).
    """

    def __init__(self, config: PlaypenEnvConfig = PlaypenEnvConfig()):
        # Imported lazily so that `clemcore` (an optional, playpen-only dependency) is only
        # required by processes that actually instantiate this env.
        #
        # IMPORTANT: importing clemcore for the first time runs
        # `logging.config.dictConfig(...)` as an import side effect (clemcore/__init__.py).
        # Python's dictConfig defaults `disable_existing_loggers=True`, which disables every
        # logger that already existed and isn't explicitly listed in clemcore's own config --
        # including MARSHAL's own loggers (roll.utils.logging.get_logger()'s per-worker
        # logger, and this module's own `logger`). Left unfixed, this silently kills all of
        # MARSHAL's own INFO/WARNING logging in any worker process that instantiates a
        # PlaypenEnv (confirmed empirically: zero MARSHAL-formatted log lines appeared in an
        # EnvironmentWorker log after this import, only clemcore's own). Re-enable anything
        # under MARSHAL's own logger names that clemcore's import just disabled.
        _pre_existing_loggers = list(logging.Logger.manager.loggerDict.keys())
        from clemcore.clemgame.envs.pettingzoo import env as clemcore_env
        from clemcore.clemgame.envs.pettingzoo.wrappers import AgentControlWrapper
        from clemcore.clemgame.instances import GameInstances, to_instance_filter
        from clemcore.clemgame.registry import GameRegistry

        for _name in _pre_existing_loggers:
            if _name.startswith("roll") or _name.startswith("log_rank_"):
                logging.getLogger(_name).disabled = False

        self.config = config
        self.render_mode = config.render_mode
        self.built_in_opponent = config.built_in_opponent
        self.opponent_player = config.opponent_player
        self.include_opponent_turn = config.include_opponent_turn

        BaseLanguageBasedEnv.__init__(self)

        game_registry = GameRegistry.from_directories_and_cwd_files()
        game_specs = game_registry.get_game_specs_that_unify_with(config.game_name)
        if not game_specs:
            raise ValueError(
                f"No clembench game named '{config.game_name}' could be found. "
                "Make sure clembench is cloned under the current working directory "
                "(e.g. MARSHAL/clembench/) or CLEMBENCH_HOME is set."
            )
        self.game_spec = game_specs[0]
        self.num_players = self.game_spec.players

        instances_filter = None
        if config.game_instance_split is not None:
            from datasets import load_dataset

            dataset = load_dataset("colab-potsdam/playpen-data", "instances", split=config.game_instance_split)
            instances_filter = to_instance_filter(dataset)

        self._game_instances = list(GameInstances.from_game_spec(self.game_spec).filter(instances_filter))
        if not self._game_instances:
            raise ValueError(f"No game instances available for '{config.game_name}' after filtering.")

        pz_env = clemcore_env(config.game_name, instances_filter=instances_filter, single_pass=config.single_pass)
        self._pz_env = AgentControlWrapper(
            pz_env, {f"player_{i}": "learner" for i in range(self.num_players)}
        )
        # Cached full first-turn context (rules + first game state, as built by clemgame's own
        # GameMaster) -- used as the fixed "prefix" prompt by get_prompt(). Set on each reset().
        self._initial_context: Optional[str] = None

    @property
    def current_player(self) -> int:
        agent_id = self._pz_env.agent_selection
        if agent_id is None:
            return 0
        return int(agent_id.split("_")[1])

    def reset(self, seed: Optional[int] = 0):
        try:
            with all_seed(seed):
                row = self._game_instances[seed % len(self._game_instances)]
                game_id = row["game_instance"]["game_id"]
                self._pz_env.reset(seed=seed, options={"game_id": game_id})
                observation, _reward, _done, _truncated, _info = self._pz_env.last()
                self._initial_context = observation["content"]
                initial_observation = {
                    # Already fully included in get_prompt()'s prefix (rules + first game
                    # state come bundled together from clemgame) -- left empty here so
                    # EnvManager's per-turn "GAME STATE" block doesn't duplicate it.
                    "observation": "",
                    "legal_actions": {},
                }
                # No built-in opponent to pre-play a turn, so there is never a preceding
                # execute_results batch here, matching hanabi/tictactoe's self-play path.
                return initial_observation, []
        except (RuntimeError, RuntimeWarning) as e:
            next_seed = abs(hash(str(seed))) % (2**32) if seed is not None else 0
            return self.reset(next_seed)

    def step(self, action):
        from clemcore.clemgame.master import Outcome

        action_str = action if isinstance(action, str) else str(action)
        current_player = self.current_player

        self._pz_env.step(action_str)
        unwrapped = self._pz_env.unwrapped

        done = all(
            unwrapped.terminations.get(agent_id, False) or unwrapped.truncations.get(agent_id, False)
            for agent_id in unwrapped.possible_agents
        )
        # Each role's reward accrued since that role was last observed -- reflects exactly
        # what GameMasterEnv.step() computed for this transition (only the acting role's
        # entry is freshly set on a non-terminal step; on the terminal step every role's
        # entry is set to the same shared outcome-based reward).
        rewards = [unwrapped._cumulative_rewards.get(f"player_{i}", 0.0) for i in range(self.num_players)]
        info = dict(unwrapped.infos.get(f"player_{current_player}", {}))

        next_agent_id = self._pz_env.agent_selection
        if done:
            state = unwrapped.game_master.state
            success = state.outcome == Outcome.SUCCESS
            info["success"] = success
            if not success:
                # clemgame's own reason for ending the episode (e.g. Taboo's clue_error, a
                # rule violation, vs. a plain parse failure) lives on the game-specific
                # GameState subclass, not in the lightweight `info` dict GameMaster.step()
                # returns -- that dict is only ever populated via log_to_self(), which is a
                # no-op unless a transcript-recording callback is attached (not done here).
                # Log it directly so the reason is visible in this worker's own log file
                # instead of only showing up as a bare "success: 0.0" metric.
                state_fields = {
                    k: _summarize_for_log(v)
                    for k, v in vars(state).items()
                    if k not in ("outcome", "describer_initial_prompt", "guesser_initial_prompt")
                }
                get_logger().info(
                    f"PlaypenEnv[{self.config.game_name}] episode ended without success "
                    f"(outcome={state.outcome}): {state_fields}"
                )
            observation_text = None
            next_player = None
            legal_actions = None
            self._pz_env.step(None)  # AEC dead-agent cleanup, mirrors clemcore's own AECToGymWrapper
        else:
            observation_text = self._pz_env.observe(next_agent_id)["content"]
            next_player = int(next_agent_id.split("_")[1])
            legal_actions = {}

        return [
            {
                "current_player": current_player,
                "action": action_str,
                "rewards": rewards,
                "done": done,
                "info": info,
                "next_player": next_player,
                "observation": observation_text,
                "legal_actions": legal_actions,
            }
        ]

    def get_prompt(self, mode: str = "prefix", think: bool = True, player_id: int = 0) -> Dict[str, str]:
        """Returns clemgame's own initial context verbatim and nothing else.

        No MARSHAL-imposed wrapper (e.g. an <answer></answer> tag instruction): EnvManager's
        _format_messages/_parse_response no longer require or force one for
        BaseLanguageBasedEnv envs (see roll/agentic/rollout/env_manager.py), so the prompt
        contains only what clemgame itself provides -- rules, game state, and the game's own
        response-format instructions (e.g. "CLUE: ..."/"GUESS: ...").
        """
        if mode != "prefix":
            raise ValueError(f"Invalid prompt mode: {mode}")
        return {"system": "", "user": self._initial_context or ""}

    def get_all_actions(self) -> Dict[int, str]:
        # Free-text action space: there is no enumerable legal-actions set.
        return {}

    def get_losing_state(self, player_id: int = 0, overlong_response: bool = False, overlong_sequence: bool = False):
        rewards = [0.0] * self.num_players
        rewards[player_id] = -1 - 10
        info: Dict[str, Any] = {
            "success": False,
            f"player_{player_id}_lose_for_wrong_format": 1,
            f"player_{player_id}_lose_for_overlong_response": 1 if overlong_response else 0,
            f"player_{player_id}_lose_for_overlong_sequence": 1 if overlong_sequence else 0,
        }
        return [
            {
                "current_player": player_id,
                "action": "",
                "rewards": rewards,
                "done": True,
                "info": info,
                "next_player": None,
                "observation": None,
                "legal_actions": None,
            }
        ]

    def render(self, mode: str = "text"):
        if mode == "text":
            return self._initial_context
        return None  # no image rendering for clembench text games

    def close(self):
        if hasattr(self, "_pz_env") and self._pz_env is not None:
            self._pz_env.close()


if __name__ == "__main__":
    # Basic unit test -- run from the MARSHAL repo root so clembench/ is discoverable, e.g.:
    #   .venv/bin/python roll/agentic/env/playpen/env.py
    import random

    print("-" * 100)
    print("Basic unit test (wordle):")
    print("-" * 100)
    env = PlaypenEnv(PlaypenEnvConfig(game_name="wordle"))

    garbage_responses = ["I like turtles", "Guess: zzzzz"]
    valid_but_wrong_guesses = ["explanation: guessing\nguess: apple", "explanation: guessing\nguess: beach"]
    for i in range(7):
        print("-" * 100)
        print(f"Episode {i}")
        print("-" * 100)
        initial_observation, execute_results = env.reset(seed=i)
        prefix_prompt = env.get_prompt(mode="prefix")
        print(f"System prompt:\n{prefix_prompt['system']}")
        print(f"User prompt:\n{prefix_prompt['user'][:500]}...")

        # Alternate between a malformed-response episode (exercises the abort path) and a
        # well-formed-but-wrong-guess episode (exercises the multi-turn non-terminal path).
        responses = garbage_responses if i % 2 == 0 else valid_but_wrong_guesses
        done = False
        turn = 0
        while not done and turn < 10:
            action = random.choice(responses)
            print(f"Player {env.current_player} taking action: {action!r}")
            execute_result = env.step(action)
            rewards = execute_result[-1]["rewards"]
            done = execute_result[-1]["done"]
            info = execute_result[-1]["info"]
            observation = execute_result[-1]["observation"]
            print(f"rewards: {rewards}, done: {done}, info: {info}")
            if observation is not None:
                print(f"next observation: {observation[:200]}")
            turn += 1
    env.close()
    print("-" * 100)
    print("Smoke test completed without exceptions.")
    print("-" * 100)
