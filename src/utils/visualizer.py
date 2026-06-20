import sys
from src.enums.Colors import Colors


class GenerationVisualizer:
    """Visualizes the token-by-token JSON generation process in the terminal.

    Displays a live updating view of each decoding step, including the
    current token, allowed and blocked token counts, and a progress bar
    reflecting the length of the generated JSON. Can be disabled to
    suppress all output.

    Attributes:
        enabled (bool): Whether visualization output is active.
        step (int): The current decoding step count.
        blocked (int): The cumulative number of blocked tokens so far.
        allowed (int): The cumulative number of allowed tokens so far.
    """
    def __init__(self, enabled: bool = True) -> None:
        """Initializes the visualizer with optional output enabling.

        Args:
            enabled (bool): Whether to display generation output.
                Defaults to True.
        """
        self.enabled = enabled
        self.step = 0
        self.blocked = 0
        self.allowed = 0

    def reset(self, prompt: str) -> None:
        """Resets the visualizer state and prints the generation header.

        Clears all step and token counters and, if enabled, prints a
        formatted header displaying the first 50 characters of the prompt.

        Args:
            prompt (str): The user prompt being processed, shown
                truncated in the header.

        Returns:
            None
        """
        self.step = 0
        self.blocked = 0
        self.allowed = 0
        if not self.enabled:
            return
        print(f"\n{'─' * 60}")
        print(
            f"  {Colors.YELLOW.value}Generating for:"
            f"{Colors.RESET.value} '{prompt[:50]}...'"
        )
        print(f"{'─' * 60}")

    def update(
        self,
        current_json: str,
        next_token: str,
        blocked: int,
        allowed: int,
    ) -> None:
        """Updates the terminal display with the latest decoding step.

        Increments the step counter, stores the latest blocked and allowed
        counts, and if enabled, overwrites the previous output lines with
        a refreshed view showing the current token, constraint stats, and
        a progress bar.

        Args:
            current_json (str): The JSON string generated so far.
            next_token (str): The token decoded at the current step.
            blocked (int): The number of tokens blocked at this step.
            allowed (int): The number of tokens allowed at this step.

        Returns:
            None
        """
        self.step += 1
        self.blocked = blocked
        self.allowed = allowed
        if not self.enabled:
            return

        bar_width = 30
        ratio = min(len(current_json) / 50, 1.0)
        filled = int(bar_width * ratio)
        bar = "█" * filled + "░" * (bar_width - filled)

        sys.stdout.write("\033[F" * 4 if self.step > 1 else "")
        print(f"  Step     : {self.step}")
        print(f"  Token    : '{next_token}'")
        print(f"  Allowed  : {allowed:6d} | Blocked: {blocked:6d}")
        print(f"  JSON     : [{bar}] {current_json[:40]}")
        sys.stdout.flush()

    def finish(self, final_json: str, success: bool) -> None:
        """Prints the final generation result and status to the terminal.

        Displays a SUCCESS or FAILED status in color along with the first
        60 characters of the final generated JSON string. Does nothing if
        the visualizer is disabled.

        Args:
            final_json (str): The final generated JSON string.
            success (bool): Whether the generation was successful.

        Returns:
            None
        """
        if not self.enabled:
            return
        status = (
            f"{Colors.GREEN.value}SUCCESS{Colors.RESET.value}"
            if success
            else f"{Colors.RED.value}FAILED{Colors.RESET.value}"
        )
        print(f"\n  Result   : {status}")
        print(f"  Final    : {final_json[:60]}")
        print(f"{'─' * 60}\n")
