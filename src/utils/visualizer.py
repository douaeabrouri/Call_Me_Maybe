import sys
from src.enums.Colors import Colors


class GenerationVisualizer:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.step = 0
        self.blocked = 0
        self.allowed = 0

    def reset(self, prompt: str) -> None:
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
