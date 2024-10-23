import asyncio
import base64
import os
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypedDict
from uuid import uuid4

import pyautogui
from anthropic.types.beta import BetaToolComputerUse20241022Param
from PIL import ImageGrab

from .base import BaseAnthropicTool, ToolError, ToolResult

OUTPUT_DIR = "./outputs"

TYPING_DELAY_MS = 12
TYPING_GROUP_SIZE = 50

IMAGE_MAX_WIDTH = 1200

Action = Literal[
    "key",
    "type",
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "screenshot",
    "cursor_position",
]


class Resolution(TypedDict):
    width: int
    height: int


class ScalingSource(StrEnum):
    COMPUTER = "computer"
    API = "api"


class ComputerToolOptions(TypedDict):
    display_height_px: int
    display_width_px: int
    display_number: int | None


def chunks(s: str, chunk_size: int) -> list[str]:
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]


class ComputerTool(BaseAnthropicTool):
    """
    A tool that allows the agent to interact with the screen, keyboard, and mouse of the current computer.
    The tool parameters are defined by Anthropic and are not editable.
    """

    name: Literal["computer"] = "computer"
    api_type: Literal["computer_20241022"] = "computer_20241022"
    width: int
    height: int
    display_num: int | None

    _screenshot_delay = 2.0
    _scaling_enabled = True

    @property
    def options(self) -> ComputerToolOptions:
        width, height = self.scale_coordinates(
            ScalingSource.COMPUTER, self.width, self.height
        )
        return {
            "display_width_px": width,
            "display_height_px": height,
            "display_number": self.display_num,
        }

    def to_params(self) -> BetaToolComputerUse20241022Param:
        return {"name": self.name, "type": self.api_type, **self.options}

    def __init__(self):
        super().__init__()

        self.width = int(os.getenv("WIDTH") or 1920)
        self.height = int(os.getenv("HEIGHT") or 1080)
        assert self.width and self.height, "WIDTH, HEIGHT must be set"
        if (display_num := os.getenv("DISPLAY_NUM")) is not None:
            self.display_num = int(display_num)
        else:
            self.display_num = None

        # Set up PyAutoGUI safety net
        pyautogui.FAILSAFE = True

    async def __call__(
        self,
        *,
        action: Action,
        text: str | None = None,
        coordinate: tuple[int, int] | None = None,
        **kwargs,
    ):
        if action in ("mouse_move", "left_click_drag"):
            if coordinate is None:
                raise ToolError(f"coordinate is required for {action}")
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if not isinstance(coordinate, list) or len(coordinate) != 2:
                raise ToolError(f"{coordinate} must be a tuple of length 2")
            if not all(isinstance(i, int) and i >= 0 for i in coordinate):
                raise ToolError(f"{coordinate} must be a tuple of non-negative ints")

            x, y = self.scale_coordinates(
                ScalingSource.API, coordinate[0], coordinate[1]
            )

            if action == "mouse_move":
                pyautogui.moveTo(x, y)
                return await self.take_action_screenshot()
            elif action == "left_click_drag":
                pyautogui.dragTo(x, y)
                return await self.take_action_screenshot()

        if action in ("key", "type"):
            if text is None:
                raise ToolError(f"text is required for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")
            if not isinstance(text, str):
                raise ToolError(output=f"{text} must be a string")

            if action == "key":
                if "+" in text:
                    # Handle hotkey combinations
                    keys = [key.strip().lower() for key in text.split("+")]
                    pyautogui.hotkey(*keys)
                else:
                    pyautogui.press(text.lower())
                return await self.take_action_screenshot()
            elif action == "type":
                pyautogui.write(text, interval=0.01)
                return await self.take_action_screenshot()

        if action in (
            "left_click",
            "right_click",
            "double_click",
            "middle_click",
            "screenshot",
            "cursor_position",
        ):
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")

            if action == "screenshot":
                return await self.screenshot()
            elif action == "cursor_position":
                x, y = pyautogui.position()
                scaled_x, scaled_y = self.scale_coordinates(
                    ScalingSource.COMPUTER, int(x), int(y)
                )
                return ToolResult(output=f"X={scaled_x},Y={scaled_y}")
            elif action == "left_click":
                pyautogui.click()
                return await self.take_action_screenshot()
            elif action == "right_click":
                pyautogui.rightClick()
                return await self.take_action_screenshot()
            elif action == "middle_click":
                pyautogui.middleClick()
                return await self.take_action_screenshot()
            elif action == "double_click":
                pyautogui.doubleClick()
                return await self.take_action_screenshot()

        raise ToolError(f"Invalid action: {action}")

    async def screenshot(self):
        """Take a screenshot of the current screen and return the base64 encoded image."""
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"screenshot_{uuid4().hex}.png"

        # Take screenshot using PIL
        screenshot = ImageGrab.grab()

        # Scale if needed
        if self._scaling_enabled:
            x, y = self.scale_coordinates(
                ScalingSource.COMPUTER, self.width, self.height
            )
            screenshot = screenshot.resize((x, y))

        screenshot.save(path)

        if path.exists():
            with open(path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode()
            return ToolResult(base64_image=base64_image)
        raise ToolError("Failed to take screenshot")

    async def take_action_screenshot(self) -> ToolResult:
        """Take a screenshot after an action with appropriate delay."""
        await asyncio.sleep(self._screenshot_delay)
        screenshot_result = await self.screenshot()
        return ToolResult(base64_image=screenshot_result.base64_image)

    def scale_coordinates(self, source: ScalingSource, x: int, y: int):
        """Scale coordinates to a target maximum resolution."""
        if not self._scaling_enabled:
            return x, y
        ratio = self.width / self.height
        target_dimension = {"width": IMAGE_MAX_WIDTH, "height": IMAGE_MAX_WIDTH / ratio}
        # should be less than 1
        x_scaling_factor = target_dimension["width"] / self.width
        y_scaling_factor = target_dimension["height"] / self.height
        if source == ScalingSource.API:
            if x > self.width or y > self.height:
                raise ToolError(f"Coordinates {x}, {y} are out of bounds")
            # scale up
            return round(x / x_scaling_factor), round(y / y_scaling_factor)
        # scale down
        return round(x * x_scaling_factor), round(y * y_scaling_factor)
