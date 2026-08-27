"""Warm, asset-free pixel renderer for Overcooked states.

The renderer intentionally draws at a 32 px logical tile size and only enlarges
the completed frame by an integer factor.  This keeps edges crisp, avoids asset
loading failures, and makes the class suitable for both the desktop UI and a
future streamed/web build.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pygame


Color = tuple[int, int, int]


def _safe_attr(value: object, name: str, default: Any = None) -> Any:
    """Read a possibly-computed state attribute without breaking rendering."""

    try:
        return getattr(value, name, default)
    except (AttributeError, TypeError, ValueError):
        return default


def _as_ingredient_name(value: object) -> str:
    return str(_safe_attr(value, "name", value)).lower()


class PixelKitchenVisualizer:
    """Programmatic pixel-art replacement for ``StateVisualizer``.

    ``render_state`` accepts the keyword shape used by the game:
    ``render_state(state=..., hud_data=..., grid=...)``.  It also recognises the
    legacy positional order ``(state, grid, hud_data)`` for easier adoption.
    """

    LOGICAL_TILE_SIZE = 32
    UNSCALED_TILE_SIZE = LOGICAL_TILE_SIZE
    HUD_HEIGHT = 68
    # 304 px lets compact 5-wide kitchens use a crisp 3x scale inside the
    # 950 px game panel (912 px rendered) instead of dropping to a tiny 2x.
    # Wider maps still derive their width directly from the 32 px tile grid.
    MIN_HUD_WIDTH = 304
    DEFAULT_MAX_SIZE = (950, 700)

    PLAYER_COLORS: tuple[Color, ...] = (
        (63, 139, 214),
        (70, 176, 112),
        (221, 119, 88),
        (164, 112, 202),
    )
    INGREDIENT_COLORS: dict[str, Color] = {
        "onion": (245, 211, 105),
        "tomato": (224, 75, 59),
    }
    STATION_COLORS: dict[str, Color] = {
        "X": (139, 82, 50),
        "O": (222, 174, 69),
        "T": (202, 68, 53),
        "P": (79, 92, 107),
        "D": (117, 179, 208),
        "S": (224, 123, 62),
    }
    SUPPORTED_TILES = frozenset((" ", "X", "O", "T", "P", "D", "S"))

    def __init__(
        self,
        *,
        tile_size: int = LOGICAL_TILE_SIZE,
        max_size: tuple[int, int] = DEFAULT_MAX_SIZE,
        pixel_scale: int | None = None,
        width: int | None = None,
        height: int | None = None,
        grid: Sequence[Sequence[str]] | None = None,
        is_rendering_hud: bool = True,
        is_rendering_cooking_timer: bool = True,
        is_rendering_action_probs: bool = True,
        player_colors: Sequence[Color] | None = None,
        **_: object,
    ) -> None:
        if int(tile_size) != self.LOGICAL_TILE_SIZE:
            raise ValueError("PixelKitchenVisualizer uses a fixed 32 px logical tile")
        if len(max_size) != 2 or min(int(v) for v in max_size) <= 0:
            raise ValueError("max_size must contain two positive integers")
        if pixel_scale is not None and int(pixel_scale) < 1:
            raise ValueError("pixel_scale must be a positive integer")

        pygame.font.init()
        self.tile_size = self.LOGICAL_TILE_SIZE
        self.max_size = (int(max_size[0]), int(max_size[1]))
        self.pixel_scale = int(pixel_scale) if pixel_scale is not None else None
        self.width = int(width) if width is not None else None
        self.height = int(height) if height is not None else None
        self.grid = grid
        self.is_rendering_hud = bool(is_rendering_hud)
        self.is_rendering_cooking_timer = bool(is_rendering_cooking_timer)
        self.is_rendering_action_probs = bool(is_rendering_action_probs)
        self.player_colors = tuple(player_colors or self.PLAYER_COLORS)
        self.last_pixel_scale = 1
        self.last_logical_size = (0, 0)
        self._fonts: dict[int, pygame.font.Font] = {}

    @staticmethod
    def default_hud_data(state: object, **kwargs: object) -> dict[str, object]:
        """Build the same core HUD payload as the upstream visualizer."""

        def recipes(name: str) -> list[object]:
            result: list[object] = []
            for recipe in _safe_attr(state, name, ()) or ():
                to_dict = _safe_attr(recipe, "to_dict")
                if callable(to_dict):
                    result.append(to_dict())
                elif isinstance(recipe, Mapping):
                    result.append(copy.deepcopy(dict(recipe)))
                else:
                    ingredients = _safe_attr(recipe, "ingredients", ())
                    result.append({"ingredients": list(ingredients or ())})
            return result

        payload: dict[str, object] = {
            "timestep": _safe_attr(state, "timestep", 0),
            "all_orders": recipes("all_orders"),
            "bonus_orders": recipes("bonus_orders"),
        }
        payload.update(copy.deepcopy(kwargs))
        return payload

    @classmethod
    def default_hud_data_from_trajectories(
        cls,
        trajectories: Mapping[str, object],
        trajectory_idx: int = 0,
    ) -> list[dict[str, object]]:
        """Compatibility helper for callers that render saved trajectories."""

        states = trajectories["ep_states"][trajectory_idx]  # type: ignore[index]
        rewards_by_trajectory = trajectories.get("ep_rewards", ())
        try:
            rewards = rewards_by_trajectory[trajectory_idx]  # type: ignore[index]
        except (IndexError, KeyError, TypeError):
            rewards = ()
        total = 0.0
        result: list[dict[str, object]] = []
        for index, state in enumerate(states):
            if index < len(rewards):
                total += float(rewards[index])
            result.append(cls.default_hud_data(state, score=total))
        return result

    def render_state(
        self,
        state: object,
        hud_data: Mapping[str, object] | Sequence[Sequence[str]] | None = None,
        grid: Sequence[Sequence[str]] | Mapping[str, object] | None = None,
        action_probs: Sequence[Sequence[float] | None] | None = None,
    ) -> pygame.Surface:
        """Render one state to a crisp integer-scaled ``pygame.Surface``."""

        # Support StateVisualizer's old positional order: (state, grid, hud).
        if self._looks_like_grid(hud_data) and isinstance(grid, Mapping):
            hud_data, grid = grid, hud_data
        elif grid is None and self._looks_like_grid(hud_data):
            grid, hud_data = hud_data, None

        selected_grid = grid if self._looks_like_grid(grid) else self.grid
        rows = self._normalise_grid(selected_grid)
        columns = len(rows[0])
        board_width = columns * self.LOGICAL_TILE_SIZE
        board_height = len(rows) * self.LOGICAL_TILE_SIZE

        render_hud = self.is_rendering_hud and hud_data is not False
        if render_hud and hud_data is None:
            hud_data = self.default_hud_data(state)
        logical_width = max(
            board_width,
            self.MIN_HUD_WIDTH if render_hud else board_width,
        )
        hud_height = self.HUD_HEIGHT if render_hud else 0
        logical_height = board_height + hud_height
        self.last_logical_size = (logical_width, logical_height)

        logical = pygame.Surface((logical_width, logical_height))
        logical.fill((38, 27, 30))
        if render_hud:
            self._draw_hud(logical, dict(hud_data or {}), logical_width)

        board_x = (logical_width - board_width) // 2
        board_origin = (board_x, hud_height)
        self._draw_board(logical, rows, board_origin)
        self._draw_objects(logical, state, rows, board_origin)
        self._draw_players(logical, state, board_origin)
        if self.is_rendering_action_probs and action_probs is not None:
            self._draw_action_probs(logical, state, action_probs, board_origin)

        limit_width = self.width or self.max_size[0]
        limit_height = self.height or self.max_size[1]
        auto_scale = min(
            max(1, limit_width // logical_width),
            max(1, limit_height // logical_height),
        )
        scale = self.pixel_scale or auto_scale
        if scale * logical_width > limit_width or scale * logical_height > limit_height:
            scale = auto_scale
        self.last_pixel_scale = max(1, int(scale))

        if self.last_pixel_scale == 1:
            rendered = logical
        else:
            rendered = pygame.transform.scale(
                logical,
                (
                    logical_width * self.last_pixel_scale,
                    logical_height * self.last_pixel_scale,
                ),
            )

        if self.width is None and self.height is None:
            return rendered

        canvas_size = (
            self.width or rendered.get_width(),
            self.height or rendered.get_height(),
        )
        canvas = pygame.Surface(canvas_size)
        canvas.fill((27, 20, 25))
        canvas.blit(rendered, rendered.get_rect(center=canvas.get_rect().center))
        return canvas

    @staticmethod
    def _looks_like_grid(value: object) -> bool:
        if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
            return False
        if not value:
            return False
        first = value[0]
        return isinstance(first, (str, Sequence)) and not isinstance(first, Mapping)

    @classmethod
    def _normalise_grid(
        cls,
        grid: Sequence[Sequence[str]] | None,
    ) -> tuple[tuple[str, ...], ...]:
        if not grid:
            raise ValueError("grid is required and cannot be empty")
        width = max(len(row) for row in grid)
        if width == 0:
            raise ValueError("grid rows cannot be empty")
        rows: list[tuple[str, ...]] = []
        for row in grid:
            cells = [str(cell)[:1] if str(cell) else " " for cell in row]
            cells.extend(" " for _ in range(width - len(cells)))
            rows.append(tuple(cells))
        return tuple(rows)

    def _font(self, size: int) -> pygame.font.Font:
        font = self._fonts.get(size)
        if font is None:
            font = pygame.font.Font(None, size)
            self._fonts[size] = font
        return font

    def _text(
        self,
        surface: pygame.Surface,
        text: object,
        position: tuple[int, int],
        *,
        size: int = 10,
        color: Color = (250, 238, 213),
        center: bool = False,
    ) -> pygame.Rect:
        image = self._font(size).render(str(text), False, color)
        rect = image.get_rect(center=position) if center else image.get_rect(topleft=position)
        surface.blit(image, rect)
        return rect

    def _draw_hud(
        self,
        surface: pygame.Surface,
        hud: dict[str, object],
        width: int,
    ) -> None:
        rect = pygame.Rect(0, 0, width, self.HUD_HEIGHT)
        pygame.draw.rect(surface, (43, 30, 34), rect)
        pygame.draw.rect(surface, (107, 62, 46), (0, 0, width, 4))
        pygame.draw.line(surface, (217, 148, 69), (0, self.HUD_HEIGHT - 2), (width, self.HUD_HEIGHT - 2), 2)
        self._text(surface, "KITCHEN", (8, 8), size=13, color=(255, 194, 80))
        self._text(surface, "ORDERS", (8, 24), size=9, color=(188, 156, 132))

        score_width = 69
        score_rect = pygame.Rect(width - score_width - 7, 8, score_width, 38)
        pygame.draw.rect(surface, (64, 43, 43), score_rect)
        pygame.draw.rect(surface, (235, 171, 73), score_rect, 1)
        self._text(surface, "SCORE", (score_rect.centerx, 14), size=9, color=(221, 181, 126), center=True)
        score = hud.get("score", 0)
        try:
            score_text = f"{float(score):.0f}"
        except (TypeError, ValueError):
            score_text = str(score)
        self._text(surface, score_text, (score_rect.centerx, 31), size=18, color=(255, 241, 188), center=True)

        orders = hud.get("all_orders") or hud.get("orders") or ()
        order_x = 48
        order_y = 17
        order_width = 39
        available_right = score_rect.left - 4
        rendered_orders = 0
        if isinstance(orders, Iterable) and not isinstance(orders, (str, bytes, Mapping)):
            order_list = list(orders)
            for order in order_list:
                if order_x + order_width > available_right:
                    break
                self._draw_order_card(surface, pygame.Rect(order_x, order_y, order_width, 30), order)
                rendered_orders += 1
                order_x += order_width + 4
            if rendered_orders < len(order_list) and order_x + 15 <= available_right:
                self._text(surface, f"+{len(order_list) - rendered_orders}", (order_x, 28), size=10)
        if rendered_orders == 0:
            self._text(surface, "FREE PLAY", (49, 28), size=10, color=(151, 197, 153))

        info_y = 51
        step = hud.get("timestep", hud.get("step", 0))
        self._text(surface, f"STEP {step}", (8, info_y), size=9, color=(207, 188, 169))
        if "time_left" in hud:
            self._text(surface, f"TIME {hud['time_left']}", (77, info_y), size=9, color=(255, 201, 104))
        bonus = hud.get("bonus_orders") or ()
        try:
            bonus_count = len(bonus)  # type: ignore[arg-type]
        except TypeError:
            bonus_count = 0
        if bonus_count:
            self._text(surface, f"BONUS x{bonus_count}", (145, info_y), size=9, color=(255, 194, 80))

    def _draw_order_card(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        order: object,
    ) -> None:
        pygame.draw.rect(surface, (88, 55, 47), rect)
        pygame.draw.rect(surface, (174, 105, 61), rect, 1)
        pygame.draw.ellipse(surface, (231, 218, 190), (rect.x + 4, rect.y + 8, 18, 13))
        pygame.draw.ellipse(surface, (104, 76, 67), (rect.x + 6, rect.y + 10, 14, 8))
        ingredients = self._ingredients(order)
        for index, ingredient in enumerate(ingredients[:3]):
            color = self.INGREDIENT_COLORS.get(ingredient, (194, 150, 103))
            pygame.draw.rect(surface, color, (rect.x + 24, rect.y + 4 + index * 8, 8, 6))
            pygame.draw.rect(surface, (54, 38, 38), (rect.x + 24, rect.y + 4 + index * 8, 8, 6), 1)

    def _draw_board(
        self,
        surface: pygame.Surface,
        rows: tuple[tuple[str, ...], ...],
        origin: tuple[int, int],
    ) -> None:
        ox, oy = origin
        width = len(rows[0]) * self.LOGICAL_TILE_SIZE
        height = len(rows) * self.LOGICAL_TILE_SIZE
        pygame.draw.rect(surface, (25, 18, 22), (ox - 2, oy, width + 4, height + 2))
        for y, row in enumerate(rows):
            for x, token in enumerate(row):
                rect = pygame.Rect(
                    ox + x * self.LOGICAL_TILE_SIZE,
                    oy + y * self.LOGICAL_TILE_SIZE,
                    self.LOGICAL_TILE_SIZE,
                    self.LOGICAL_TILE_SIZE,
                )
                self._draw_floor(surface, rect, x, y)
                if token != " ":
                    self._draw_station(surface, rect, token)

    @staticmethod
    def _draw_floor(surface: pygame.Surface, rect: pygame.Rect, x: int, y: int) -> None:
        base = (203, 144, 99) if (x + y) % 2 == 0 else (194, 132, 91)
        pygame.draw.rect(surface, base, rect)
        pygame.draw.line(surface, (155, 95, 70), rect.bottomleft, rect.bottomright, 1)
        pygame.draw.line(surface, (226, 174, 123), rect.topleft, rect.topright, 1)
        if (x * 3 + y) % 4 == 0:
            pygame.draw.rect(surface, (174, 112, 80), (rect.x + 7, rect.y + 9, 2, 2))
            pygame.draw.rect(surface, (224, 165, 112), (rect.x + 24, rect.y + 22, 2, 2))

    def _draw_station(self, surface: pygame.Surface, rect: pygame.Rect, token: str) -> None:
        token = token.upper()
        if token not in self.SUPPORTED_TILES:
            token = "X"
        self._draw_counter(surface, rect)
        if token == "O":
            self._draw_dispenser(surface, rect, "onion")
        elif token == "T":
            self._draw_dispenser(surface, rect, "tomato")
        elif token == "P":
            self._draw_pot(surface, rect)
        elif token == "D":
            self._draw_dishes(surface, rect)
        elif token == "S":
            self._draw_service(surface, rect)
        # Generic counters should read as furniture, not debug tiles. Keep the
        # small letter badge only on interactive stations where it aids fast
        # recognition during play.
        if token != "X":
            self._draw_station_badge(surface, rect, token)

    @staticmethod
    def _draw_counter(surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, (91, 51, 42), (rect.x + 2, rect.y + 7, 28, 25))
        pygame.draw.rect(surface, (137, 75, 47), (rect.x + 3, rect.y + 8, 26, 21))
        pygame.draw.rect(surface, (112, 60, 44), (rect.x + 6, rect.y + 13, 20, 13))
        pygame.draw.rect(surface, (71, 40, 37), (rect.x + 7, rect.y + 14, 18, 11), 1)
        pygame.draw.rect(surface, (224, 180, 125), (rect.x + 1, rect.y + 4, 30, 7))
        pygame.draw.rect(surface, (249, 211, 153), (rect.x + 2, rect.y + 3, 28, 3))
        pygame.draw.line(surface, (151, 103, 75), (rect.x + 2, rect.y + 10), (rect.right - 2, rect.y + 10), 1)
        pygame.draw.rect(surface, (222, 151, 72), (rect.centerx - 2, rect.y + 18, 4, 2))

    def _draw_dispenser(self, surface: pygame.Surface, rect: pygame.Rect, ingredient: str) -> None:
        color = self.INGREDIENT_COLORS[ingredient]
        pygame.draw.rect(surface, (65, 47, 43), (rect.x + 6, rect.y + 5, 20, 13))
        pygame.draw.rect(surface, (113, 82, 64), (rect.x + 7, rect.y + 5, 18, 11))
        positions = ((11, 10), (17, 8), (22, 11), (15, 13))
        for px, py in positions:
            pygame.draw.circle(surface, (67, 41, 37), (rect.x + px + 1, rect.y + py + 1), 4)
            pygame.draw.circle(surface, color, (rect.x + px, rect.y + py), 4)
        if ingredient == "onion":
            pygame.draw.rect(surface, (245, 236, 190), (rect.x + 15, rect.y + 6, 2, 3))
        else:
            pygame.draw.rect(surface, (79, 145, 67), (rect.x + 15, rect.y + 5, 4, 2))

    @staticmethod
    def _draw_pot(surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, (54, 48, 50), (rect.x + 5, rect.y + 6, 22, 4))
        pygame.draw.rect(surface, (38, 42, 48), (rect.x + 7, rect.y + 7, 18, 12))
        pygame.draw.rect(surface, (94, 106, 111), (rect.x + 8, rect.y + 9, 16, 8))
        pygame.draw.rect(surface, (31, 34, 40), (rect.x + 4, rect.y + 10, 4, 3))
        pygame.draw.rect(surface, (31, 34, 40), (rect.x + 24, rect.y + 10, 4, 3))
        pygame.draw.rect(surface, (224, 104, 53), (rect.x + 9, rect.y + 20, 14, 2))

    @staticmethod
    def _draw_dishes(surface: pygame.Surface, rect: pygame.Rect) -> None:
        for offset in (4, 2, 0):
            pygame.draw.ellipse(surface, (58, 65, 72), (rect.x + 7, rect.y + 8 + offset, 19, 7))
            pygame.draw.ellipse(surface, (213, 236, 233), (rect.x + 7, rect.y + 7 + offset, 18, 6))
            pygame.draw.ellipse(surface, (130, 184, 200), (rect.x + 11, rect.y + 9 + offset, 10, 2))

    @staticmethod
    def _draw_service(surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, (105, 45, 38), (rect.x + 4, rect.y + 5, 24, 15))
        pygame.draw.rect(surface, (236, 129, 56), (rect.x + 5, rect.y + 6, 22, 12))
        pygame.draw.rect(surface, (255, 208, 105), (rect.x + 7, rect.y + 8, 18, 3))
        pygame.draw.circle(surface, (242, 218, 146), (rect.centerx, rect.y + 15), 4)
        pygame.draw.rect(surface, (91, 54, 45), (rect.centerx - 5, rect.y + 18, 10, 2))

    def _draw_station_badge(self, surface: pygame.Surface, rect: pygame.Rect, token: str) -> None:
        color = self.STATION_COLORS.get(token, self.STATION_COLORS["X"])
        badge = pygame.Rect(rect.x + 2, rect.bottom - 10, 10, 8)
        pygame.draw.rect(surface, (48, 31, 31), badge)
        pygame.draw.rect(surface, color, badge, 1)
        self._text(surface, token, (badge.centerx, badge.centery), size=9, color=(255, 245, 220), center=True)

    def _draw_objects(
        self,
        surface: pygame.Surface,
        state: object,
        rows: tuple[tuple[str, ...], ...],
        origin: tuple[int, int],
    ) -> None:
        objects = _safe_attr(state, "objects", {}) or {}
        values = objects.values() if isinstance(objects, Mapping) else objects
        for obj in values:
            position = _safe_attr(obj, "position")
            if not self._valid_position(position, rows):
                continue
            x, y = int(position[0]), int(position[1])
            center = (
                origin[0] + x * self.LOGICAL_TILE_SIZE + 16,
                origin[1] + y * self.LOGICAL_TILE_SIZE + 16,
            )
            token = rows[y][x].upper()
            self._draw_item(surface, center, obj, in_pot=token == "P")
            if token == "P" and str(_safe_attr(obj, "name", "")).lower() == "soup":
                self._draw_cooking_progress(surface, obj, center)

    @staticmethod
    def _valid_position(position: object, rows: tuple[tuple[str, ...], ...]) -> bool:
        if not isinstance(position, Sequence) or len(position) < 2:
            return False
        try:
            x, y = int(position[0]), int(position[1])
        except (TypeError, ValueError):
            return False
        return 0 <= y < len(rows) and 0 <= x < len(rows[0])

    def _draw_item(
        self,
        surface: pygame.Surface,
        center: tuple[int, int],
        obj: object,
        *,
        in_pot: bool = False,
        mini: bool = False,
    ) -> None:
        name = str(_safe_attr(obj, "name", obj)).lower()
        cx, cy = center
        radius = 4 if mini else 6
        if name == "onion":
            pygame.draw.circle(surface, (62, 39, 37), (cx + 1, cy + 2), radius + 1)
            pygame.draw.circle(surface, self.INGREDIENT_COLORS["onion"], (cx, cy), radius)
            pygame.draw.rect(surface, (251, 238, 176), (cx - 1, cy - radius - 2, 2, 3))
        elif name == "tomato":
            pygame.draw.circle(surface, (75, 38, 35), (cx + 1, cy + 2), radius + 1)
            pygame.draw.circle(surface, self.INGREDIENT_COLORS["tomato"], (cx, cy), radius)
            pygame.draw.rect(surface, (75, 141, 64), (cx - 3, cy - radius - 1, 6, 2))
        elif name == "dish":
            pygame.draw.ellipse(surface, (57, 57, 61), (cx - 8, cy - 3, 17, 8))
            pygame.draw.ellipse(surface, (219, 239, 235), (cx - 8, cy - 5, 16, 8))
            pygame.draw.ellipse(surface, (125, 179, 198), (cx - 4, cy - 3, 8, 3))
        elif name == "soup":
            bowl_y = cy - 1 if in_pot else cy + 1
            pygame.draw.ellipse(surface, (52, 42, 44), (cx - 10, bowl_y - 6, 20, 12))
            pygame.draw.rect(surface, (220, 224, 207), (cx - 9, bowl_y - 1, 18, 7))
            pygame.draw.ellipse(surface, (246, 235, 196), (cx - 9, bowl_y - 6, 18, 9))
            pygame.draw.ellipse(surface, (174, 84, 48), (cx - 7, bowl_y - 4, 14, 6))
            for index, ingredient in enumerate(self._ingredients(obj)[:3]):
                color = self.INGREDIENT_COLORS.get(ingredient, (224, 164, 77))
                pygame.draw.rect(surface, color, (cx - 5 + index * 4, bowl_y - 3, 3, 3))
        else:
            pygame.draw.rect(surface, (250, 211, 111), (cx - radius, cy - radius, radius * 2, radius * 2))
            pygame.draw.rect(surface, (74, 48, 43), (cx - radius, cy - radius, radius * 2, radius * 2), 1)

    @staticmethod
    def _ingredients(value: object) -> list[str]:
        if isinstance(value, Mapping):
            raw = value.get("ingredients", ())
        else:
            raw = _safe_attr(value, "ingredients", ())
        return [_as_ingredient_name(item) for item in (raw or ())]

    def _draw_cooking_progress(
        self,
        surface: pygame.Surface,
        soup: object,
        center: tuple[int, int],
    ) -> None:
        if not self.is_rendering_cooking_timer:
            return
        tick = _safe_attr(soup, "_cooking_tick", _safe_attr(soup, "cooking_tick", -1))
        cook_time = _safe_attr(soup, "_cook_time")
        if cook_time is None:
            cook_time = _safe_attr(soup, "cook_time", 20)
        try:
            tick_value = float(tick)
            cook_value = max(1.0, float(cook_time))
        except (TypeError, ValueError):
            tick_value, cook_value = -1.0, 20.0
        ready = bool(_safe_attr(soup, "is_ready", tick_value >= cook_value))
        x, y = center[0] - 12, center[1] + 10
        pygame.draw.rect(surface, (47, 36, 38), (x, y, 24, 5))
        if ready:
            progress = 1.0
            color = (101, 218, 128)
        elif tick_value < 0:
            progress = min(1.0, len(self._ingredients(soup)) / 3.0)
            color = (221, 166, 74)
        else:
            progress = max(0.0, min(1.0, tick_value / cook_value))
            color = (240, 125, 57)
        pygame.draw.rect(surface, color, (x + 1, y + 1, int(22 * progress), 3))
        if ready:
            pygame.draw.rect(surface, (255, 242, 155), (center[0] + 9, center[1] - 9, 2, 5))
            pygame.draw.rect(surface, (255, 242, 155), (center[0] + 7, center[1] - 7, 6, 2))

    def _draw_players(
        self,
        surface: pygame.Surface,
        state: object,
        origin: tuple[int, int],
    ) -> None:
        for index, player in enumerate(_safe_attr(state, "players", ()) or ()):
            position = _safe_attr(player, "position")
            if not isinstance(position, Sequence) or len(position) < 2:
                continue
            center = (
                origin[0] + int(position[0]) * self.LOGICAL_TILE_SIZE + 16,
                origin[1] + int(position[1]) * self.LOGICAL_TILE_SIZE + 16,
            )
            color = self.player_colors[index % len(self.player_colors)]
            orientation = tuple(_safe_attr(player, "orientation", (0, 1)))
            self._draw_chef(surface, center, color, index, orientation)
            held = _safe_attr(player, "held_object")
            if held is not None:
                dx, dy = self._orientation_offset(orientation, 9)
                hand_center = (center[0] + dx, center[1] + dy + 2)
                pygame.draw.circle(surface, (236, 177, 127), (hand_center[0] - 4, hand_center[1] + 3), 2)
                pygame.draw.circle(surface, (236, 177, 127), (hand_center[0] + 4, hand_center[1] + 3), 2)
                self._draw_item(surface, hand_center, held, mini=True)

    def _draw_chef(
        self,
        surface: pygame.Surface,
        center: tuple[int, int],
        color: Color,
        index: int,
        orientation: tuple[int, ...],
    ) -> None:
        cx, cy = center
        pygame.draw.ellipse(surface, (93, 55, 48), (cx - 9, cy + 8, 18, 7))
        pygame.draw.rect(surface, (43, 45, 56), (cx - 7, cy + 4, 5, 8))
        pygame.draw.rect(surface, (43, 45, 56), (cx + 2, cy + 4, 5, 8))
        pygame.draw.rect(surface, (39, 31, 35), (cx - 8, cy + 10, 6, 3))
        pygame.draw.rect(surface, (39, 31, 35), (cx + 2, cy + 10, 6, 3))
        pygame.draw.rect(surface, (47, 36, 43), (cx - 9, cy - 2, 18, 13))
        pygame.draw.rect(surface, color, (cx - 8, cy - 3, 16, 12))
        pygame.draw.rect(surface, (246, 231, 203), (cx - 4, cy, 8, 9))
        pygame.draw.rect(surface, (89, 63, 61), (cx - 1, cy + 1, 2, 6))
        pygame.draw.circle(surface, (92, 57, 52), (cx, cy - 8), 7)
        pygame.draw.circle(surface, (239, 180, 132), (cx, cy - 9), 6)
        self._draw_face(surface, center, orientation)
        # Three-block chef hat remains legible after nearest-neighbour scaling.
        pygame.draw.rect(surface, (238, 232, 213), (cx - 7, cy - 17, 14, 7))
        pygame.draw.circle(surface, (252, 247, 226), (cx - 5, cy - 17), 4)
        pygame.draw.circle(surface, (252, 247, 226), (cx, cy - 19), 5)
        pygame.draw.circle(surface, (252, 247, 226), (cx + 5, cy - 17), 4)
        badge_center = (cx + 7, cy + 6)
        pygame.draw.circle(surface, (43, 31, 35), badge_center, 4)
        self._text(surface, str(index + 1), badge_center, size=8, color=(255, 244, 218), center=True)

    @staticmethod
    def _draw_face(
        surface: pygame.Surface,
        center: tuple[int, int],
        orientation: tuple[int, ...],
    ) -> None:
        cx, cy = center
        dx, dy = PixelKitchenVisualizer._orientation_offset(orientation, 2)
        if abs(dx) >= abs(dy):
            pygame.draw.rect(surface, (54, 42, 44), (cx + dx - 1, cy - 11, 2, 2))
        elif dy >= 0:
            pygame.draw.rect(surface, (54, 42, 44), (cx - 3, cy - 10, 2, 2))
            pygame.draw.rect(surface, (54, 42, 44), (cx + 2, cy - 10, 2, 2))

    @staticmethod
    def _orientation_offset(orientation: tuple[int, ...], amount: int) -> tuple[int, int]:
        if len(orientation) < 2:
            return (0, amount)
        try:
            return (int(orientation[0]) * amount, int(orientation[1]) * amount)
        except (TypeError, ValueError):
            return (0, amount)

    def _draw_action_probs(
        self,
        surface: pygame.Surface,
        state: object,
        action_probs: Sequence[Sequence[float] | None],
        origin: tuple[int, int],
    ) -> None:
        for player, probabilities in zip(
            _safe_attr(state, "players", ()) or (), action_probs
        ):
            if probabilities is None or len(probabilities) == 0:
                continue
            position = _safe_attr(player, "position")
            if not isinstance(position, Sequence) or len(position) < 2:
                continue
            try:
                best = max(range(len(probabilities)), key=lambda i: float(probabilities[i]))
            except (TypeError, ValueError):
                continue
            cx = origin[0] + int(position[0]) * self.LOGICAL_TILE_SIZE + 16
            cy = origin[1] + int(position[1]) * self.LOGICAL_TILE_SIZE + 16
            triangles = {
                0: ((cx, cy - 15), (cx - 3, cy - 11), (cx + 3, cy - 11)),
                1: ((cx, cy + 15), (cx - 3, cy + 11), (cx + 3, cy + 11)),
                2: ((cx + 15, cy), (cx + 11, cy - 3), (cx + 11, cy + 3)),
                3: ((cx - 15, cy), (cx - 11, cy - 3), (cx - 11, cy + 3)),
            }
            if best in triangles:
                pygame.draw.polygon(surface, (255, 220, 92), triangles[best])


# A drop-in import alias is useful while callers migrate one file at a time.
StateVisualizer = PixelKitchenVisualizer


__all__ = ["PixelKitchenVisualizer", "StateVisualizer"]
