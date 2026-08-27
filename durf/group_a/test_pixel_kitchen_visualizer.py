"""Tests for the asset-free pixel kitchen renderer."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from durf.group_a.pixel_kitchen_visualizer import PixelKitchenVisualizer


class _Recipe:
    def __init__(self, *ingredients: str) -> None:
        self.ingredients = ingredients

    def to_dict(self) -> dict[str, object]:
        return {"ingredients": list(self.ingredients)}


def _player(
    position: tuple[int, int],
    orientation: tuple[int, int],
    held_object: object | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        position=position,
        orientation=orientation,
        held_object=held_object,
    )


def _state(
    *,
    players: tuple[object, ...] = (),
    objects: dict[tuple[int, int], object] | None = None,
    timestep: int = 7,
) -> SimpleNamespace:
    return SimpleNamespace(
        players=players,
        objects=objects or {},
        timestep=timestep,
        all_orders=[_Recipe("tomato", "tomato", "onion")],
        bonus_orders=[_Recipe("onion", "onion", "onion")],
    )


class PixelKitchenVisualizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pygame.font.init()

    @classmethod
    def tearDownClass(cls) -> None:
        pygame.font.quit()

    def test_default_hud_data_matches_state_visualizer_shape(self) -> None:
        state = _state(timestep=19)
        hud = PixelKitchenVisualizer.default_hud_data(state, score=42.5)

        self.assertEqual(hud["timestep"], 19)
        self.assertEqual(
            hud["all_orders"],
            [{"ingredients": ["tomato", "tomato", "onion"]}],
        )
        self.assertEqual(
            hud["bonus_orders"],
            [{"ingredients": ["onion", "onion", "onion"]}],
        )
        self.assertEqual(hud["score"], 42.5)

    def test_ten_by_six_map_uses_integer_nearest_neighbour_scale(self) -> None:
        grid = (
            "XXXXPXXXXX",
            "D        X",
            "X XXXXXX X",
            "X XXXXXX S",
            "T        X",
            "XXXXOXXXXX",
        )
        visualizer = PixelKitchenVisualizer()
        hud = visualizer.default_hud_data(_state(), score=12)

        with patch.object(
            pygame.transform,
            "smoothscale",
            side_effect=AssertionError("smoothscale must not be used"),
        ):
            surface = visualizer.render_state(
                state=_state(),
                hud_data=hud,
                grid=grid,
            )

        self.assertEqual(visualizer.tile_size, 32)
        self.assertEqual(visualizer.last_pixel_scale, 2)
        self.assertEqual(surface.get_size(), (640, 520))
        self.assertLessEqual(surface.get_width(), 950)
        self.assertLessEqual(surface.get_height(), 700)

    def test_each_station_has_a_distinct_pixel_sprite(self) -> None:
        grid = ("XOTPD S",)
        visualizer = PixelKitchenVisualizer(
            pixel_scale=1,
            is_rendering_hud=False,
        )
        surface = visualizer.render_state(_state(), grid=grid)

        station_indexes = (0, 1, 2, 3, 4, 6)
        tile_images = {
            pygame.image.tostring(
                surface.subsurface((index * 32, 0, 32, 32)),
                "RGB",
            )
            for index in station_indexes
        }
        self.assertEqual(len(tile_images), 6)
        self.assertEqual(len(set(visualizer.STATION_COLORS.values())), 6)

    def test_players_held_items_objects_and_cooking_progress_are_drawn(self) -> None:
        grid = ("XXXXX", "O P T", "D   S", "XXXXX")
        soup = SimpleNamespace(
            name="soup",
            position=(2, 1),
            ingredients=("onion", "tomato", "onion"),
            _cooking_tick=10,
            _cook_time=20,
            is_ready=False,
        )
        onion = SimpleNamespace(name="onion", position=(1, 2))
        held_tomato = SimpleNamespace(name="tomato", position=(1, 1))
        rich_state = _state(
            players=(
                _player((1, 1), (1, 0), held_tomato),
                _player((3, 2), (-1, 0)),
            ),
            objects={(2, 1): soup, (1, 2): onion},
        )
        visualizer = PixelKitchenVisualizer(pixel_scale=1, is_rendering_hud=False)

        empty = visualizer.render_state(_state(), grid=grid)
        populated = visualizer.render_state(rich_state, grid=grid)

        self.assertNotEqual(
            pygame.image.tostring(empty, "RGB"),
            pygame.image.tostring(populated, "RGB"),
        )
        # Sample the left side of each jacket rather than the shared white apron.
        player_one = populated.get_at((1 * 32 + 10, 1 * 32 + 16))[:3]
        player_two = populated.get_at((3 * 32 + 10, 2 * 32 + 16))[:3]
        self.assertNotEqual(player_one, player_two)

    def test_legacy_positional_render_order_is_supported(self) -> None:
        grid = ("XPX", "O S")
        state = _state()
        hud = PixelKitchenVisualizer.default_hud_data(state, score=3)
        visualizer = PixelKitchenVisualizer(pixel_scale=1)

        keyword_surface = visualizer.render_state(
            state=state,
            hud_data=hud,
            grid=grid,
        )
        positional_surface = visualizer.render_state(state, grid, hud)

        self.assertEqual(keyword_surface.get_size(), positional_surface.get_size())
        self.assertEqual(
            pygame.image.tostring(keyword_surface, "RGB"),
            pygame.image.tostring(positional_surface, "RGB"),
        )


if __name__ == "__main__":
    unittest.main()
