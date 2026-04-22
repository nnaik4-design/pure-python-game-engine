"""
Mutation-killing tests for InputManager.
Targets surviving mutants from mutmut analysis.
"""
import pytest
from engine.input.input_manager import InputManager, InputProfile, GamepadState
from engine.math.vector2 import Vector2


class TestInputManagerMutationKillers:

    def test_gamepad_button_names_exact(self):
        """Kill mutants on button_names string values"""
        gs = GamepadState(0)
        assert gs.button_names[0] == 'a'
        assert gs.button_names[1] == 'b'
        assert gs.button_names[2] == 'x'
        assert gs.button_names[3] == 'y'
        assert gs.button_names[4] == 'left_bumper'
        assert gs.button_names[5] == 'right_bumper'
        assert gs.button_names[6] == 'back'
        assert gs.button_names[7] == 'start'
        assert gs.button_names[8] == 'left_stick_button'
        assert gs.button_names[9] == 'right_stick_button'

    def test_gamepad_initial_trigger_values(self):
        """Kill mutants on trigger default values"""
        gs = GamepadState(0)
        assert gs.left_trigger == 0.0
        assert gs.right_trigger == 0.0

    def test_gamepad_initial_dpad_values(self):
        """Kill mutants on dpad default values"""
        gs = GamepadState(0)
        assert gs.dpad_up is False
        assert gs.dpad_down is False
        assert gs.dpad_left is False
        assert gs.dpad_right is False

    def test_gamepad_initial_connected(self):
        """Kill mutant: connected default False"""
        gs = GamepadState(0)
        assert gs.connected is False

    def test_key_map_exact_values(self):
        """Kill mutants on key_map string values"""
        im = InputManager()
        assert im.key_map['Up'] == 'up'
        assert im.key_map['Down'] == 'down'
        assert im.key_map['Left'] == 'left'
        assert im.key_map['Right'] == 'right'
        assert im.key_map['w'] == 'w'
        assert im.key_map['a'] == 'a'
        assert im.key_map['s'] == 's'
        assert im.key_map['d'] == 'd'
        assert im.key_map['q'] == 'q'
        assert im.key_map['e'] == 'e'
        assert im.key_map['space'] == 'space'
        assert im.key_map['Escape'] == 'escape'
        assert im.key_map['F11'] == 'f11'

    def test_default_profile_mappings(self):
        """Kill mutants on default profile key mappings"""
        im = InputManager()
        kb = im.profiles["default_keyboard"]
        assert kb.key_mappings["move_up"] == "w"
        assert kb.key_mappings["move_down"] == "s"
        assert kb.key_mappings["move_left"] == "a"
        assert kb.key_mappings["move_right"] == "d"
        assert kb.key_mappings["rotate_left"] == "q"
        assert kb.key_mappings["rotate_right"] == "e"
        assert kb.key_mappings["action"] == "space"
        assert kb.key_mappings["pause"] == "escape"

    def test_arrow_profile_mappings(self):
        """Kill mutants on arrow keys profile"""
        im = InputManager()
        arrow = im.profiles["arrow_keys"]
        assert arrow.key_mappings["move_up"] == "up"
        assert arrow.key_mappings["move_down"] == "down"
        assert arrow.key_mappings["move_left"] == "left"
        assert arrow.key_mappings["move_right"] == "right"

    def test_gamepad_profile_mappings(self):
        """Kill mutants on gamepad profile"""
        im = InputManager()
        gp = im.profiles["default_gamepad"]
        assert gp.gamepad_mappings["action"] == "a"
        assert gp.gamepad_mappings["back"] == "b"
        assert gp.gamepad_mappings["special"] == "x"
        assert gp.gamepad_mappings["menu"] == "y"
        assert gp.gamepad_mappings["pause"] == "start"

    def test_active_profile_set_to_default_keyboard(self):
        """Kill mutant: active_profile = default_kb"""
        im = InputManager()
        assert im.active_profile is im.profiles["default_keyboard"]

    def test_button_code_left_middle_right(self):
        """Kill mutants on button code map values"""
        im = InputManager()
        assert im._get_button_code('left') == 1
        assert im._get_button_code('middle') == 2
        assert im._get_button_code('right') == 3

    def test_four_gamepads_created(self):
        """Kill mutant: range(4) changed"""
        im = InputManager()
        assert len(im.gamepads) == 4
        assert all(i in im.gamepads for i in range(4))

    def test_gamepad_stick_clamp_values(self):
        """Kill mutants on stick clamp bounds -1.0/1.0"""
        im = InputManager()
        im.simulate_gamepad_connection(0)
        im.simulate_gamepad_stick_input("left", 2.0, -2.0, 0)
        gp = im.gamepads[0]
        assert gp.left_stick.x == pytest.approx(1.0)
        assert gp.left_stick.y == pytest.approx(-1.0)

    def test_action_movement_dead_zone(self):
        """Kill mutant: dead zone 0.1 threshold"""
        im = InputManager()
        im.simulate_gamepad_connection(0)
        # Below dead zone (0.05 magnitude)
        im.simulate_gamepad_stick_input("left", 0.05, 0.0, 0)
        movement = im.get_action_movement_vector()
        assert movement.x == pytest.approx(0.0)
        # Above dead zone (0.2 magnitude)
        im.simulate_gamepad_stick_input("left", 0.2, 0.0, 0)
        movement = im.get_action_movement_vector()
        assert movement.x > 0

    def test_movement_normalize_at_magnitude_gt_1(self):
        """Kill mutant: magnitude > 1 threshold for normalizing"""
        im = InputManager()
        im.simulate_gamepad_connection(0)
        # Stick + key = over 1.0 magnitude
        im.on_key_press('w', 0)
        im.simulate_gamepad_stick_input("left", 0.0, -0.5, 0)
        im.update()
        movement = im.get_action_movement_vector()
        # Should be normalized to at most magnitude 1
        assert movement.magnitude <= 1.0 + 1e-6

    def test_profile_name_stored(self):
        """Kill mutant: InputProfile name string mutation"""
        p = InputProfile("MyProfile")
        assert p.name == "MyProfile"

    def test_profile_map_key_lowercases(self):
        """Kill mutant: key.lower() mutation"""
        p = InputProfile("test")
        p.map_key("jump", "SPACE")
        assert p.key_mappings["jump"] == "space"

    def test_profile_map_gamepad_lowercases(self):
        """Kill mutant: button.lower() mutation"""
        p = InputProfile("test")
        p.map_gamepad_button("fire", "A")
        assert p.gamepad_mappings["fire"] == "a"

    def test_profile_map_mouse_lowercases(self):
        """Kill mutant: button.lower() mutation"""
        p = InputProfile("test")
        p.map_mouse_button("shoot", "LEFT")
        assert p.mouse_mappings["shoot"] == "left"
