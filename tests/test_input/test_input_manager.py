"""
Blackbox tests for InputManager, InputProfile, and GamepadState classes.
Tests use Equivalence Partitioning (EP), Boundary Analysis (BA), and Error Guessing (EG).
"""
import pytest
from engine.input.input_manager import InputManager, InputProfile, GamepadState
from engine.math.vector2 import Vector2


# ================================================================
# InputProfile Tests
# ================================================================

class TestInputProfile:

    def test_constructor(self):
        """EP: Profile has name and empty mappings"""
        profile = InputProfile("Default")
        assert profile.name == "Default"
        assert profile.key_mappings == {}
        assert profile.gamepad_mappings == {}
        assert profile.mouse_mappings == {}

    def test_map_key(self):
        """EP: Map action to key"""
        profile = InputProfile("Test")
        profile.map_key("jump", "Space")
        assert profile.get_key_for_action("jump") == "space"

    def test_map_key_lowercase(self):
        """EP: Keys are normalized to lowercase"""
        profile = InputProfile("Test")
        profile.map_key("fire", "F")
        assert profile.get_key_for_action("fire") == "f"

    def test_map_gamepad_button(self):
        """EP: Map action to gamepad button"""
        profile = InputProfile("Test")
        profile.map_gamepad_button("jump", "A")
        assert profile.get_gamepad_button_for_action("jump") == "a"

    def test_map_mouse_button(self):
        """EP: Map action to mouse button"""
        profile = InputProfile("Test")
        profile.map_mouse_button("fire", "Left")
        assert profile.get_mouse_button_for_action("fire") == "left"

    def test_get_nonexistent_action(self):
        """BA: Getting unmapped action returns None"""
        profile = InputProfile("Test")
        assert profile.get_key_for_action("nonexistent") is None
        assert profile.get_gamepad_button_for_action("nonexistent") is None
        assert profile.get_mouse_button_for_action("nonexistent") is None

    def test_overwrite_mapping(self):
        """EG: Remapping an action overwrites the old mapping"""
        profile = InputProfile("Test")
        profile.map_key("jump", "space")
        profile.map_key("jump", "w")
        assert profile.get_key_for_action("jump") == "w"


# ================================================================
# GamepadState Tests
# ================================================================

class TestGamepadState:

    def test_constructor(self):
        """EP: GamepadState initializes with correct defaults"""
        gp = GamepadState(0)
        assert gp.id == 0
        assert gp.connected is False
        assert gp.left_stick == Vector2.zero()
        assert gp.right_stick == Vector2.zero()
        assert gp.left_trigger == 0.0
        assert gp.right_trigger == 0.0

    def test_button_names_mapping(self):
        """EP: Standard button names are mapped"""
        gp = GamepadState(0)
        assert gp.button_names[0] == 'a'
        assert gp.button_names[1] == 'b'
        assert gp.button_names[7] == 'start'

    def test_dpad_defaults(self):
        """EP: D-pad defaults to all False"""
        gp = GamepadState(0)
        assert gp.dpad_up is False
        assert gp.dpad_down is False
        assert gp.dpad_left is False
        assert gp.dpad_right is False


# ================================================================
# InputManager Constructor Tests
# ================================================================

class TestInputManagerConstructor:

    def test_default_constructor(self):
        """EP: InputManager initializes with empty state"""
        im = InputManager()
        assert len(im.keys_pressed) == 0
        assert im.mouse_position == Vector2.zero()
        assert len(im.mouse_buttons_pressed) == 0

    def test_four_gamepads_created(self):
        """EP: 4 gamepads are created by default"""
        im = InputManager()
        assert len(im.gamepads) == 4
        for i in range(4):
            assert i in im.gamepads

    def test_default_profiles_created(self):
        """EP: Default input profiles are created"""
        im = InputManager()
        assert "default_keyboard" in im.profiles
        assert "arrow_keys" in im.profiles
        assert "default_gamepad" in im.profiles

    def test_active_profile_set(self):
        """EP: Active profile is set to default keyboard"""
        im = InputManager()
        assert im.active_profile is not None
        assert im.active_profile.name == "Default Keyboard"


# ================================================================
# Keyboard Input Tests
# ================================================================

class TestKeyboardInput:

    def test_key_press(self):
        """EP: Key press registers key as pressed"""
        im = InputManager()
        im.on_key_press("w", 0)
        assert im.is_key_pressed("w")

    def test_key_release(self):
        """EP: Key release removes key from pressed"""
        im = InputManager()
        im.on_key_press("w", 0)
        im.on_key_release("w", 0)
        assert not im.is_key_pressed("w")

    def test_key_just_pressed(self):
        """EP: Just pressed detected after update"""
        im = InputManager()
        im.on_key_press("space", 0)
        im.update()
        assert im.is_key_just_pressed("space")

    def test_key_just_released(self):
        """EP: Just released detected after update"""
        im = InputManager()
        im.on_key_press("w", 0)
        im.update()  # Process the press
        im.on_key_release("w", 0)
        im.update()  # Process the release
        assert im.is_key_just_released("w")

    def test_key_map_converts_names(self):
        """EP: Key map converts tkinter keysyms to lowercase names"""
        im = InputManager()
        im.on_key_press("Up", 0)
        assert im.is_key_pressed("up")

    def test_key_not_pressed(self):
        """BA: Unpressed key returns False"""
        im = InputManager()
        assert not im.is_key_pressed("w")

    def test_case_insensitive_check(self):
        """EG: is_key_pressed is case insensitive"""
        im = InputManager()
        im.on_key_press("w", 0)
        assert im.is_key_pressed("W")
        assert im.is_key_pressed("w")

    def test_duplicate_press_ignored(self):
        """EG: Pressing already pressed key doesn't add again"""
        im = InputManager()
        im.on_key_press("w", 0)
        im.on_key_press("w", 0)
        # Still should be in pressed set
        assert im.is_key_pressed("w")
        # Only one entry in frame_key_presses
        assert len(im.frame_key_presses) == 1


# ================================================================
# Mouse Input Tests
# ================================================================

class TestMouseInput:

    def test_mouse_move(self):
        """EP: Mouse move updates position"""
        im = InputManager()
        im.on_mouse_event('move', 0, 100, 200)
        pos = im.get_mouse_position()
        assert pos.x == 100
        assert pos.y == 200

    def test_mouse_click(self):
        """EP: Mouse click registers button"""
        im = InputManager()
        im.on_mouse_event('click', 1, 0, 0)
        assert im.is_mouse_button_pressed(1)

    def test_mouse_release(self):
        """EP: Mouse release removes button"""
        im = InputManager()
        im.on_mouse_event('click', 1, 0, 0)
        im.on_mouse_event('release', 1, 0, 0)
        assert not im.is_mouse_button_pressed(1)

    def test_mouse_button_by_name(self):
        """EP: Mouse button can be checked by name"""
        im = InputManager()
        im.on_mouse_event('click', 1, 0, 0)
        assert im.is_mouse_button_pressed('left')

    def test_mouse_button_right(self):
        """EP: Right mouse button maps to code 3"""
        im = InputManager()
        im.on_mouse_event('click', 3, 0, 0)
        assert im.is_mouse_button_pressed('right')

    def test_mouse_button_middle(self):
        """EP: Middle mouse button maps to code 2"""
        im = InputManager()
        im.on_mouse_event('click', 2, 0, 0)
        assert im.is_mouse_button_pressed('middle')

    def test_mouse_position_copy(self):
        """EG: get_mouse_position returns a copy"""
        im = InputManager()
        im.on_mouse_event('move', 0, 50, 50)
        pos = im.get_mouse_position()
        pos.x = 999
        assert im.get_mouse_position().x == 50

    def test_mouse_just_pressed(self):
        """EP: Mouse button just pressed detected after update"""
        im = InputManager()
        im.on_mouse_event('click', 1, 0, 0)
        im.update()
        assert im.is_mouse_button_just_pressed(1)


# ================================================================
# Movement Vector Tests
# ================================================================

class TestMovementVector:

    def test_no_keys_zero_vector(self):
        """BA: No keys pressed returns zero vector"""
        im = InputManager()
        mv = im.get_movement_vector()
        assert mv == Vector2.zero()

    def test_arrow_up(self):
        """EP: Up arrow produces negative Y"""
        im = InputManager()
        im.on_key_press("Up", 0)
        mv = im.get_movement_vector()
        assert mv.y < 0

    def test_arrow_down(self):
        """EP: Down arrow produces positive Y"""
        im = InputManager()
        im.on_key_press("Down", 0)
        mv = im.get_movement_vector()
        assert mv.y > 0

    def test_arrow_left(self):
        """EP: Left arrow produces negative X"""
        im = InputManager()
        im.on_key_press("Left", 0)
        mv = im.get_movement_vector()
        assert mv.x < 0

    def test_arrow_right(self):
        """EP: Right arrow produces positive X"""
        im = InputManager()
        im.on_key_press("Right", 0)
        mv = im.get_movement_vector()
        assert mv.x > 0

    def test_diagonal_normalized(self):
        """BA: Diagonal movement is normalized to magnitude ~1"""
        im = InputManager()
        im.on_key_press("Up", 0)
        im.on_key_press("Right", 0)
        mv = im.get_movement_vector()
        assert mv.magnitude == pytest.approx(1.0, abs=0.01)

    def test_wasd_fallback(self):
        """EP: WASD keys work as fallback when no arrows pressed"""
        im = InputManager()
        im.on_key_press("w", 0)
        mv = im.get_movement_vector()
        assert mv.y < 0

    def test_is_arrow_key_pressed(self):
        """EP: is_arrow_key_pressed returns correct tuple"""
        im = InputManager()
        im.on_key_press("Up", 0)
        im.on_key_press("Right", 0)
        up, down, left, right = im.is_arrow_key_pressed()
        assert up is True
        assert down is False
        assert left is False
        assert right is True

    def test_is_wasd_pressed(self):
        """EP: is_wasd_pressed returns correct tuple"""
        im = InputManager()
        im.on_key_press("w", 0)
        im.on_key_press("d", 0)
        w, a, s, d = im.is_wasd_pressed()
        assert w is True
        assert a is False
        assert s is False
        assert d is True


# ================================================================
# Profile Management Tests
# ================================================================

class TestProfileManagement:

    def test_set_active_profile(self):
        """EP: Set active profile by name"""
        im = InputManager()
        im.set_active_profile("arrow_keys")
        assert im.active_profile.name == "Arrow Keys"

    def test_set_nonexistent_profile(self):
        """EG: Setting nonexistent profile keeps current"""
        im = InputManager()
        old_profile = im.active_profile
        im.set_active_profile("nonexistent")
        assert im.active_profile is old_profile

    def test_create_profile(self):
        """EP: Create a custom profile"""
        im = InputManager()
        profile = im.create_profile("custom")
        assert profile.name == "custom"
        assert "custom" in im.profiles

    def test_get_profile(self):
        """EP: Get profile by name"""
        im = InputManager()
        profile = im.get_profile("default_keyboard")
        assert profile is not None
        assert profile.name == "Default Keyboard"

    def test_get_nonexistent_profile(self):
        """BA: Get nonexistent profile returns None"""
        im = InputManager()
        assert im.get_profile("nonexistent") is None

    def test_list_profiles(self):
        """EP: List all profile names"""
        im = InputManager()
        names = im.list_profiles()
        assert "default_keyboard" in names
        assert "arrow_keys" in names
        assert "default_gamepad" in names


# ================================================================
# Action-Based Input Tests
# ================================================================

class TestActionInput:

    def test_is_action_pressed_keyboard(self):
        """EP: Action pressed via keyboard mapping"""
        im = InputManager()
        im.set_active_profile("default_keyboard")
        im.on_key_press("w", 0)
        assert im.is_action_pressed("move_up")

    def test_is_action_not_pressed(self):
        """BA: Action not pressed returns False"""
        im = InputManager()
        assert not im.is_action_pressed("move_up")

    def test_is_action_pressed_no_profile(self):
        """BA: No active profile returns False"""
        im = InputManager()
        im.active_profile = None
        assert not im.is_action_pressed("move_up")

    def test_action_movement_vector(self):
        """EP: Action movement vector from profile mappings"""
        im = InputManager()
        im.set_active_profile("default_keyboard")
        im.on_key_press("w", 0)
        im.on_key_press("d", 0)
        mv = im.get_action_movement_vector()
        assert mv.x > 0
        assert mv.y < 0


# ================================================================
# Gamepad Tests
# ================================================================

class TestGamepadInput:

    def test_gamepad_not_connected_by_default(self):
        """EP: Gamepads start disconnected"""
        im = InputManager()
        assert not im.is_gamepad_connected(0)

    def test_simulate_connection(self):
        """EP: Simulate connecting a gamepad"""
        im = InputManager()
        im.simulate_gamepad_connection(0)
        assert im.is_gamepad_connected(0)

    def test_simulate_button_press(self):
        """EP: Simulate pressing a gamepad button"""
        im = InputManager()
        im.simulate_gamepad_connection(0)
        im.simulate_gamepad_button_press("a", 0)
        assert im.is_gamepad_button_pressed("a", 0)

    def test_gamepad_button_not_connected(self):
        """BA: Button check on disconnected gamepad returns False"""
        im = InputManager()
        assert not im.is_gamepad_button_pressed("a", 0)

    def test_gamepad_stick_not_connected(self):
        """BA: Stick check on disconnected gamepad returns zero"""
        im = InputManager()
        stick = im.get_gamepad_stick("left", 0)
        assert stick == Vector2.zero()

    def test_simulate_stick_input(self):
        """EP: Simulate analog stick input"""
        im = InputManager()
        im.simulate_gamepad_connection(0)
        im.simulate_gamepad_stick_input("left", 0.5, -0.3, 0)
        stick = im.get_gamepad_stick("left", 0)
        assert stick.x == pytest.approx(0.5)
        assert stick.y == pytest.approx(-0.3)

    def test_simulate_stick_clamped(self):
        """BA: Stick input is clamped to [-1, 1]"""
        im = InputManager()
        im.simulate_gamepad_connection(0)
        im.simulate_gamepad_stick_input("left", 5.0, -5.0, 0)
        stick = im.get_gamepad_stick("left", 0)
        assert stick.x == pytest.approx(1.0)
        assert stick.y == pytest.approx(-1.0)

    def test_gamepad_trigger_not_connected(self):
        """BA: Trigger on disconnected gamepad returns 0"""
        im = InputManager()
        assert im.get_gamepad_trigger("left", 0) == 0.0

    def test_gamepad_trigger_connected(self):
        """EP: Trigger on connected gamepad returns value"""
        im = InputManager()
        im.simulate_gamepad_connection(0)
        im.gamepads[0].left_trigger = 0.8
        assert im.get_gamepad_trigger("left", 0) == pytest.approx(0.8)

    def test_gamepad_trigger_unknown_name(self):
        """EG: Unknown trigger name returns 0"""
        im = InputManager()
        im.simulate_gamepad_connection(0)
        assert im.get_gamepad_trigger("unknown", 0) == 0.0

    def test_gamepad_stick_unknown_name(self):
        """EG: Unknown stick name returns zero vector"""
        im = InputManager()
        im.simulate_gamepad_connection(0)
        assert im.get_gamepad_stick("unknown", 0) == Vector2.zero()


# ================================================================
# Callback System Tests
# ================================================================

class TestCallbackSystem:

    def test_register_and_trigger_callback(self):
        """EP: Register callback and trigger it"""
        im = InputManager()
        result = []
        im.register_input_callback("on_fire", lambda: result.append("fired"))
        im.trigger_callback("on_fire")
        assert result == ["fired"]

    def test_unregister_callback(self):
        """EP: Unregistered callback is not triggered"""
        im = InputManager()
        result = []
        im.register_input_callback("on_fire", lambda: result.append("fired"))
        im.unregister_input_callback("on_fire")
        im.trigger_callback("on_fire")
        assert result == []

    def test_trigger_nonexistent_callback(self):
        """EG: Triggering non-registered callback doesn't crash"""
        im = InputManager()
        im.trigger_callback("nonexistent")  # Should not raise

    def test_callback_with_args(self):
        """EP: Callbacks receive arguments"""
        im = InputManager()
        result = []
        im.register_input_callback("on_hit", lambda dmg: result.append(dmg))
        im.trigger_callback("on_hit", 50)
        assert result == [50]


# ================================================================
# Update Cycle Tests
# ================================================================

class TestUpdateCycle:

    def test_update_clears_frame_events(self):
        """EP: Update clears frame-specific events"""
        im = InputManager()
        im.on_key_press("w", 0)
        im.update()
        # After update, frame events should be cleared
        assert len(im.frame_key_presses) == 0
        assert len(im.frame_key_releases) == 0
        assert len(im.frame_mouse_clicks) == 0

    def test_just_pressed_only_for_one_frame(self):
        """BA: Just pressed is only True for one update cycle"""
        im = InputManager()
        im.on_key_press("space", 0)
        im.update()
        assert im.is_key_just_pressed("space")
        im.update()  # Second frame with no new events
        assert not im.is_key_just_pressed("space")

    def test_previous_keys_stored(self):
        """EP: Previous keys are stored after update"""
        im = InputManager()
        im.on_key_press("w", 0)
        im.update()
        assert "w" in im.previous_keys


# ================================================================
# Error Guessing
# ================================================================

class TestInputManagerErrorGuessing:

    def test_multiple_keys_simultaneously(self):
        """EG: Multiple keys pressed at once"""
        im = InputManager()
        im.on_key_press("w", 0)
        im.on_key_press("a", 0)
        im.on_key_press("space", 0)
        assert im.is_key_pressed("w")
        assert im.is_key_pressed("a")
        assert im.is_key_pressed("space")

    def test_release_unpressed_key(self):
        """EG: Releasing a key that wasn't pressed doesn't crash"""
        im = InputManager()
        im.on_key_release("x", 0)  # Should not raise

    def test_release_unclicked_mouse(self):
        """EG: Releasing unclicked mouse button doesn't crash"""
        im = InputManager()
        im.on_mouse_event('release', 1, 0, 0)  # Should not raise

    def test_invalid_gamepad_id(self):
        """EG: Invalid gamepad ID returns False"""
        im = InputManager()
        assert not im.is_gamepad_connected(99)
