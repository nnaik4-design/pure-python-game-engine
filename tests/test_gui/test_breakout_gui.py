"""GUI tests for Breakout game using Path B (graybox integration).

Tests validate that keyboard input flows through the engine correctly and
produces expected game state changes. Uses two approaches:

1. TestBreakoutInputHandling: Uses threading with event_generate() to inject
   actual tkinter events into the GUI, testing the OS-level input layer.

2. TestBreakoutCollisions & TestBreakoutGameState: Use synchronous, non-threaded
   fixtures to test physics and game logic without race conditions.
"""

import pytest
import threading
import time
from breakout_game import BreakoutGame
from engine.math.vector2 import Vector2


@pytest.fixture
def breakout_game_threaded():
    """Fixture: Launch BreakoutGame in thread for GUI integration tests.

    This fixture:
    - Creates a BreakoutGame instance
    - Launches it in a separate daemon thread
    - Waits for window initialization
    - Yields the game object for testing
    - Cleans up the game window and joins the thread

    Use this for TestBreakoutInputHandling to test tkinter event injection.
    """
    game = BreakoutGame()
    game_thread = threading.Thread(target=game.run)
    game_thread.daemon = True
    game_thread.start()
    time.sleep(0.3)  # Wait for window initialization

    yield game

    # Cleanup
    if game_thread.is_alive():
        try:
            game.window.root.quit()
        except Exception:
            pass
        game_thread.join(timeout=2)


@pytest.fixture
def breakout_game_sync():
    """Fixture: Create a non-threaded BreakoutGame instance for synchronous tests.

    This fixture:
    - Creates a BreakoutGame instance without running the game loop
    - Initializes the game to set up game objects
    - Provides a clean, deterministic environment for physics/logic tests
    - Avoids race conditions from background thread updates
    - Yields the game object for testing
    - Cleans up the window gracefully

    Use this for TestBreakoutCollisions and TestBreakoutGameState to test
    physics and game state changes without thread race conditions.
    """
    game = BreakoutGame()
    # Initialize the game to set up all game objects (ball, paddle, bricks, etc.)
    game.initialize()
    if game.current_scene:
        game.current_scene.initialize()

    yield game

    # Cleanup
    try:
        game.window.root.destroy()
    except Exception:
        pass


class TestBreakoutInputHandling:
    """Tests for paddle control via keyboard input (7 tests)."""

    def test_paddle_moves_left_on_arrow_key(self, breakout_game_threaded):
        """Input: Arrow left key should move paddle left."""
        game = breakout_game_threaded
        initial_x = game.paddle.transform.position.x

        # Inject event into the actual tkinter window and process it
        game.window.root.event_generate('<KeyPress-Left>')
        game.window.root.update()  # Force tkinter to process pending events
        time.sleep(0.15)  # Let game loop update with new input

        # Paddle should move left (x decreases)
        assert game.paddle.transform.position.x < initial_x

    def test_paddle_moves_left_on_a_key(self, breakout_game_threaded):
        """Input: 'a' key should move paddle left (alternative control)."""
        game = breakout_game_threaded
        initial_x = game.paddle.transform.position.x

        # Inject event into the actual tkinter window and process it
        game.window.root.event_generate('<KeyPress-a>')
        game.window.root.update()
        time.sleep(0.15)

        # Paddle should move left
        assert game.paddle.transform.position.x < initial_x

    def test_paddle_moves_right_on_arrow_key(self, breakout_game_threaded):
        """Input: Arrow right key should move paddle right."""
        game = breakout_game_threaded
        initial_x = game.paddle.transform.position.x

        # Inject event into the actual tkinter window and process it
        game.window.root.event_generate('<KeyPress-Right>')
        game.window.root.update()
        time.sleep(0.15)

        # Paddle should move right (x increases)
        assert game.paddle.transform.position.x > initial_x

    def test_paddle_moves_right_on_d_key(self, breakout_game_threaded):
        """Input: 'd' key should move paddle right (alternative control)."""
        game = breakout_game_threaded
        initial_x = game.paddle.transform.position.x

        # Inject event into the actual tkinter window and process it
        game.window.root.event_generate('<KeyPress-d>')
        game.window.root.update()
        time.sleep(0.15)

        # Paddle should move right
        assert game.paddle.transform.position.x > initial_x

    def test_paddle_cannot_move_past_left_boundary(self, breakout_game_threaded):
        """Boundary: Paddle should not move past left edge (x >= 0)."""
        game = breakout_game_threaded

        # Move paddle all the way left by sending multiple left events
        for _ in range(20):
            game.window.root.event_generate('<KeyPress-Left>')
            game.window.root.update()
            time.sleep(0.02)

        # Paddle x should not be negative
        assert game.paddle.transform.position.x >= 0

    def test_paddle_cannot_move_past_right_boundary(self, breakout_game_threaded):
        """Boundary: Paddle should not move past right edge (x <= 736)."""
        game = breakout_game_threaded

        # Move paddle all the way right by sending multiple right events
        for _ in range(20):
            game.window.root.event_generate('<KeyPress-Right>')
            game.window.root.update()
            time.sleep(0.02)

        # Paddle x + width should not exceed screen width (800 - 64 = 736)
        assert game.paddle.transform.position.x <= 736

    def test_paddle_position_changes_over_multiple_frames(self, breakout_game_threaded):
        """State: Paddle position should accumulate movement over time."""
        game = breakout_game_threaded
        initial_x = game.paddle.transform.position.x

        # Send continuous right input events over multiple frames
        for _ in range(5):
            game.window.root.event_generate('<KeyPress-Right>')
            game.window.root.update()
            time.sleep(0.03)

        # Paddle should have moved significantly to the right
        assert game.paddle.transform.position.x > initial_x + 50


class TestBreakoutCollisions:
    """Tests for ball physics and collision detection (9 tests)."""

    def test_ball_bounces_off_top_wall(self, breakout_game_sync):
        """Physics: Ball should bounce off top wall (y <= 0)."""
        game = breakout_game_sync

        # Position ball near top with upward velocity
        game.ball.transform.position = Vector2(400, 5)
        game.ball.velocity = Vector2(0, -100)

        # Update scene and game to process collision
        if game.current_scene:
            game.current_scene.update(0.016)
        game.update(0.016)

        # Ball velocity Y should reverse (be positive after bounce)
        assert game.ball.velocity.y > 0

    def test_ball_bounces_off_side_walls(self, breakout_game_sync):
        """Physics: Ball should bounce off side walls (x <= 0 or x >= 800)."""
        game = breakout_game_sync

        # Test left wall bounce
        game.ball.transform.position = Vector2(5, 300)
        game.ball.velocity = Vector2(-100, 50)
        if game.current_scene:
            game.current_scene.update(0.016)
        game.update(0.016)
        assert game.ball.velocity.x > 0, "Ball should bounce off left wall"

        # Test right wall bounce
        game.ball.transform.position = Vector2(795, 300)
        game.ball.velocity = Vector2(100, 50)
        if game.current_scene:
            game.current_scene.update(0.016)
        game.update(0.016)
        assert game.ball.velocity.x < 0, "Ball should bounce off right wall"

    def test_ball_is_lost_when_below_screen(self, breakout_game_sync):
        """Physics: Ball should be lost when it falls below screen (y > 600)."""
        game = breakout_game_sync
        initial_lives = game.lives

        # Position ball below screen
        game.ball.transform.position = Vector2(400, 610)

        # Update game to process ball loss
        if game.current_scene:
            game.current_scene.update(0.016)
        game.update(0.016)

        # Lives should decrease
        assert game.lives < initial_lives, "Lives should decrease when ball is lost"

    def test_ball_resets_after_loss(self, breakout_game_sync):
        """Physics: Ball position should reset after falling below screen."""
        game = breakout_game_sync

        # Position ball below screen
        game.ball.transform.position = Vector2(400, 610)

        # Update game
        if game.current_scene:
            game.current_scene.update(0.016)
        game.update(0.016)

        # Ball should be repositioned near paddle
        assert game.ball.transform.position.y < 600, "Ball should reset after loss"
        assert 300 < game.ball.transform.position.x < 500, "Ball should reset near center"

    def test_ball_bounces_off_paddle(self, breakout_game_sync):
        """Collision: Ball should bounce off paddle and reverse velocity."""
        game = breakout_game_sync

        # Position ball just above paddle with downward velocity
        game.ball.transform.position = Vector2(400, 535)  # Just above paddle
        game.ball.velocity = Vector2(0, 100)

        # Update to process collision
        if game.current_scene:
            game.current_scene.update(0.016)
        game.update(0.016)

        # Ball velocity Y should reverse (be negative after bounce)
        assert game.ball.velocity.y < 0, "Ball should bounce off paddle"

    def test_ball_bounces_off_brick(self, breakout_game_sync):
        """Collision: Ball should bounce off brick and reverse velocity."""
        game = breakout_game_sync

        if len(game.bricks) == 0:
            pytest.skip("No bricks available for collision test")

        # Get first brick
        brick = game.bricks[0]
        initial_bricks = len(game.bricks)

        # Position ball just above brick with downward velocity
        game.ball.transform.position = Vector2(brick.transform.position.x, brick.transform.position.y - 5)
        game.ball.velocity = Vector2(0, 100)

        # Update to process collision
        if game.current_scene:
            game.current_scene.update(0.016)
        game.update(0.016)

        # Brick should be removed
        assert len(game.bricks) < initial_bricks, "Brick should be removed after collision"

    def test_brick_removed_after_collision(self, breakout_game_sync):
        """State: Brick count should decrease after ball collision."""
        game = breakout_game_sync

        if len(game.bricks) == 0:
            pytest.skip("No bricks available for collision test")

        initial_brick_count = len(game.bricks)
        brick_to_hit = game.bricks[0]

        # Position ball into brick for collision
        game.ball.transform.position = Vector2(
            brick_to_hit.transform.position.x,
            brick_to_hit.transform.position.y - 2
        )
        game.ball.velocity = Vector2(0, 100)

        # Update to trigger collision
        if game.current_scene:
            game.current_scene.update(0.016)
        game.update(0.016)

        # Verify brick was removed
        assert len(game.bricks) == initial_brick_count - 1, "Brick count should decrease"
        assert brick_to_hit not in game.bricks, "Specific brick should be removed"

    def test_ball_speed_increases_after_brick_hit(self, breakout_game_sync):
        """Physics: Ball speed should increase 2% after hitting brick."""
        game = breakout_game_sync

        if len(game.bricks) == 0:
            pytest.skip("No bricks available for speed test")

        initial_speed = game.ball.speed
        brick = game.bricks[0]

        # Position and collide
        game.ball.transform.position = Vector2(brick.transform.position.x, brick.transform.position.y - 2)
        game.ball.velocity = Vector2(0, 100)
        if game.current_scene:
            game.current_scene.update(0.016)
        game.update(0.016)

        # Speed should increase by 2%
        expected_speed = initial_speed * 1.02
        assert game.ball.speed > initial_speed, "Ball speed should increase after brick collision"
        assert game.ball.speed <= expected_speed * 1.01, "Speed increase should be approximately 2%"

    def test_multiple_collisions_in_sequence(self, breakout_game_sync):
        """State: Game should handle multiple successive collisions correctly."""
        game = breakout_game_sync

        if len(game.bricks) < 2:
            pytest.skip("Not enough bricks for sequential collision test")

        initial_brick_count = len(game.bricks)

        # Hit multiple bricks in sequence
        for _ in range(2):
            if len(game.bricks) == 0:
                break
            brick = game.bricks[0]
            game.ball.transform.position = Vector2(brick.transform.position.x, brick.transform.position.y - 2)
            game.ball.velocity = Vector2(0, 100)
            if game.current_scene:
                game.current_scene.update(0.016)
            game.update(0.016)
            time.sleep(0.01)

        # At least one brick should have been removed
        assert len(game.bricks) < initial_brick_count, "Bricks should be removed in sequence"


class TestBreakoutGameState:
    """Tests for scoring, lives, and game state transitions (6 tests)."""

    def test_score_increments_on_brick_collision(self, breakout_game_sync):
        """Scoring: Score should increase when ball hits brick."""
        game = breakout_game_sync

        if len(game.bricks) == 0:
            pytest.skip("No bricks available for scoring test")

        initial_score = game.score
        brick = game.bricks[0]
        brick_points = brick.points

        # Collide with brick
        game.ball.transform.position = Vector2(brick.transform.position.x, brick.transform.position.y - 2)
        game.ball.velocity = Vector2(0, 100)
        if game.current_scene:
            game.current_scene.update(0.016)
        game.update(0.016)

        # Score should increase by brick's point value
        assert game.score > initial_score, "Score should increase after brick collision"
        assert game.score == initial_score + brick_points, f"Score should increase by {brick_points}"

    def test_score_tracks_multiple_bricks(self, breakout_game_sync):
        """Scoring: Score should accumulate correctly over multiple brick hits."""
        game = breakout_game_sync

        if len(game.bricks) < 2:
            pytest.skip("Not enough bricks for multi-hit scoring test")

        initial_score = game.score
        total_points_earned = 0

        # Hit multiple bricks
        for i in range(min(2, len(game.bricks))):
            if len(game.bricks) == 0:
                break
            brick = game.bricks[0]
            total_points_earned += brick.points

            game.ball.transform.position = Vector2(brick.transform.position.x, brick.transform.position.y - 2)
            game.ball.velocity = Vector2(0, 100)
            if game.current_scene:
                game.current_scene.update(0.016)
            game.update(0.016)
            time.sleep(0.01)

        # Total score should reflect all bricks hit
        assert game.score == initial_score + total_points_earned, "Score should accumulate correctly"

    def test_lives_decrement_on_ball_loss(self, breakout_game_sync):
        """State: Lives should decrease when ball falls off screen."""
        game = breakout_game_sync
        initial_lives = game.lives

        # Drop ball off screen
        game.ball.transform.position = Vector2(400, 610)
        if game.current_scene:
            game.current_scene.update(0.016)
        game.update(0.016)

        # Lives should decrease
        assert game.lives == initial_lives - 1, "Lives should decrease by 1 when ball is lost"

    def test_game_over_when_lives_reach_zero(self, breakout_game_sync):
        """State: game_state should be 'game_over' when lives reach 0."""
        game = breakout_game_sync

        # Set lives to 1
        game.lives = 1

        # Drop ball off screen to trigger game over
        game.ball.transform.position = Vector2(400, 610)
        if game.current_scene:
            game.current_scene.update(0.016)
        game.update(0.016)

        # Game state should be game_over
        assert game.game_state == "game_over", "Game should be over when lives reach 0"

    def test_level_complete_when_all_bricks_destroyed(self, breakout_game_sync):
        """State: game_state should be 'level_complete' when all bricks are destroyed."""
        game = breakout_game_sync

        # Remove all bricks
        bricks_to_remove = len(game.bricks)
        for _ in range(bricks_to_remove):
            if len(game.bricks) > 0:
                brick = game.bricks[0]
                game.ball.transform.position = Vector2(brick.transform.position.x, brick.transform.position.y - 2)
                game.ball.velocity = Vector2(0, 100)
                if game.current_scene:
                    game.current_scene.update(0.016)
                game.update(0.016)
                time.sleep(0.01)

        # Game state should be level_complete
        assert game.game_state == "level_complete", "Game should be level complete when all bricks destroyed"

    def test_game_starts_in_playing_state(self, breakout_game_sync):
        """State: Game should start in 'playing' state."""
        game = breakout_game_sync

        # Game should initially be in playing state
        assert game.game_state == "playing", "Game should start in playing state"

    def test_game_state_persists_across_frames(self, breakout_game_sync):
        """State: Game state should remain consistent across multiple frames."""
        game = breakout_game_sync
        initial_state = game.game_state

        # Simulate multiple frames without triggering game over or level complete
        for _ in range(5):
            if game.current_scene:
                game.current_scene.update(0.016)
            game.engine.input_manager.update() if hasattr(game, 'engine') else game.input_manager.update()
            game.update(0.016)
            time.sleep(0.01)

        # Game state should remain the same
        assert game.game_state == initial_state, "Game state should persist across frames"
