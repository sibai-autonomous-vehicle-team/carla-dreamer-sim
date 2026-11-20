import contextlib
import io
import random
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import carla
import collect_lane_offsets as collector


class LaneRetriesTest(unittest.TestCase):
    def run_retry(self, attempts=3):
        with contextlib.redirect_stdout(io.StringIO()):
            return collector.collect_episode_with_retries(
                None, None, None, 100, random.Random(42), idx=7, max_attempts=attempts,
            )

    def test_transient_failures_retry_same_episode(self):
        data = {'images': object()}
        with patch.object(collector, 'collect_episode', side_effect=[
            collector.EpisodeCollectionError('black frame'),
            collector.EpisodeCollectionError('missing frame'), data,
        ]) as episode:
            self.assertIs(self.run_retry(), data)
            self.assertEqual(episode.call_count, 3)

    def test_persistent_failure_is_bounded_and_preserves_cause(self):
        error = collector.EpisodeCollectionError('no spawn available')
        with patch.object(collector, 'collect_episode', side_effect=error) as episode:
            with self.assertRaisesRegex(RuntimeError, 'Episode 007 failed after 3 attempts') as caught:
                self.run_retry()
            self.assertIs(caught.exception.__cause__, error)
            self.assertEqual(episode.call_count, 3)

    def test_unexpected_failure_and_interrupt_are_not_retried(self):
        for error in (RuntimeError('cleanup failed'), ValueError('bug'), KeyboardInterrupt()):
            with self.subTest(error=type(error)), patch.object(
                collector, 'collect_episode', side_effect=error,
            ) as episode:
                with self.assertRaises(type(error)):
                    self.run_retry()
                self.assertEqual(episode.call_count, 1)

    def test_black_frame_cleans_actors_before_retry(self):
        world, road_map, traffic = MagicMock(), MagicMock(), MagicMock()
        ego, camera = MagicMock(), MagicMock()
        road_map.get_spawn_points.return_value = [carla.Transform()]
        world.try_spawn_actor.return_value = ego
        world.spawn_actor.return_value = camera
        events = []
        camera.stop.side_effect = lambda: events.append('camera.stop')
        camera.destroy.side_effect = lambda: events.append('camera.destroy')
        ego.set_autopilot.side_effect = lambda enabled, port: events.append(f'autopilot:{enabled}')
        ego.destroy.side_effect = lambda: events.append('ego.destroy')
        frame = 0

        def tick():
            nonlocal frame
            frame += 1
            events.append('tick')
            if frame == 6:
                callback = camera.listen.call_args.args[0]
                callback(SimpleNamespace(frame=6, width=2, height=2, raw_data=bytes(16)))
            return frame

        world.tick.side_effect = tick
        world.get_snapshot.side_effect = lambda: SimpleNamespace(
            frame=frame, find=lambda _: MagicMock(),
        )
        with patch.object(collector, 'signed_lane_offset', return_value=0.1):
            with self.assertRaisesRegex(collector.EpisodeCollectionError, 'black'):
                collector.collect_episode(world, road_map, traffic, 10, random.Random(42))
        self.assertEqual(events[-6:], [
            'camera.stop', 'camera.destroy', 'autopilot:False', 'tick', 'ego.destroy', 'tick',
        ])


if __name__ == '__main__':
    unittest.main()
