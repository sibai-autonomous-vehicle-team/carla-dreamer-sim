import unittest
from dataclasses import replace

import torch

from training.config import Settings
from training.dataset import decision_features, episode_slices, hj_examples, safety_margin, start_groups
from training.prepare_data import combined_cache, transition_counts


def episodes(gaps, starts=None):
    lengths = torch.tensor([len(episode) for episode in gaps])
    poses = torch.zeros(int(lengths.sum()), 12)
    data = dict(states=poses, lengths=lengths, distances=torch.tensor(sum(gaps, []), dtype=torch.float32),
                actions=torch.zeros(len(poses), 2), costs=torch.zeros(len(poses)),
                conditions=["day_sunny"] * len(gaps), episode_ids=torch.arange(len(gaps)))
    for index, rows in enumerate(episode_slices(data)):
        x = starts[index] if starts is not None else 100 * index
        poses[rows, 0] = x + torch.arange(lengths[index]) * .25
        poses[rows, 6] = x + 10
    return data


class HJDataTests(unittest.TestCase):
    def test_local_nine_meter_default_and_upper_crossing(self):
        cfg = Settings()
        self.assertEqual((cfg.safety_buffer_m, cfg.max_gap_m), (1., 9.))
        self.assertEqual(len(cfg.shared_conditions), 9)
        data = episodes([[7., 8., 9., 5., 4.]])
        state, _, successor, _, next_margin, failed = hj_examples(data, [0], cfg)
        self.assertEqual(state[:, 0].tolist(), [7., 8.])
        self.assertEqual(successor[:, 0].tolist(), [8., 9.])
        self.assertEqual(failed.tolist(), [False, True])
        self.assertEqual(next_margin.tolist(), [1., 0.])

    def test_two_boundaries_and_midpoint(self):
        distance = torch.tensor([.5, 1., 15.5, 30., 31.])
        self.assertEqual(safety_margin(distance, 1., 30.).tolist(), [-.5, 0., 14.5, 0., -1.])
        with self.assertRaises(ValueError):
            safety_margin(distance, 1., 1.)

    def test_first_hit_absorbs_and_unknown_final_row_has_no_target(self):
        cfg = replace(Settings(), max_gap_m=30.)
        data = episodes([[4., .5, 12., 19.], [14., 29., 31., 15., 4.], [5., 6., 7., 8.]])
        data["actions"][:, 0] = torch.arange(len(data["states"])) / 10
        state, action, successor, margin, next_margin, failed = hj_examples(data, [0, 1, 2], cfg)
        torch.testing.assert_close(state[:, 0], torch.tensor([4., 14., 29., 5., 6.]))
        torch.testing.assert_close(successor[:, 0], torch.tensor([.5, 29., 31., 6., 7.]))
        self.assertEqual(failed.tolist(), [True, False, True, False, False])
        self.assertEqual(next_margin[failed].tolist(), [-.5, -1.])
        self.assertTrue((margin > 0).all())
        torch.testing.assert_close(action[:, 0], data["actions"][[1, 5, 6, 10, 11], 0])
        counts = [transition_counts(data, rows, cfg) for rows in episode_slices(data)]
        self.assertEqual(counts[0]["later_return_inside_after_first_hit"], 1)
        self.assertEqual(counts[1]["eligible_first_far_successors"], 1)
        self.assertEqual(counts[2]["unknown_ends_without_buffer_failure"], 1)
        for group in counts:
            self.assertEqual(group["rows"], group["eligible_transitions"] + group["unusable_rows"])
            self.assertEqual(group["unusable_rows"], group["post_hit_aligned_rows_excluded"] +
                             group["rows_missing_initial_alignment_or_history"] +
                             group["rows_missing_final_alignment_or_successor"])

    def test_observed_failure_without_eligible_transition(self):
        cfg = Settings()
        data = episodes([[.5, 2., 5.], [4., 3., .5], [4.]])
        _, _, _, _, _, failed = hj_examples(data, [0, 1, 2], cfg)
        self.assertEqual(failed.tolist(), [False])
        counts = [transition_counts(data, rows, cfg) for rows in episode_slices(data)]
        self.assertEqual([c["first_failures_without_trainable_entry"] for c in counts], [1, 1, 0])
        self.assertEqual(counts[2]["unusable_rows"], 1)

    def test_logged_actions_use_saturated_request_units_and_keep_steering_sign(self):
        data = episodes([[5.] * 4])
        data["actions"][1:3] = torch.tensor([[4., -.8], [-4., .8]])
        original = data["actions"].clone()
        _, actions, *_ = hj_examples(data, [0], Settings())
        torch.testing.assert_close(actions, torch.tensor([[3., -.8], [-3., .8]]))
        torch.testing.assert_close(data["actions"], original)

    def test_archive_alignment_matches_live_decision_without_future_pose(self):
        data = episodes([[6., 5., 4., 3.]])
        state, *_ = hj_examples(data, [0], Settings())
        pose = data["states"]
        single = decision_features(pose[0], pose[1], 6., .1)
        torch.testing.assert_close(single, state[0])
        torch.testing.assert_close(single, torch.tensor([6., 9.75, 0., 0., 1., 2.5, -2.5, 0.]))
        pose[2:, :2] += 100
        data["distances"][1:] += 1
        changed, *_ = hj_examples(data, [0], Settings())
        torch.testing.assert_close(changed[0], single)

    def test_start_matching_crosses_cells_sources_and_indirect_neighbors(self):
        cfg = replace(Settings(), shared_conditions=("day_sunny",))
        old = episodes([[4.] * 4] * 4, [.24, .67, 2., 4.])
        new = episodes([[4.] * 4] * 3, [.45, 10., 12.])
        combined = combined_cache(old, new, cfg)
        self.assertEqual(combined["group_ids"][0], combined["group_ids"][1])
        self.assertEqual(combined["group_ids"][0], combined["group_ids"][4])
        splits = combined["splits"]
        self.assertTrue(any({0, 1, 4} <= set(ids) for ids in splits.values()))
        self.assertEqual(sorted(sum(splits.values(), [])), list(range(7)))
        self.assertEqual(combined["split_id"], combined_cache(old, new, replace(cfg, seed=918))["split_id"])
        new["actions"][0, 0] = -2.
        self.assertNotEqual(combined["dataset_id"], combined_cache(old, new, cfg)["dataset_id"])
        old["states"][0, 6] += 3.
        moved = combined_cache(old, new, cfg)
        self.assertNotEqual(moved["group_ids"][0], moved["group_ids"][4])
        self.assertEqual(start_groups(moved, .25), moved["group_ids"])


if __name__ == "__main__":
    unittest.main()
