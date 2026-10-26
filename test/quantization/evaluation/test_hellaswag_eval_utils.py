# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from tico.quantization.evaluation.hellaswag_eval_utils import get_hellaswag_accuracy


class TestGetHellaswagAccuracy(unittest.TestCase):
    def test_reads_lm_eval_0_4_filtered_metric_keys(self):
        # lm_eval >= 0.4 stores "<metric>,<filter>" keys.
        results = {
            "results": {
                "hellaswag": {
                    "alias": "hellaswag",
                    "acc,none": 0.322,
                    "acc_stderr,none": 0.0148,
                    "acc_norm,none": 0.381,
                    "acc_norm_stderr,none": 0.0154,
                }
            }
        }
        self.assertEqual(
            get_hellaswag_accuracy(results), {"acc": 0.322, "acc_norm": 0.381}
        )

    def test_reads_legacy_bare_metric_keys(self):
        results = {"results": {"hellaswag": {"acc": 0.5, "acc_norm": 0.6}}}
        self.assertEqual(get_hellaswag_accuracy(results), {"acc": 0.5, "acc_norm": 0.6})

    def test_bare_key_takes_precedence_over_filtered_key(self):
        results = {
            "results": {
                "hellaswag": {
                    "acc": 0.5,
                    "acc,none": 0.1,
                    "acc_norm": 0.6,
                    "acc_norm,none": 0.2,
                }
            }
        }
        self.assertEqual(get_hellaswag_accuracy(results), {"acc": 0.5, "acc_norm": 0.6})

    def test_does_not_confuse_stderr_or_prefix_metrics(self):
        # "acc_stderr,none" and "acc_norm,none" must not satisfy the "acc" lookup.
        results = {
            "results": {"hellaswag": {"acc_stderr,none": 0.01, "acc_norm,none": 0.381}}
        }
        with self.assertRaises(KeyError):
            get_hellaswag_accuracy(results)

    def test_missing_metric_raises_instead_of_reporting_zero(self):
        with self.assertRaises(KeyError):
            get_hellaswag_accuracy({"results": {"hellaswag": {"acc,none": 0.3}}})

    def test_missing_task_entry_raises(self):
        with self.assertRaises(KeyError):
            get_hellaswag_accuracy({"results": {}})
        with self.assertRaises(KeyError):
            get_hellaswag_accuracy({})

    def test_ambiguous_filters_raise(self):
        results = {
            "results": {
                "hellaswag": {
                    "acc,none": 0.3,
                    "acc,strict-match": 0.2,
                    "acc_norm,none": 0.4,
                }
            }
        }
        with self.assertRaises(KeyError):
            get_hellaswag_accuracy(results)


if __name__ == "__main__":
    unittest.main()
