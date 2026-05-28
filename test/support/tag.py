# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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


def skip(reason):
    def __inner_skip(orig_class):
        setattr(orig_class, "__tag_skip", True)
        setattr(orig_class, "__tag_skip_reason", reason)

        def __init__(self, *args_, **kwargs_):
            pass

        # Ignore initialization of skipped modules
        orig_class.__init__ = __init__

        return orig_class

    return __inner_skip


def skip_if(predicate, reason):
    def __inner_skip(orig_class):
        setattr(orig_class, "__tag_skip", True)
        setattr(orig_class, "__tag_skip_reason", reason)

        def __init__(self, *args_, **kwargs_):
            pass

        # Ignore initialization of skipped modules
        orig_class.__init__ = __init__

        return orig_class

    if predicate:
        return __inner_skip
    else:
        return lambda x: x


def test_without_inference(orig_class):
    setattr(orig_class, "__tag_test_without_inference", True)
    return orig_class


def test_without_pt2(orig_class):
    setattr(orig_class, "__tag_test_without_pt2", True)
    return orig_class


def test_negative(expected_err):
    def __inner_test_negative(orig_class):
        setattr(orig_class, "__tag_test_negative", True)
        setattr(orig_class, "__tag_expected_err", expected_err)

        return orig_class

    return __inner_test_negative


def target(orig_class):
    setattr(orig_class, "__tag_target", True)
    return orig_class


def use_onert(orig_class):
    """
    Decorator to mark a test class so that Circle models are executed
     with the 'onert' runtime.

    Useful when the default 'circle-interpreter' cannot run the model
     under test.
    """
    setattr(orig_class, "__tag_use_onert", True)
    return orig_class


def with_golden(orig_class):
    """
    Decorator to mark a test class so that it should be compared against with_golden outputs.
    'get_golden_outputs' must return a list of tensors which will be compared against the output of the model under test.
    """
    setattr(orig_class, "__tag_with_golden", True)

    assert hasattr(
        orig_class, "get_golden_outputs"
    ), f"{orig_class} with_golden test must implement get_golden_outputs()"
    return orig_class


def is_tagged(cls, tag: str):
    return hasattr(cls, f"__tag_{tag}")
