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


def init_args(*args, **kwargs):
    def __inner_init_args(orig_class):
        orig_init = orig_class.__init__
        # Make copy of original __init__, so we can call it without recursion

        def __init__(self, *args_, **kwargs_):
            args_ = (*args, *args_)
            kwargs_ = {**kwargs, **kwargs_}

            orig_init(self, *args_, **kwargs_)  # Call the original __init__

        orig_class.__init__ = __init__
        return orig_class

    return __inner_init_args


def is_tagged(cls, tag: str):
    return hasattr(cls, f"__tag_{tag}")
