import statistics

from tico import convert

from test.performance.utils import (
    load_model,
    measure_time,
    size_of_bytes,
    temp_state_dict_size,
)


def run_benchmark(model_name, speed_threshold, size_threshold):
    print(f"Start performance test with {model_name}")

    model, inputs = load_model(model_name)

    # Conversion speed benchmark
    timings = measure_time(convert, model, inputs)
    mean_time = statistics.mean(timings) * model.num_hidden_layers()  # type: ignore[operator]
    print(
        f"Mean conversion time (Single decoder layer * num_hidden_layers): {mean_time:.2f}s (threshold {speed_threshold}s)"
    )
    if mean_time > speed_threshold:
        raise AssertionError(
            f"Conversion too slow: {mean_time:.2f}s exceeds threshold {speed_threshold}s"
        )

    # Model size benchmark
    circle_model = convert(model, inputs)
    circle_size = size_of_bytes(circle_model.circle_binary)
    state_dict_size = temp_state_dict_size(model)
    print(f"Circle size: {circle_size} bytes")
    print(f"State dict size: {state_dict_size} bytes")
    print(f"Circle / State dict ratio: {circle_size / state_dict_size}")
    if circle_size > state_dict_size * size_threshold:
        raise AssertionError(
            f"Circle size {circle_size} is increased by {(size_threshold - 1)*100}% compared to state_dict size {state_dict_size}"
        )


if __name__ == "__main__":
    models = ["Llama-3.2-1B", "Llama-3.2-3B"]

    # Conversion speed thresholds (seconds)
    speed_thresholds = [60, 180]

    # Model size increase thresholds (1.01 = 1% increase)
    size_thresholds = [1.01, 1.01]

    for model, speed_threshold, size_threshold in zip(
        models, speed_thresholds, size_thresholds
    ):
        run_benchmark(model, speed_threshold, size_threshold)
