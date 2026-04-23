import importlib.metadata
import importlib.util

from packaging import version


def package_version_is_at_least(
    package_name: str,
    required_version: str,
) -> bool:
    if importlib.util.find_spec(package_name) is None:
        return False

    current_version = importlib.metadata.version(package_name)

    return version.parse(current_version) >= version.parse(required_version)
