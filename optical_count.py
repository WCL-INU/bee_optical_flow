from pathlib import Path

from bee_entrance_count import Config, compare_videos


DEFAULT_COMPARE_VIDEOS = [
    Path("videos") / "ANU-25-summer-6_20260405_060000.mp4",
    Path("videos") / "ANU-25-summer-6_20260405_070000.mp4",
]


def main():
    compare_videos(
        DEFAULT_COMPARE_VIDEOS,
        Path("bee_count_output") / "persistence_filter",
        Config(),
    )


if __name__ == "__main__":
    main()
