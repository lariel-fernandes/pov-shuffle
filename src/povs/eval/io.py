from pathlib import Path

from .reports import BiasPerIterReport, BreakingPointPerDeckSizeReport, TimePerDeckSizeReport
from .utils import yaml


def save_time_per_deck_size_report(path: Path, report: TimePerDeckSizeReport) -> None:
    path.mkdir(parents=True, exist_ok=True)

    details = report._asdict()
    plot = details.pop("plot")
    timings = details.pop("timings")

    (path / "details.yml").write_text(yaml.dump(details))
    (path / "README.md").write_text(report.__doc__)
    timings.to_csv(path / "timings.csv", index=False)
    plot.savefig(path / "plot.png")


def save_breaking_point_report(path: Path, report: BreakingPointPerDeckSizeReport) -> None:
    path.mkdir(parents=True, exist_ok=True)

    details = report._asdict()
    plot = details.pop("plot")
    breaking_points = details.pop("breaking_points")

    (path / "details.yml").write_text(yaml.dump(details))
    (path / "README.md").write_text(report.__doc__)
    breaking_points.to_csv(path / "breaking_points.csv", index=False)
    plot.savefig(path / "plot.png")


def save_bias_per_iter_report(path: Path, report: BiasPerIterReport) -> None:
    path.mkdir(parents=True, exist_ok=True)

    details = report._asdict()
    plot = details.pop("plot")
    biases = details.pop("biases")

    (path / "details.yml").write_text(yaml.dump(details))
    (path / "README.md").write_text(report.__doc__)
    biases.to_csv(path / "biases.csv", index=False)
    plot.savefig(path / "plot.png")
