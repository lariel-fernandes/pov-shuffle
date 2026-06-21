from pathlib import Path

from .reports import BreakingPointPerDeckSizeReport, TimePerDeckSizeReport, TVDPerIterReport
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


def save_tvd_per_iter_report(path: Path, report: TVDPerIterReport) -> None:
    path.mkdir(parents=True, exist_ok=True)

    details = report._asdict()
    plot = details.pop("plot")
    tvds = details.pop("tvds")
    ngram_tvds = details.pop("ngram_tvds")
    lstm_predictabilities = details.pop("lstm_predictabilities")

    (path / "details.yml").write_text(yaml.dump(details))

    (path / "README.md").write_text(report.__doc__)

    tvds.to_csv(path / "tvds.csv", index=False)

    ngram_tvds.to_csv(path / "ngram_tvds.csv", index=False)

    if lstm_predictabilities is not None:
        lstm_predictabilities.to_csv(path / "lstm_predictabilities.csv", index=False)

    plot.savefig(path / "plot.png")
