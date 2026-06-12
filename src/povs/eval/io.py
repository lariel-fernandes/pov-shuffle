from pathlib import Path

from .reports import TimePerDeckSizeReport, TVDPerIterReport
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


def save_tvd_per_iter_report(path: Path, report: TVDPerIterReport) -> None:
    path.mkdir(parents=True, exist_ok=True)

    details = report._asdict()
    plot = details.pop("plot")
    tvds = details.pop("tvds")
    ngram_tvds = details.pop("ngram_tvds")

    (path / "details.yml").write_text(yaml.dump(details))

    (path / "README.md").write_text(report.__doc__)

    tvds.to_csv(path / "tvds.csv", index=False)

    ngram_tvds.to_csv(path / "ngram_tvds.csv", index=False)

    plot.savefig(path / "plot.png")
