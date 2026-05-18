from pathlib import Path

from .reports import TVDPerIterReport
from .utils import yaml


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
