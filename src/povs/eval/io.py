from pathlib import Path

from .reports import TVDPerIterReport
from .utils import yaml


def save_tvd_per_iter_report(path: Path, report: TVDPerIterReport) -> None:
    path.mkdir(parents=True, exist_ok=True)

    details = report._asdict()
    details.pop("plot")
    details.pop("tvds")
    (path / "details.yml").write_text(yaml.dump(details))

    report.tvds.to_csv(path / "tvds.csv", index=False)

    report.plot.savefig(path / "tvd_per_iteration.png")
