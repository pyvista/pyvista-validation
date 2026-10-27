"""Drop the lines coverage excludes from mypy's Cobertura typing report."""

from __future__ import annotations

from pathlib import Path
import re
import sys
import tomllib
import xml.etree.ElementTree as ET


def _code(line: str) -> str:
    """Return ``line`` without its trailing comment and whitespace."""
    return line.split('#', 1)[0].rstrip()


def excluded_lines(source: str, patterns: list[re.Pattern[str]]) -> set[int]:
    """Return the lines the patterns match, extended to whole statements and their blocks."""
    lines = source.split('\n')
    excluded: set[int] = set()
    for pattern in patterns:
        for match in pattern.finditer(source):
            first = source.count('\n', 0, match.start()) + 1
            if not lines[first - 1].strip():
                first += 1
            last = source.count('\n', 0, match.end()) + 1
            depth = sum(
                _code(line).count('(') - _code(line).count(')') for line in lines[first - 1 : last]
            )
            while depth > 0 and last < len(lines):
                last += 1
                depth += _code(lines[last - 1]).count('(') - _code(lines[last - 1]).count(')')
            excluded.update(range(first, last + 1))
            opener = lines[first - 1]
            if _code(lines[last - 1]).endswith(':'):
                indent = len(opener) - len(opener.lstrip())
                for number in range(last + 1, len(lines) + 1):
                    line = lines[number - 1]
                    if line.strip() and len(line) - len(line.lstrip()) <= indent:
                        break
                    excluded.add(number)
    return excluded


def main(report: Path) -> None:
    """Rewrite ``report`` without the lines ``[tool.coverage.report]`` excludes."""
    config = tomllib.loads(Path('pyproject.toml').read_text())
    patterns = [
        re.compile(p, re.MULTILINE) for p in config['tool']['coverage']['report']['exclude_also']
    ]
    tree = ET.parse(report)
    for cls in tree.iter('class'):
        lines = cls.find('lines')
        filename = cls.get('filename')
        if lines is None or not filename:
            continue
        excluded = excluded_lines(Path(filename).read_text(), patterns)
        for line in list(lines):
            if int(line.get('number', '0')) in excluded:
                lines.remove(line)
    total = sum(1 for _ in tree.iter('line'))
    hit = sum(1 for line in tree.iter('line') if int(line.get('hits', '0')) > 0)
    root = tree.getroot()
    root.set('lines-valid', str(total))
    root.set('lines-covered', str(hit))
    root.set('line-rate', f'{hit / total:.4f}' if total else '1')
    tree.write(report)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
