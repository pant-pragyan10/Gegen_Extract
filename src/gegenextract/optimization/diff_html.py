from typing import Tuple
from .artifact import PromptArtifact
from difflib import HtmlDiff


def prompt_diff_html(a: PromptArtifact, b: PromptArtifact) -> str:
    a_txt = a.serialize().splitlines()
    b_txt = b.serialize().splitlines()
    hd = HtmlDiff(tabsize=2)
    html = hd.make_file(a_txt, b_txt, fromdesc=f"v{a.version}", todesc=f"v{b.version}")
    return html
