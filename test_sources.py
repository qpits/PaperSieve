#!/usr/bin/env python3
"""Offline check of the CVF HTML parsing. Run: python test_sources.py

Fixtures are trimmed verbatim from openaccess.thecvf.com/WACV2024?day=all and
one of its paper detail pages, so they exercise the real markup shape: entities
in the abstract, an entry with no arXiv link, and the trailing page footer that
follows the last <dt>.
"""

from sources import parse_cvf_abstract, parse_cvf_listing

LISTING = """<div id="content">
<h3>Papers</h3>
<dl>
<dt class="ptitle"><br><a href="/content/WACV2024/html/Zhang_Object-Centric_Video_WACV_2024_paper.html">Object-Centric Video &amp; Action Anticipation</a></dt>
<dd>
<form id="form-CeZhang" action="/WACV2024" method="post" class="authsearch">
<input type="hidden" name="query_author" value="Ce Zhang">
<a href="#" onclick="submit();">Ce Zhang</a>,
</form>
<form id="form-ChenSun" action="/WACV2024" method="post" class="authsearch">
<input type="hidden" name="query_author" value="Chen Sun">
<a href="#" onclick="submit();">Chen Sun</a>
</form>
</dd>
<dd>
[<a href="/content/WACV2024/papers/Zhang_Object-Centric_Video_WACV_2024_paper.pdf">pdf</a>]
[<a href="http://arxiv.org/abs/2311.00180">arXiv</a>]
<div class="link2">[<a class="fakelink">bibtex</a>]
<div class="bibref pre-white-space">@InProceedings{Zhang_2024_WACV}</div>
</div>
</dd>
<dt class="ptitle"><br><a href="/content/WACV2024/html/Doe_A_Second_Paper_WACV_2024_paper.html">A Second Paper</a></dt>
<dd>
<form id="form-JaneDoe" action="/WACV2024" method="post" class="authsearch">
<input type="hidden" name="query_author" value="Jane Doe">
<a href="#" onclick="submit();">Jane Doe</a>
</form>
</dd>
<dd>
[<a href="/content/WACV2024/papers/Doe_A_Second_Paper_WACV_2024_paper.pdf">pdf</a>]
</dd>
</dl>
</div>
<div id="footer">Copyright notice</div>
"""

DETAIL = """<div id="content">
<div id="papertitle">Object-Centric Video &amp; Action Anticipation</div>
<div id="authors"><b><i>Ce Zhang, Chen Sun</i></b>; Proceedings of WACV, 2024, pp. 6751-6761</div>
<div id="abstract">
    We build object-centric representations, since a &quot;background&quot; object
    may be used by the actor later.
</div>
<div class="link2">[<a href="x.pdf">pdf</a>]</div>
</div>
"""

DETAIL_NO_ABSTRACT = '<div id="content"><div id="papertitle">Broken</div></div>'


def test_listing():
    papers = parse_cvf_listing(LISTING, "WACV2024")
    assert len(papers) == 2, papers

    first = papers[0]
    assert first["title"] == "Object-Centric Video & Action Anticipation", first["title"]
    assert first["id"] == "Zhang_Object-Centric_Video_WACV_2024", first["id"]
    assert first["authors"] == ["Ce Zhang", "Chen Sun"], first["authors"]
    assert first["arxiv_url"] == "http://arxiv.org/abs/2311.00180", first["arxiv_url"]
    assert first["forum_url"] == (
        "https://openaccess.thecvf.com/content/WACV2024/html/"
        "Zhang_Object-Centric_Video_WACV_2024_paper.html"
    ), first["forum_url"]
    assert first["venue"] == "WACV2024"

    # entries without an arXiv link must still parse, with an empty arxiv_url
    assert papers[1]["title"] == "A Second Paper"
    assert papers[1]["authors"] == ["Jane Doe"]
    assert papers[1]["arxiv_url"] == ""

    # the schema the rest of the pipeline reads
    for paper in papers:
        assert set(paper) == {
            "id", "number", "title", "abstract", "tldr", "keywords", "authors",
            "primary_area", "venue", "venueid", "forum_url", "arxiv_url",
        }, sorted(paper)


def test_abstract():
    abstract = parse_cvf_abstract(DETAIL)
    assert abstract.startswith("We build object-centric"), abstract
    assert '"background"' in abstract, abstract       # entities unescaped
    assert "<" not in abstract and "\n    " not in abstract, abstract
    assert parse_cvf_abstract(DETAIL_NO_ABSTRACT) == ""


def test_no_papers_is_detectable():
    # a pre-2021 venue (or a typo'd id) returns a page with no <dt class="ptitle">;
    # fetch_cvf turns an empty list into an explanatory error.
    assert parse_cvf_listing("<html><body>no papers here</body></html>", "CVPR2020") == []


if __name__ == "__main__":
    test_listing()
    test_abstract()
    test_no_papers_is_detectable()
    print("ok")
