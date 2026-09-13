"""MkDocs hook: add the copy-to-clipboard button to fenced code blocks inside
notebook markdown cells (e.g. `pip install ...` snippets).

mkdocs-jupyter already renders a working copy button (a `<clipboard-copy>`
custom element, styled to match the rest of the notebook) on actual code
cells. But fenced code blocks written inside a notebook's markdown cells go
through a different renderer on the way to HTML and end up as a plain
`<div class="highlight"><pre>...</pre></div>` with no button at all - not
even Material's native one, since that requires a `<code>` element inside
`<pre>` which this renderer never adds.

This hook leaves every already-working button untouched (both the notebook
code-cell widget and Material's own button on regular, non-notebook pages)
and just replicates the same code-cell widget markup onto those bare
`.highlight` blocks so every code block in a tutorial looks and behaves the
same way.
"""

from bs4 import BeautifulSoup

COPY_ICON_SVG = (
    '<svg aria-hidden="true" class="clipboard-copy-icon" data-view-component="true" '
    'height="20" version="1.1" viewbox="0 0 16 16" width="20">'
    '<path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 010 1.5h-1.5a.25.25 0 00-.25.25v7.5c0 '
    ".138.112.25.25.25h7.5a.25.25 0 00.25-.25v-1.5a.75.75 0 011.5 0v1.5A1.75 1.75 0 019.25 16h-7.5A1.75 "
    '1.75 0 010 14.25v-7.5z" fill="currentColor" fill-rule="evenodd"></path>'
    '<path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0114.25 '
    "11h-7.5A1.75 1.75 0 015 9.25v-7.5zm1.75-.25a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 "
    '0 00.25-.25v-7.5a.25.25 0 00-.25-.25h-7.5z" fill="currentColor" fill-rule="evenodd"></path>'
    "</svg>"
)


def on_page_content(html, page, config, files):
    if not page.file.src_uri.endswith(".ipynb") or "highlight" not in html:
        return html

    soup = BeautifulSoup(html, "html.parser")
    changed = False

    for i, div in enumerate(soup.find_all("div", class_="highlight"), start=1):
        pre = div.find("pre", recursive=False)
        if pre is None or div.find("clipboard-copy") is not None:
            continue

        cell_id = f"md-fence-copy-{i}"
        code_text = pre.get_text()

        style = div.get("style", "")
        div["style"] = f"{style};position:relative".lstrip(";")

        widget = BeautifulSoup(
            '<div class="zeroclipboard-container">'
            f'<clipboard-copy aria-label="Copy to Clipboard" for="{cell_id}">'
            f"<div><span class=\"notice\" hidden>Copied!</span>{COPY_ICON_SVG}</div>"
            "</clipboard-copy></div>",
            "html.parser",
        )
        div.insert(0, widget)

        txt_div = soup.new_tag("div", attrs={"class": "clipboard-copy-txt", "id": cell_id})
        txt_div.string = code_text
        div.append(txt_div)
        changed = True

    return str(soup) if changed else html
