#!/usr/bin/env python3
"""Exercise Material and Zensical builds in a browser and compare key states."""

from __future__ import annotations

import argparse
import json
import threading
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image, ImageChops
from playwright.sync_api import Page, sync_playwright

ROUTES = (
    ("home", "/"),
    ("guide", "/getting-started/first-program/"),
    ("api", "/api/models/LM/"),
    ("notebook", "/tutorials/rag/"),
)
SEARCH_QUERIES = ("configure cache", "MIPROv2", "saving and loading")
SEARCH_EXPECTATIONS = {
    "configure cache": "/api/utils/configure_cache/",
    "MIPROv2": "/api/optimizers/MIPROv2/",
    "saving and loading": "/tutorials/saving/",
}
REQUIRED_TABS = {"Overview", "Getting Started", "Diving Deeper", "Tutorials", "API Reference", "Community", "FAQ"}


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


@contextmanager
def serve(directory: Path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(directory)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()


def stable_page(page: Page, url: str) -> None:
    page.goto(url, wait_until="networkidle")
    page.evaluate("document.fonts.ready")
    page.add_style_tag(
        content="*,*::before,*::after{animation:none!important;transition:none!important;caret-color:transparent!important}"
    )


def selector_options(page: Page) -> list[str]:
    selector = page.locator(".md-version")
    selector.wait_for()
    return selector.locator(".md-version__link").evaluate_all(
        "nodes => nodes.map(node => node.childNodes[0].textContent.trim())"
    )


def search_results(page: Page, query: str) -> list[str]:
    material_field = page.locator('input[data-md-component="search-query"]')
    if material_field.count():
        if not page.locator("#__search").is_checked():
            page.locator('label[for="__search"]:visible').first.click(force=True)
        field = material_field
        links = page.locator(".md-search-result__link")
    else:
        page.locator(".md-search__button:visible").click()
        field = page.locator('input[placeholder="Search"]:visible')
        links = field.locator("../../..").locator("a")
    field.click()
    field.fill("")
    field.press_sequentially(query, delay=20)
    links.first.wait_for()
    results = []
    for href in links.evaluate_all("nodes => nodes.slice(0, 10).map(node => node.href)"):
        parsed = urlparse(href)
        results.append(f"{parsed.path}#{parsed.fragment}".rstrip("#"))
    return results


def exercise(page: Page, base_url: str) -> dict[str, object]:
    stable_page(page, f"{base_url}/")
    options = selector_options(page)
    version_button = page.locator(".md-version__current")
    version_list = page.locator(".md-version__list")
    version_button.focus()
    version_list.wait_for(state="visible")
    version_menu = version_list.is_visible()
    version_links = page.locator(".md-version__link").count()
    version_geometry = version_list.evaluate(
        "element => ({ width: element.getBoundingClientRect().width, "
        "height: element.getBoundingClientRect().height, overflowY: getComputedStyle(element).overflowY })"
    )
    aliases_hidden = page.locator(".md-version__alias").evaluate_all(
        "elements => elements.every(element => getComputedStyle(element).display === 'none')"
    )
    search_geometry = page.locator(".md-search").evaluate(
        "element => ({ width: element.getBoundingClientRect().width, height: element.getBoundingClientRect().height })"
    )
    tabs = {" ".join(text.split()) for text in page.locator(".md-tabs a:visible").all_text_contents()}
    scripts = {
        Path(source).name for source in page.locator("script[src]").evaluate_all("nodes => nodes.map(node => node.src)")
    }
    hero_before = page.locator("#hp-hero-code").inner_text()
    page.locator("#hp-btn-lm").click()
    hero_interaction = page.locator("#hp-hero-code").inner_text() != hero_before
    page.locator('label[for="hp-tab-agent"]').click()
    homepage_tabs = page.locator("#hp-tab-agent").is_checked()
    stable_page(page, f"{base_url}/getting-started/first-program/")
    announcement = "DSPy.LM improvements" in " ".join(page.locator(".md-banner").inner_text().split())
    footer_navigation = page.locator(".md-footer__link").count() == 2
    stable_page(page, f"{base_url}/")
    theme = page.locator('label[for="__palette_1"]:visible').first
    theme.click()
    dark_mode = page.locator("body").get_attribute("data-md-color-scheme") == "slate"
    page.reload(wait_until="networkidle")
    theme_persistence = page.locator("body").get_attribute("data-md-color-scheme") == "slate"
    searches = {}
    for query in SEARCH_QUERIES:
        stable_page(page, f"{base_url}/")
        searches[query] = search_results(page, query)

    stable_page(page, f"{base_url}/tutorials/rag/")
    tutorial_navigation = (
        page.locator(".learn-more-item").count() > 0
        and page.locator('.md-nav__item[style*="display: none"]').count() > 0
    )
    page.set_viewport_size({"width": 390, "height": 844})
    stable_page(page, f"{base_url}/getting-started/first-program/")
    page.locator('label[for="__drawer"]:visible').first.click(force=True)
    mobile_navigation = (
        page.locator("#__drawer").is_checked() and page.locator(".md-sidebar--primary a:visible").count() > 0
    )
    page.set_viewport_size({"width": 1440, "height": 1000})
    return {
        "selector_options": options,
        "version_menu": version_menu and version_links == len(options),
        "version_geometry": version_geometry,
        "version_aliases_hidden": aliases_hidden,
        "search_geometry": search_geometry,
        "dark_mode": dark_mode,
        "search": searches,
        "mobile_navigation": mobile_navigation,
        "tabs": sorted(tabs),
        "announcement": announcement,
        "custom_scripts": sorted(scripts & {"runllm-widget.js", "tutorial-nav.js", "hero-interactive.js"}),
        "hero_interaction": hero_interaction,
        "homepage_tabs": homepage_tabs,
        "footer_navigation": footer_navigation,
        "theme_persistence": theme_persistence,
        "tutorial_navigation": tutorial_navigation,
    }


def screenshot_difference(material: Path, zensical: Path) -> dict[str, float]:
    left = Image.open(material).convert("RGB")
    right = Image.open(zensical).convert("RGB")
    if left.size != right.size:
        return {"changed_pixel_ratio": 1.0, "mean_channel_difference": 255.0}
    difference = ImageChops.difference(left, right)
    pixels = list(difference.get_flattened_data())
    changed = sum(max(pixel) > 24 for pixel in pixels) / len(pixels)
    mean = sum(sum(pixel) for pixel in pixels) / (len(pixels) * 3)
    return {"changed_pixel_ratio": round(changed, 6), "mean_channel_difference": round(mean, 3)}


def screenshots(page: Page, base_url: str, output: Path, renderer: str) -> dict[str, Path]:
    paths = {}
    page.set_viewport_size({"width": 1440, "height": 1000})
    for name, route in ROUTES:
        stable_page(page, f"{base_url}{route}")
        path = output / f"{renderer}-{name}.png"
        page.screenshot(path=path)
        paths[name] = path
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--zensical", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--screenshots", type=Path, required=True)
    parser.add_argument("--browser-executable", type=Path)
    parser.add_argument("--no-fail", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.screenshots.mkdir(parents=True, exist_ok=True)
    with serve(args.material.resolve()) as material_url, serve(args.zensical.resolve()) as zensical_url:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                executable_path=str(args.browser_executable) if args.browser_executable else None,
            )
            context = browser.new_context(viewport={"width": 1440, "height": 1000})
            page = context.new_page()
            material_behavior = exercise(page, material_url)
            zensical_behavior = exercise(page, zensical_url)
            material_shots = screenshots(page, material_url, args.screenshots, "material")
            zensical_shots = screenshots(page, zensical_url, args.screenshots, "zensical")
            browser.close()

    differences = {name: screenshot_difference(material_shots[name], zensical_shots[name]) for name, _ in ROUTES}
    behavior_failures = []
    if material_behavior["selector_options"] != zensical_behavior["selector_options"]:
        behavior_failures.append(
            {
                "feature": "version selector options",
                "material": material_behavior["selector_options"],
                "zensical": zensical_behavior["selector_options"],
            }
        )
    for renderer, behavior in (("material", material_behavior), ("zensical", zensical_behavior)):
        for feature in (
            "dark_mode",
            "mobile_navigation",
            "announcement",
            "hero_interaction",
            "homepage_tabs",
            "footer_navigation",
            "theme_persistence",
            "tutorial_navigation",
            "version_menu",
            "version_aliases_hidden",
        ):
            if not behavior[feature]:
                behavior_failures.append({"renderer": renderer, "feature": feature})
        if not 180 <= behavior["version_geometry"]["width"] <= 200:
            behavior_failures.append(
                {"renderer": renderer, "feature": "version menu width", "actual": behavior["version_geometry"]}
            )
        if behavior["version_geometry"]["overflowY"] != "auto":
            behavior_failures.append(
                {"renderer": renderer, "feature": "version menu scrolling", "actual": behavior["version_geometry"]}
            )
        if not 250 <= behavior["search_geometry"]["width"] <= 270:
            behavior_failures.append(
                {"renderer": renderer, "feature": "search width", "actual": behavior["search_geometry"]}
            )
        if not 46 <= behavior["search_geometry"]["height"] <= 50:
            behavior_failures.append(
                {"renderer": renderer, "feature": "search height", "actual": behavior["search_geometry"]}
            )
        missing_tabs = sorted(REQUIRED_TABS - set(behavior["tabs"]))
        if missing_tabs:
            behavior_failures.append(
                {"renderer": renderer, "feature": "primary navigation tabs", "missing": missing_tabs}
            )
        missing_scripts = sorted(
            {"runllm-widget.js", "tutorial-nav.js", "hero-interactive.js"} - set(behavior["custom_scripts"])
        )
        if missing_scripts:
            behavior_failures.append({"renderer": renderer, "feature": "custom scripts", "missing": missing_scripts})
        for query, expected in SEARCH_EXPECTATIONS.items():
            results = behavior["search"][query]
            if not any(result.split("#", 1)[0] == expected for result in results):
                behavior_failures.append(
                    {
                        "renderer": renderer,
                        "feature": "search discoverability",
                        "query": query,
                        "expected": expected,
                        "results": results,
                    }
                )
    visual_failures = {
        name: difference
        for name, difference in differences.items()
        if difference["changed_pixel_ratio"] > 0.05 or difference["mean_channel_difference"] > 5
    }
    report = {
        "passed": not behavior_failures,
        "behavior": {
            "passed": not behavior_failures,
            "material": material_behavior,
            "zensical": zensical_behavior,
            "failures": behavior_failures,
        },
        "visual": {
            "informational": True,
            "differences": differences,
            "review_recommended": visual_failures,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"] and not args.no_fail:
        raise SystemExit(f"Browser parity failed; see {args.report}")


if __name__ == "__main__":
    main()
