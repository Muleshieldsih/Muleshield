# -*- coding: utf-8 -*-
"""
Render each .dc.html artboard to a high-DPI PNG for embedding in the PPTX.

The artboards are Design Component files: a <helmet> block holding the styles
and one root <div> that is the artboard itself. A browser renders both without
knowing the custom elements -- <style> and <link> apply from anywhere in the
DOM -- so screenshotting that root div gives exactly the frame, with no page
margin to crop off. The ./support.js 404 is expected and harmless; nothing in
these artboards is scripted.
"""

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
OUT = HERE / "png"
SCALE = 3  # 3x so the images stay sharp when PowerPoint scales them into a slide


def main() -> None:
    names = sys.argv[1:] or [p.name for p in sorted(HERE.glob("*.dc.html"))]
    OUT.mkdir(exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 1100},
                                device_scale_factor=SCALE)
        for name in names:
            src = HERE / name
            page.goto(src.as_uri(), wait_until="load")
            # Webfonts arrive after load; without this the shot can capture the
            # fallback face mid-swap and the metrics shift under the text.
            page.evaluate("() => document.fonts.ready")
            page.wait_for_timeout(900)

            root = page.locator("x-dc > div").first
            box = root.bounding_box()

            # An artboard's height is declared, not derived, so content that grows
            # past it is silently cut off at the frame edge -- the flow chart lost
            # its last two boxes that way and the clipped PNG looked deliberate.
            # scrollHeight tells us what the content actually needed.
            overflow = root.evaluate(
                "el => Math.max(0, el.scrollHeight - el.clientHeight)")
            if overflow > 1:
                raise SystemExit(
                    f"{src.name}: content overflows the frame by {overflow}px "
                    f"-- raise the artboard height instead of shipping a clipped image."
                )

            dest = OUT / (src.name.replace(".dc.html", "") + ".png")
            root.screenshot(path=str(dest))
            print(f"  {dest.name:20} {int(box['width'])}x{int(box['height'])} css px"
                  f"  -> {dest.stat().st_size // 1024} KB")
        browser.close()


if __name__ == "__main__":
    main()
