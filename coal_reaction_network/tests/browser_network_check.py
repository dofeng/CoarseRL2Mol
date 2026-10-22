"""Optional Chromium acceptance checks against a read-only existing v3 database."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dashboard_app import start_dashboard
from playwright.sync_api import sync_playwright


def loaded(page):
    page.wait_for_function("() => state.channels.length>0 && !document.querySelector('#graphArea').classList.contains('loading')", timeout=65000)


def pixels(page):
    markup=page.evaluate("""() => {
      const source=document.querySelector('#networkSvg'),svg=source.cloneNode(true),box=source.viewBox.baseVal;
      svg.setAttribute('width',box.width);svg.setAttribute('height',box.height);
      return new XMLSerializer().serializeToString(svg);
    }""")
    page.unroute("**/__check-network.svg")
    page.route("**/__check-network.svg",lambda route:route.fulfill(status=200,content_type="image/svg+xml",body=markup))
    return page.evaluate("""async () => {
      const image=new Image();
      image.src='/__check-network.svg';
      await image.decode();
      const canvas=document.createElement('canvas');canvas.width=550;canvas.height=450;
      const context=canvas.getContext('2d');context.fillStyle='white';context.fillRect(0,0,550,450);
      context.drawImage(image,0,0,550,450);
      const data=context.getImageData(0,0,550,450).data;let colored=0;
      for(let index=0;index<data.length;index+=4)if(Math.min(data[index],data[index+1],data[index+2])<200)colored++;
      return colored/(550*450);
    }""")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--screenshots", type=Path, default=Path("/tmp/coal-ui-screenshots"))
    args=parser.parse_args()
    args.screenshots.mkdir(parents=True,exist_ok=True)
    server,thread=start_dashboard(port=0)
    try:
        with sync_playwright() as playwright:
            browser=playwright.chromium.launch(headless=True,args=["--no-sandbox"])
            page=browser.new_page(viewport={"width":1440,"height":1000},device_scale_factor=1)
            page.add_init_script("localStorage.setItem('coal-atlas-db',"+json.dumps(str(args.database.resolve()))+");")
            errors=[]
            page.on("pageerror",lambda error:errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}")
            loaded(page)
            assert page.locator("#representation").input_value()=="selected"
            assert page.locator("#activity").input_value()=="changes"
            assert page.evaluate("state.channels.every(channel=>Object.keys(channel.net_coefficients).length>0)")
            assert page.evaluate("state.summary.reaction_events===state.coarseDiagnostics.event_totals.reactions")
            assert page.locator(".tools svg.lucide").count()==5
            assert pixels(page)>.008
            page.screenshot(path=str(args.screenshots/"desktop.png"))
            page.locator(".reaction").first.click()
            page.wait_for_selector("#eventList .event-card")
            assert page.locator("#detail").inner_text().find("每次事件的类别净变化")>=0
            page.locator(".event-card").first.click()
            page.locator(".fragment-button").first.click()
            page.wait_for_selector(".atom-list")
            assert page.locator(".atom-list").first.inner_text()
            page.click("#backEvent")
            assert page.locator(".fragment-button").count()>0
            page.click("#backChannel")
            page.wait_for_selector(".event-card")
            page.click("#tableTab")
            assert page.locator("#tableArea tr[data-index]").count()>0
            page.evaluate("scrollTo(0,0)")
            page.screenshot(path=str(args.screenshots/"table.png"))
            page.click("#graphTab")
            page.select_option("#activity","internal")
            page.click("#applyFilters")
            loaded(page)
            assert page.evaluate("state.channels.every(channel=>channel.activity==='internal')")
            page.select_option("#activity","all")
            page.select_option("#flowWeight","carbon")
            page.select_option("#representation","flow")
            page.wait_for_selector(".flow-node",timeout=65000)
            loaded(page)
            assert page.evaluate("!state.coarseNetwork.include_internal && state.coarseNetwork.total_weight===state.coarseNetwork.flow_totals.cross_carbon")
            assert pixels(page)>.003
            page.evaluate("scrollTo(0,0)")
            page.screenshot(path=str(args.screenshots/"flow.png"))
            page.click("#resetFilters")
            loaded(page)
            for width,height in ((1720,1080),(1024,900),(390,844)):
                page.set_viewport_size({"width":width,"height":height})
                page.wait_for_timeout(200)
                page.click("#fit")
                assert page.evaluate("document.documentElement.scrollWidth<=innerWidth+1")
                if width==390:
                    scale=page.evaluate("state.displayScale")
                    page.click("#zoomOut")
                    assert page.evaluate("state.displayScale")<scale
                    page.click("#fit")
                    assert page.locator("#networkSvg").bounding_box()["width"]<=page.locator("#graphArea").bounding_box()["width"]
                    assert not page.locator("#filterForm").is_visible()
                    assert page.locator("#graphArea").bounding_box()["y"]<500
                    page.click("#toggleFilters")
                    assert page.locator("#filterForm").is_visible()
                    page.click("#toggleFilters")
                page.evaluate("scrollTo(0,0)")
                page.screenshot(path=str(args.screenshots/f"viewport-{width}.png"))
                page.locator("#graphArea").scroll_into_view_if_needed()
                assert page.locator("#graphArea").bounding_box()["width"]>250
                page.locator("#graphArea").screenshot(path=str(args.screenshots/f"graph-{width}.png"))
                assert pixels(page)>.008
            page.click("#toggleSetup")
            assert page.locator("dialog").is_visible()
            page.click("#closeSetup")
            assert not page.locator("dialog").is_visible()
            assert not errors,errors
            print("PASS: real data, event/fragment evidence, activity filters, cross-category flow, desktop/tablet/mobile, SVG pixels, local icons, task dialog")
            print(f"Screenshots: {args.screenshots}")
            browser.close()
    finally:
        server.shutdown();server.server_close();thread.join()


if __name__=="__main__":
    main()
