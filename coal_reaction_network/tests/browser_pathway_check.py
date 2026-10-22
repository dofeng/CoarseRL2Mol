"""Optional real Chromium acceptance check; requires Playwright outside runtime dependencies."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))

from playwright.sync_api import sync_playwright
from dashboard_app import start_dashboard
from network_postprocess import build_network_views
from test_regressions import Workspace
from trajectory_pipeline import _rebuild_reaction_channels


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--result',type=Path,help='Optional existing pathway JSON for read-only rendering checks')
    arguments=parser.parse_args()
    fixture=Workspace()
    fixture.setUp()
    server=None
    try:
        chain=[(atom,atom+1) for atom in range(1,16)]
        halves=[edge for edge in chain if edge[0]!=5]
        quarters=[edge for edge in chain if edge[0] not in {5,9,13}]
        pairs=[edge for edge in chain if edge[0]%2]
        database,connection=fixture.database([chain,halves,quarters,pairs,quarters,halves,chain],elements=(1,)*16)
        build_network_views(connection)
        _rebuild_reaction_channels(connection)
        with connection:
            connection.execute("INSERT OR REPLACE INTO metadata VALUES ('views_ready','true')")
        connection.close()
        server,thread=start_dashboard(port=0)
        server.pathways.root=fixture.root/'pathways'
        with sync_playwright() as playwright:
            browser=playwright.chromium.launch(headless=True,args=['--no-sandbox'])
            page=browser.new_page(viewport={'width':1720,'height':1080},device_scale_factor=1)
            errors=[]
            page.on('pageerror',lambda error: errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{server.server_port}')
            page.fill('#database',str(database))
            page.click('#openDb')
            page.wait_for_selector('#species option[value="C6-20_R0_O0"]',state='attached')
            page.wait_for_timeout(300)
            page.select_option('#representation','pathflow')
            page.fill('#pathFragment','1')
            page.click('#pathStart')
            page.wait_for_selector('[data-path-node]',timeout=30000)
            page.locator('#graphArea').scroll_into_view_if_needed()
            assert page.locator('#representation').input_value()=='pathflow'
            assert all(' C ' in value for value in page.locator('.path-curve').evaluate_all('(nodes)=>nodes.map(node=>node.getAttribute("d"))'))
            page.locator('#graphArea').screenshot(path='/tmp/coal-process-flow.png')
            page.locator('[data-path-node="overview-p0"] rect').click()
            assert '多事件' in page.locator('#detail').inner_text() or '不是单步反应' in page.locator('#detail').inner_text()
            page.select_option('#representation','path3d')
            page.locator('#graphArea').scroll_into_view_if_needed()
            page.locator('#graphArea').screenshot(path='/tmp/coal-pathway-3d-default.png')
            before=page.locator('#networkSvg').get_attribute('viewBox')
            area=page.locator('#graphArea').bounding_box()
            page.mouse.move(area['x']+100,area['y']+100)
            page.mouse.wheel(0,350)
            page.wait_for_timeout(200)
            assert page.locator('#networkSvg').get_attribute('viewBox')==before
            initial=page.locator('.path-node circle').first.get_attribute('cx')
            page.evaluate('window.savedSvg=document.querySelector("#networkSvg");window.savedTable=document.querySelector("#tableArea table");')
            page.mouse.move(area['x']+60,area['y']+55)
            page.mouse.down()
            page.mouse.move(area['x']+220,area['y']+120,steps=12)
            page.mouse.up()
            page.wait_for_timeout(200)
            assert page.locator('.path-node circle').first.get_attribute('cx')!=initial
            assert page.evaluate('window.savedSvg===document.querySelector("#networkSvg") && window.savedTable===document.querySelector("#tableArea table")')
            previous=page.locator('#graphScale').input_value()
            page.mouse.move(area['x']+300,area['y']+250)
            page.mouse.down();page.mouse.wheel(0,-220);page.wait_for_timeout(200);page.mouse.up()
            assert float(page.locator('#graphScale').input_value())>float(previous)
            previous=page.locator('#graphScale').input_value()
            page.mouse.wheel(0,-220);page.wait_for_timeout(150)
            assert page.locator('#graphScale').input_value()==previous
            page.locator('#graphArea').screenshot(path='/tmp/coal-pathway-3d.png')
            page.locator('#graphScale').evaluate('(element)=>{element.value=5;element.dispatchEvent(new Event("input"));}')
            assert page.locator('#pathScaleValue').inner_text()=='5.0%'
            page.click('#fit')
            assert page.evaluate('state.pathScene.nodes.every(entry=>{const point=state.pathScene.project(entry.node);return point.sx>=40 && point.sx<=state.pathScene.width-40 && point.sy>=50 && point.sy<=state.pathScene.height-40;})')
            first_position=page.locator('.path-node circle').first.get_attribute('cx')
            page.locator('#pathBrowse').evaluate('(element)=>{element.value=900;element.dispatchEvent(new Event("input"));}')
            page.wait_for_timeout(100)
            assert page.locator('#pathBrowseControls').is_visible()
            assert page.locator('.path-node circle').first.get_attribute('cx')!=first_position
            page.evaluate('state.pathwaySelected=state.pathScene.nodes[0].node.id;')
            page.click('#pathFocus')
            assert page.evaluate('Math.abs(state.pathScene.project(state.pathScene.nodes[0].node).sx-state.pathScene.width/2)<1')
            page.set_viewport_size({'width':1500,'height':1000})
            page.wait_for_timeout(200)
            assert page.evaluate('Math.abs(state.pathScene.width-document.querySelector("#graphArea").clientWidth)<1')
            page.select_option('#representation','path2d')
            page.locator('#graphScale').evaluate('(element)=>{element.value=150;element.dispatchEvent(new Event("input"));}')
            page.wait_for_timeout(200)
            page.locator('#graphArea').scroll_into_view_if_needed()
            print('2D metrics',page.locator('#graphArea').evaluate('(element)=>({height:element.clientHeight,scrollHeight:element.scrollHeight,html:element.querySelector("svg").getAttribute("viewBox")})'))
            page.locator('#graphArea').evaluate('(element)=>{element.scrollTop=0;}')
            area=page.locator('#graphArea').bounding_box()
            page.mouse.move(area['x']+100,area['y']+100)
            page.keyboard.down('Shift');page.mouse.wheel(0,300);page.keyboard.up('Shift')
            page.wait_for_timeout(100)
            assert page.locator('#graphArea').evaluate('(element)=>element.scrollLeft')>0
            area=page.locator('#graphArea').bounding_box()
            page.mouse.move(area['x']+100,area['y']+100)
            before=page.locator('#networkSvg').get_attribute('viewBox')
            page.mouse.wheel(0,450)
            page.wait_for_timeout(200)
            assert page.locator('#graphArea').evaluate('(element)=>element.scrollTop')>0
            assert page.locator('#networkSvg').get_attribute('viewBox')==before
            page.locator('#graphArea').evaluate('(element)=>{element.scrollTop=0;}')
            page.locator('#graphArea').screenshot(path='/tmp/coal-pathway-2d.png')
            page.select_option('#representation','all')
            page.wait_for_selector('#networkSvg .reaction')
            before=page.locator('#networkSvg').get_attribute('viewBox')
            page.mouse.wheel(0,300)
            page.wait_for_timeout(100)
            assert page.locator('#networkSvg').get_attribute('viewBox')==before
            if arguments.result:
                data=json.loads(arguments.result.read_text())
                page.evaluate('(data)=>{state.pathway=data;state.pathwayLayouts={};}',data)
                for mode in ('pathflow','pathdetail3d','path3d'):
                    page.select_option('#representation',mode)
                    page.click('#fit')
                    page.locator('#graphArea').screenshot(path=f'/tmp/coal-real-{mode}.png')
                    assert page.locator('.path-node').count()>0
                    assert page.evaluate('state.pathScene.nodes.every(entry=>Number.isFinite(entry.node.x+entry.node.y+entry.node.z))')
                print('PASS: precomputed real trajectory rendered in 2D and both 3D views')
            assert not errors,errors
            browser.close()
        print('PASS: real browser rendering, 3D rotation, 2D scrolling, legacy wheel behavior; screenshots /tmp/coal-pathway-{2d,3d}.png')
    finally:
        if server:
            server.shutdown();server.server_close();thread.join()
        fixture.tearDown()


if __name__=='__main__':
    main()
