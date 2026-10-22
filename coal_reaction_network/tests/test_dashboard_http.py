import json
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0,str(Path(__file__).parents[1] / "src"))

from dashboard_app import start_dashboard
import test_regressions as fixtures


class DashboardHTTPTests(fixtures.Workspace):
    prepared = fixtures.QueryAndVisualTests.prepared
    def setUp(self):
        super().setUp()
        self.server,self.thread=start_dashboard(port=0)
        self.server.pathways.root=self.root / 'pathway_tasks'
        self.base=f"http://127.0.0.1:{self.server.server_port}"
        self.task_patch=patch("dashboard_app.TASK_ROOT",self.root / "tasks")
        self.task_patch.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.task_patch.stop()
        super().tearDown()

    def request(self,path,body=None,origin=None):
        headers={"Content-Type":"application/json"}
        if origin:
            headers["Origin"]=origin
        request=Request(self.base+path,data=json.dumps(body).encode() if body is not None else None,headers=headers)
        try:
            with urlopen(request,timeout=15) as response:
                return response.status,response.read()
        except HTTPError as error:
            return error.code,error.read()

    def test_static_assets_and_api(self):
        from urllib.parse import urlencode
        database=self.prepared()
        for path in ("/","/app.js","/style.css","/coarse-network.js"):
            status,body=self.request(path)
            self.assertEqual(status,200)
            self.assertTrue(body)
        status,body=self.request("/api/network?"+urlencode({"db":str(database),"limit":1}))
        self.assertEqual(status,200)
        self.assertEqual(len(json.loads(body)["channels"]),1)

    def test_coarse_network_routes_and_readiness(self):
        from urllib.parse import urlencode
        from coarse_network.build import build_analytics
        from trajectory_pipeline import _connect_database
        database=self.prepared()
        query=urlencode({"db":str(database)})
        status,body=self.request("/api/coarse-network?"+query)
        self.assertEqual(status,200)
        self.assertFalse(json.loads(body)["ready"])
        connection=_connect_database(database)
        build_analytics(connection,{1:"C",2:"H",3:"O"})
        with connection:
            connection.execute("INSERT OR REPLACE INTO metadata VALUES ('views_ready','true')")
        connection.close()
        status,body=self.request("/api/coarse-network?"+query)
        self.assertEqual(status,200,body)
        self.assertTrue(json.loads(body)["ready"])
        self.assertGreater(json.loads(body)["total_events"],0)
        status,body=self.request("/api/coarse-diagnostics?"+query)
        self.assertEqual(status,200,body)
        self.assertGreater(json.loads(body)["audit"]["audited_pairs"],0)

    def test_cross_origin_launch_and_task_traversal_rejected(self):
        status,body=self.request("/api/run",{},origin="http://untrusted.example")
        self.assertEqual(status,400)
        self.assertIn("Cross-origin",body.decode())
        status,body=self.request("/api/task?id=../../outside")
        self.assertEqual(status,400)

    def test_pathway_background_query(self):
        database=self.prepared()
        status,body=self.request('/api/pathway/start',{'db':str(database),'start':0,'end':2,'fragment':1})
        self.assertEqual(status,202,body)
        task=json.loads(body)['id']
        for attempt in range(100):
            status,body=self.request('/api/pathway/status?id='+task)
            result=json.loads(body)
            if result['status']!='running':break
            time.sleep(.02)
        self.assertEqual(result['status'],'completed',result)
        status,body=self.request('/api/pathway/result?id='+task)
        self.assertEqual(status,200,body)
        self.assertGreater(json.loads(body)['total_events'],0)

    def test_supervised_pipeline_task(self):
        self.file("bond0.reaxff",((1,2),(2,3),(3,4)))
        self.file("bond1.reaxff",((1,2),(3,4)))
        status,body=self.request("/api/run",{"input":str(self.root),"database":str(self.root / "job/network.sqlite"),"workers":1})
        self.assertEqual(status,202,body)
        task=json.loads(body)["task_id"]
        for attempt in range(100):
            status,body=self.request("/api/task?id="+task)
            result=json.loads(body)
            if result["status"] not in {"running","queued"}:
                break
            time.sleep(.05)
        self.assertEqual(result["status"],"completed",result)
        self.assertEqual(result["progress"]["stage"],"done")

    def test_cancel_then_resume_without_losing_committed_frames(self):
        from benchmark_pipeline import frame_text
        inputs=self.root / "inputs"
        inputs.mkdir()
        text=frame_text(96,False)
        for frame in range(2000):
            (inputs / f"bond{frame}.reaxff").write_text(text)
        payload={"input":str(inputs),"database":str(self.root / "job/network.sqlite"),"workers":1}
        status,body=self.request("/api/run",payload)
        self.assertEqual(status,202,body)
        task=json.loads(body)["task_id"]
        for attempt in range(400):
            status,body=self.request("/api/task?id="+task)
            result=json.loads(body)
            if result.get("progress",{}).get("completed",0)>0:
                break
            time.sleep(.02)
        self.assertEqual(result["status"],"running")
        status,body=self.request("/api/run",payload)
        self.assertEqual(status,400)
        status,body=self.request("/api/cancel",{"task_id":task})
        self.assertEqual(status,200,body)
        for attempt in range(200):
            status,body=self.request("/api/task?id="+task)
            if json.loads(body)["status"]=="cancelled":
                break
            time.sleep(.02)
        self.assertEqual(json.loads(body)["status"],"cancelled")
        status,body=self.request("/api/run",payload)
        self.assertEqual(status,202,body)
        task=json.loads(body)["task_id"]
        for attempt in range(600):
            status,body=self.request("/api/task?id="+task)
            result=json.loads(body)
            if result["status"] not in {"running","queued"}:
                break
            time.sleep(.05)
        self.assertEqual(result["status"],"completed",result)
        self.assertEqual(result["progress"]["summary"]["frames"],2000)
        self.assertEqual(result["progress"]["summary"]["frame_pairs"],1999)


if __name__=="__main__":
    unittest.main()
