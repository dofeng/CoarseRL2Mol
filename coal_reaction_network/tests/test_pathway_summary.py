from pathway.tracing import trace
from test_regressions import Workspace


class SummaryTests(Workspace):
    def test_internal_activity_compresses_but_split_does_not(self):
        path,connection=self.database([[(1,2),(2,3),(3,4)],[(1,2),(1,3),(1,4)],[(1,2),(2,3),(3,4)],[(1,2),(3,4)]])
        connection.close()
        result=trace(path,dict(start=0,end=3,fragment=1))
        compact=result['compact']
        self.assertEqual(compact['collapsed_events'],2)
        self.assertEqual(len(compact['events']),1)
        self.assertEqual(compact['events'][0]['event_type'],'fragmentation')
        self.assertEqual(len(compact['states'][0]['activity_events']),2)
        identifiers={node['id'] for node in compact['states']+compact['events']}
        self.assertTrue(all(link['source'] in identifiers and link['target'] in identifiers for link in compact['links']))

    def test_overview_covers_events_beyond_current_page(self):
        path,connection=self.database([[(1,2)],[],[(1,2)],[],[(1,2)]])
        connection.close()
        result=trace(path,dict(start=0,end=4,fragment=1,event_limit=1,summary_stages=2))
        self.assertEqual(len(result['events']),1)
        summary=result['overview']
        self.assertEqual(summary['event_count'],4)
        self.assertEqual(summary['boundaries'],[0,2,4])
        self.assertEqual([node['event_count'] for node in summary['events']],[2,2])
        self.assertEqual([node['evidence_offset'] for node in summary['events']],[0,2])
        identifiers={node['id'] for node in summary['states']+summary['events']}
        self.assertTrue(all(link['source'] in identifiers and link['target'] in identifiers for link in summary['links']))
        self.assertEqual(sum(item['atoms'] for item in summary['transfers']),4)

    def test_no_reaction_stage_is_not_fabricated_event(self):
        path,connection=self.database([[(1,2)],[(1,2)],[(1,2)]])
        connection.close()
        summary=trace(path,dict(start=0,end=2,fragment=1))['overview']
        self.assertEqual(summary['event_count'],0)
        self.assertTrue(all(node['evidence_offset'] is None for node in summary['events']))
        self.assertTrue(all(node['process'] for node in summary['events']))

    def test_summary_size_is_bounded_by_stages_not_frames(self):
        path,connection=self.database([[(1,2)] if index%2 else [] for index in range(100)])
        connection.close()
        result=trace(path,dict(start=0,end=99,fragment=1,summary_stages=4,event_limit=1))
        self.assertEqual(result['overview']['event_count'],99)
        self.assertEqual(len(result['overview']['events']),4)
        self.assertEqual(len(result['overview']['states']),5)
