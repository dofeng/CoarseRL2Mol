from pathway.tracing import trace
from test_regressions import Workspace


class PathwayTests(Workspace):
    def run_trace(self, frames, query=None):
        path, connection = self.database(frames)
        connection.close()
        return trace(path, dict(start=0, end=len(frames)-1, fragment=1, **(query or {})))

    def test_split_fusion_chains_share_actual_states(self):
        result = self.run_trace([[(1,2),(2,3),(3,4)],[(1,2),(3,4)],[(1,2),(2,3),(3,4)]])
        self.assertEqual(len(result['events']), 2)
        outgoing = {link['target'] for link in result['links'] if link['source']==result['events'][0]['id']}
        incoming = {link['source'] for link in result['links'] if link['target']==result['events'][1]['id']}
        self.assertEqual(outgoing, incoming)
        self.assertEqual(result['seed_atoms'], 4)

    def test_same_class_unrelated_fragments_do_not_join(self):
        result = self.run_trace([[(1,2),(3,4)],[(1,2)],[(1,2),(3,4)]])
        self.assertEqual(result['events'], [])
        self.assertEqual(result['states'][0]['last_frame'], 2)

    def test_foreign_co_reactant_does_not_expand_seed(self):
        result = self.run_trace([[(1,2),(3,4)],[(1,2),(2,3),(3,4)],[(1,2),(3,4)]])
        self.assertEqual(result['seed_atoms'], 2)
        self.assertTrue(any(node['context'] and node['tracked']==0 for node in result['states']))
        self.assertEqual(len(result['events']), 2)
        self.assertLessEqual(max(node['tracked'] for node in result['states']), 2)

    def test_backward_anchor_uses_terminal_atom_set(self):
        result = self.run_trace([[(1,2),(2,3),(3,4)],[(1,2),(3,4)]], {'direction':'backward'})
        self.assertEqual(result['anchor'], 1)
        self.assertEqual(result['seed_atoms'], 2)
        self.assertEqual(len(result['events']), 1)

    def test_persistence_is_compressed_between_events(self):
        result = self.run_trace([[(1,2),(2,3),(3,4)],[(1,2),(3,4)],[(1,2),(3,4)],[(1,2),(2,3),(3,4)]])
        self.assertEqual(len(result['events']), 2)
        self.assertTrue(any(node['first_frame']==1 and node['last_frame']==2 for node in result['states']))

    def test_event_pagination_is_explicit_and_integral(self):
        result = self.run_trace([[(1,2),(2,3),(3,4)],[(1,2),(3,4)],[(1,2),(2,3),(3,4)]], {'event_limit':1})
        self.assertEqual(result['next_offset'], 1)
        self.assertEqual(result['total_events'], 2)
        self.assertEqual(len(result['events']), 1)
        identifiers={node['id'] for node in result['states']+result['events']}
        self.assertTrue(all(link['source'] in identifiers and link['target'] in identifiers for link in result['links']))
        second=trace(self.root/'network.sqlite',dict(start=0,end=2,fragment=1,event_limit=1,offset=1))
        self.assertEqual(second['events'][0]['event_type'],'fusion')

    def test_missing_pair_is_not_bridged(self):
        path, connection=self.database([[],[],[]])
        with connection:
            connection.execute("UPDATE frame_pairs SET status='failed' WHERE parent_frame_id=1")
        connection.close()
        with self.assertRaisesRegex(ValueError,'缺帧'):
            trace(path,dict(start=0,end=2,fragment=1))

    def test_cancellation(self):
        path, connection=self.database([[],[]]);connection.close()
        with self.assertRaises(InterruptedError):
            trace(path,dict(start=0,end=1,fragment=1),lambda:True)

    def test_target_without_atom_evidence_is_rejected(self):
        path, connection=self.database([[],[]]);connection.close()
        with self.assertRaisesRegex(ValueError,'共享原子'):
            trace(path,dict(start=0,end=1,fragment=1,target_species='nonexistent'))

    def test_exact_event_lookup_avoids_signature_page_limit(self):
        from dashboard_queries import records
        path,connection=self.database([[(1,2),(3,4)],[]]);connection.close()
        result=records(path,'events',dict(start=0,end=1,event_id=2,limit=1))
        self.assertEqual(len(result['rows']),1)
        self.assertEqual(result['rows'][0]['event_id'],2)
