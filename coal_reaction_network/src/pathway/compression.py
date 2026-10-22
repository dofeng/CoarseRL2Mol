"""Collapse only evidenced one-to-one within-class changes on an event page."""

from copy import deepcopy


def compress_page(states, events, links):
    states = {node['id']:deepcopy(node) for node in states}
    aliases = {identifier:identifier for identifier in states}
    inputs, outputs = {}, {}
    for link in links:
        if link['target'].startswith('e'):
            inputs.setdefault(link['target'], []).append(link['source'])
        if link['source'].startswith('e'):
            outputs.setdefault(link['source'], []).append(link['target'])
    def resolve(identifier):
        while aliases[identifier]!=identifier:
            identifier=aliases[identifier]
        return identifier
    removed = set()
    for event in events:
        parents, children = inputs.get(event['id'], []), outputs.get(event['id'], [])
        if event['event_type']!='internal_rearrangement' or len(parents)!=1 or len(children)!=1:
            continue
        parent, child = resolve(parents[0]), resolve(children[0])
        before, after = states[parent], states[child]
        if before['species']!=after['species'] or before['tracked']!=after['tracked']:
            continue
        before['last_frame']=after['last_frame']
        before.setdefault('activity_events', []).append(dict(frame=event['frame'],event_id=event['event_id']))
        aliases[child]=parent
        removed.add(event['id'])
    compact_links=[]
    for link in links:
        if link['source'] in removed or link['target'] in removed:
            continue
        item=dict(link)
        for side in ('source','target'):
            if item[side] in aliases:
                item[side]=resolve(item[side])
        compact_links.append(item)
    return dict(states=[node for identifier,node in states.items() if resolve(identifier)==identifier],
                events=[event for event in events if event['id'] not in removed],links=compact_links,
                collapsed_events=len(removed))
