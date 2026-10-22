"""Bounded whole-interval overview. Stage blocks are processes, not reactions."""

from collections import Counter


class ProcessSummary:
    def __init__(self, start, end, seed, stages=8):
        self.seed = seed
        self.boundaries = sorted({start + (end-start)*index//stages for index in range(stages+1)})
        self.snapshots = {}
        self.counts = [Counter() for boundary in self.boundaries[1:]]
        self.first_offsets = [None for boundary in self.boundaries[1:]]
        self.index = 0

    def observe(self, frame, fragments):
        if frame in self.boundaries:
            self.snapshots[frame] = {atom: item['species'] for item in fragments.values()
                                     for atom in item['atoms'] & self.seed}

    def event(self, frame, event_type, offset):
        while self.index+1 < len(self.counts) and frame >= self.boundaries[self.index+1]:
            self.index += 1
        self.counts[self.index][event_type] += 1
        if self.first_offsets[self.index] is None:
            self.first_offsets[self.index] = offset

    def result(self):
        states, events, links, transfers = [], [], [], []
        identifiers = {}
        for boundary_index, frame in enumerate(self.boundaries):
            for species, count in sorted(Counter(self.snapshots[frame].values()).items()):
                identifier = f'overview-s{len(states)}'
                identifiers[(frame, species)] = identifier
                states.append(dict(id=identifier, kind='state', species=species, tracked=count,
                                   first_frame=frame, last_frame=frame, context=False,
                                   aggregate=True, boundary_index=boundary_index))
        if len(states)>1600:
            raise ValueError('摘要状态超过 1600，请减少阶段数或缩小追踪对象')
        for index, (start, end) in enumerate(zip(self.boundaries, self.boundaries[1:])):
            identifier = f'overview-p{index}'
            counts = self.counts[index]
            events.append(dict(id=identifier, kind='event', process=True, frame=start, child_frame=end,
                               event_id=index+1, event_type='process', signature='多事件过程块（不是单步反应）',
                               resolution='summary', reactants={}, products={}, event_count=sum(counts.values()),
                               event_types=dict(counts), evidence_offset=self.first_offsets[index]))
            for frame, role in ((start, 'input'), (end, 'branch')):
                counts_by_species = Counter(self.snapshots[frame].values())
                maximum = max(counts_by_species.values(), default=0)
                for species, count in sorted(counts_by_species.items()):
                    state_id = identifiers[(frame, species)]
                    links.append(dict(source=state_id if frame==start else identifier,
                                      target=identifier if frame==start else state_id, tracked=count,
                                      role=('dominant_input' if frame==start else 'dominant') if count==maximum else role,
                                      process=True))
            before, after = self.snapshots[start], self.snapshots[end]
            amounts = Counter((before[atom], after[atom]) for atom in before.keys() & after.keys())
            transfers.extend(dict(process_id=identifier, source=source, target=target, atoms=count)
                             for (source, target), count in sorted(amounts.items()))
        return dict(states=states, events=events, links=links, transfers=transfers,
                    boundaries=self.boundaries, event_count=sum(sum(counts.values()) for counts in self.counts),
                    scope='whole interval stage overview; boundary categories are cohorts, not individual intermediates',
                    warning='过程块不能作为单步反应或任意输入输出组合的路径证据；实际原子对应见 transfers，逐事件路径需展开。')
