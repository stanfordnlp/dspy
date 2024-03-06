from Queue import PriorityQueue, Queue, Full

import numpy as np

from dspy.generator import Generator
from dspy.lib import rechannel, t2f


class Player(Generator):
    def __init__(self, sequence=[], channels=2, live=True, loop=False, clip=True, max_size=0):
        Generator.__init__(self)
        self.max_size = max_size
        self._generators = PriorityQueue(max_size)
        self._finished = Queue()
        self._gain = 0.1
        self._live = live
        self._length_cache = max([f+g.length() for (f, g) in sequence] + [0])
        self.loop = loop
        self.clip = clip
        self.num_channels = channels

        self.num_gens = 0

        if sequence:
            for (f, g) in sequence:
                self._append_gen(f, g)

        if live:
            assert(not loop)


        self.auto_reset = loop

    def _append_gen(self, frame, gen):
        try:
            self._generators.put((t2f(frame), gen), False)
        except Full:
            print('Too many generators to append another')
            return

    def add(self, gen, time=None):
        if time is None:
            frame = self.frame
        else:
            frame = t2f(time)
        self._length_cache = max(self._length_cache, gen.length() + frame)

        if self.num_gens < self.max_size or self.max_size == 0:
            self.num_gens += 1
            self._append_gen(frame, gen)

    def _reset(self):
        if self._live:
            raise Exception('Cannot reset if Player is live')

        sequence = []
        while not self._finished.empty():
            frame, gen = self._finished.get()
            gen.reset()
            sequence.append((frame, gen))

        while not self._generators.empty():
            frame, gen = self._generators.get()
            gen.reset()
            sequence.append((frame, gen))

        for frame, gen in sequence:
            self._append_gen(frame, gen)

    def _length(self):
        if self._live:
            return float('inf')

        return self._length_cache

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, value):
        self._gain = np.clip(value, 0, 1)

    def _generate(self, frame_count):
        output = np.zeros(frame_count * self.num_channels, dtype=np.float32)
        not_done = []
        while not self._generators.empty():
            frame, gen = self._generators.get()
            if frame > self.frame + frame_count:
                not_done.append((frame, gen))
                break

            delay = 0
            if frame > self.frame:
                delay = frame - self.frame

            signal, continue_flag = gen.generate(frame_count - delay)
            signal = rechannel(signal, gen.num_channels, self.num_channels)
            output[delay * self.num_channels:] += signal
            if continue_flag:
                not_done.append((frame, gen))
            else:
                if not self._live:
                    self._finished.put((frame, gen))
                self.num_gens -= 1

        for frame, gen in not_done:
            self._append_gen(frame, gen)

        output *= self.gain
        if self.clip:
            output = np.clip(output, -1, 1)
        return output
