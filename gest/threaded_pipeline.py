import contextlib
import threading


class ClosableIterable:

    def __init__(self):
        self.condition = threading.Condition()
        self.closed = False

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()

    def __iter__(self):
        raise NotImplementedError()


class Factory(ClosableIterable):

    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def __iter__(self):
        while True:
            with self.condition:
                if self.closed:
                    return
            yield self.callback()


class Updatable(ClosableIterable):

    def __init__(self):
        super().__init__()
        self.version = 0
        self.value = None

    def update(self, value):
        with self.condition:
            self.version += 1
            self.value = value
            self.condition.notify_all()

    def __iter__(self):
        previous = 0
        while True:
            with self.condition:
                while not self.closed and self.version == previous:
                    self.condition.wait()
                if self.closed:
                    return
                value = self.value
                previous = self.version
            yield value


class PipelineRun:

    class Thread(threading.Thread):

        def __init__(self, component, input, output):
            super().__init__(daemon=True)
            self.component = component
            self.input = input
            self.output = output
            self.exception = None

        def run(self):
            try:
                for value in self.component(self.input):
                    self.output.update(value)
            except Exception as e:
                self.exception = e
                raise
            finally:
                self.output.close()

    def __init__(self, source: ClosableIterable, components):
        self.values = [source]
        self.threads = []

        for component in components:
            self.values.append(Updatable())
            self.threads.append(PipelineRun.Thread(component, *self.values[-2:]))

    def start(self):
        for thread in self.threads:
            thread.start()

    def stop(self):
        self.values[0].close()
        for thread in self.threads:
            thread.join()

    def __iter__(self):
        return iter(self.values[-1])

    @contextlib.contextmanager
    def __call__(self):
        self.start()
        yield self
        self.stop()
