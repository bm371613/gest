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
            
            
class Pipeline:
    
    def __init__(self, components, default_input_factory=None):
        self.components = components
        self.default_input_factory = default_input_factory

    def sequential(self, input=None):
        return SequentialPipelineRun(input or self.default_input_factory(), self.components)

    def threaded(self, input=None):
        return ThreadedPipelineRun(input or self.default_input_factory(), self.components)


class PipelineRun:

    def __init__(self, input: ClosableIterable):
        self.input = input
        self.output = self.input

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.input.close()

    def __iter__(self):
        return iter(self.output)


class SequentialPipelineRun(PipelineRun):

    def __init__(self, input: ClosableIterable, components):
        super().__init__(input)
        for component in components:
            self.output = component(self.output)


class ThreadedPipelineRun(PipelineRun):

    class Thread(threading.Thread):

        def __init__(self, component, input):
            super().__init__(daemon=True, name=component.__name__)
            self.component = component
            self.input = input
            self.output = Updatable()

        def run(self):
            try:
                for value in self.component(self.input):
                    self.output.update(value)
            finally:
                self.output.close()

    def __init__(self, input: ClosableIterable, components):
        super().__init__(input)
        self.threads = []
        for component in components:
            thread = ThreadedPipelineRun.Thread(component, input=self.output)
            self.output = thread.output
            self.threads.append(thread)

    def __enter__(self):
        for thread in self.threads:
            thread.start()
        return super().__enter__()

    def __exit__(self, *exc_info):
        super().__exit__(*exc_info)
        for thread in self.threads:
            thread.join()
