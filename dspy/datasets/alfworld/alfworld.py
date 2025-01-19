import os
import queue
import random

def env_worker(inq, outq):
    """
    Worker process: creates a single AlfredTWEnv instance,
    handles 'init' (with task idx) and 'step' (with action).
    """

    try:
        import io
        import yaml
        import alfworld.agents.environment as environment
        from contextlib import redirect_stdout, redirect_stderr
    except ImportError:
        raise ImportError("alfworld is not installed. " \
            "Please install it via `pip install alfworld==0.3.5` then run `alfworld-download`.")

    buf = io.StringIO()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'base_config.yml')

    with open(config_path) as f:
        config = yaml.safe_load(f)

    with redirect_stdout(buf), redirect_stderr(buf):
        base_env = environment.AlfredTWEnv(config, train_eval="train")

    env = None
    while True:
        cmd, data = inq.get()
        if cmd == 'init':
            env = base_env.init_env(batch_size=1)
            env.skip(data)
            task_def, info = env.reset()
            outq.put((task_def[0], info))
        elif cmd == 'step':
            obs, rew, done, info = env.step([data])
            outq.put((obs, rew, done, info))
        elif cmd == 'close':
            outq.put('CLOSED')
            break
        else:
            outq.put('UNKNOWN_CMD')


class EnvPool:
    """
    Pool of processes, each with a unique env_worker.
    Acquire a worker using a context manager for safe usage:
        with pool.session() as sess:
            sess.init(5)              # init with idx=5
            obs, rew, done, info = sess.step("go north")
            ...
    """
    def __init__(self, size=2):
        self.size = size
        self.workers = []
        self.available = queue.Queue()

        try:
            import multiprocess as mp
        except ImportError:
            raise ImportError("multiprocess is not installed. " \
                "Please install it via `pip install multiprocess`.")

        # Must call set_start_method('spawn') here, before creating any processes
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            # If it's already set, ignore
            pass

        ctx = mp.get_context("spawn")
        for i in range(size):
            inq = ctx.Queue()
            outq = ctx.Queue()
            p = ctx.Process(target=env_worker, args=(inq, outq), daemon=True)
            p.start()
            self.workers.append((inq, outq, p))
            self.available.put(i)

    def _acquire(self):
        wid = self.available.get()
        return wid, self.workers[wid][0], self.workers[wid][1]

    def _release(self, wid):
        self.available.put(wid)

    def close_all(self):
        """Close all processes in the pool."""
        while not self.available.empty():
            wid = self.available.get()
            inq, outq, proc = self.workers[wid]
            inq.put(('close', None))
            outq.get()  # Wait 'CLOSED'
            inq.close()
            outq.close()
            proc.join()

    def session(self):
        """Context manager that acquires/releases a single worker."""
        return _EnvSession(self)


class _EnvSession:
    """
    A context manager that acquires a worker from the pool,
    provides .init(idx) and .step(action), then releases the worker.
    """
    def __init__(self, pool: EnvPool):
        self.pool = pool
        self.wid = None
        self.inq = None
        self.outq = None

    def __enter__(self):
        self.wid, self.inq, self.outq = self.pool._acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool._release(self.wid)

    def init(self, idx):
        self.inq.put(('init', idx))
        return self.outq.get()  # (task_def, info)

    def step(self, action):
        self.inq.put(('step', action))
        return self.outq.get()  # (obs, rew, done, info)


class AlfWorld:
    def __init__(self, max_threads=20):
        self.POOL = EnvPool(size=max_threads)

        import dspy
        dataset = [dspy.Example(idx=idx).with_inputs('idx') for idx in range(3500)]
        random.Random(0).shuffle(dataset)

        trainset, devset = dataset[:3000], dataset[-500:]
        assert len(trainset) + len(devset) <= len(dataset)

        self.trainset = trainset
        self.devset = devset

    def __del__(self):
        self.POOL.close_all()
