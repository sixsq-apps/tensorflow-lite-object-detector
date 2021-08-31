# coding:utf-8

import queue
import weakref
import threading


class Singleton(type):
    ''' This class is meant to be used as a metaclass to transform a class into a singleton '''

    _instances = weakref.WeakValueDictionary()
    _singleton_lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._singleton_lock:
            if cls not in cls._instances:
                instance = super(Singleton, cls).__call__(*args, **kwargs)
                cls._instances[cls] = instance
                return instance
            return cls._instances[cls]


class RenewQueue(queue.Queue, object):
    ''' A queue which can contain only one element.
        When an item is put into the queue it will replace the existing one if any. '''

    def __init__(self):
        super(RenewQueue, self).__init__(maxsize=1)
        self.put_super = super(RenewQueue, self).put
        self.get_super = super(RenewQueue, self).get
        self._lock = threading.Lock()

    def put(self, item):
        with self._lock:
            try:
                self.put_super(item, False)
            except queue.Full:
                self.get_super(False)
                self.put_super(item, False)

    def get(self, block=True, timeout=None):
        self._lock.acquire()
        acquired = True
        try:
            return self.get_super(False, timeout)
        except queue.Empty:
            self._lock.release()
            acquired = False
            if not block:
                raise
            return self.get_super(True, timeout)
        finally:
            if acquired:
                self._lock.release()


def log_threads_stackstraces():
    print_args = dict(file=sys.stderr, flush=True)
    print("\nfaulthandler.dump_traceback()", **print_args)
    faulthandler.dump_traceback()
    print("\nthreading.enumerate()", **print_args)
    for th in threading.enumerate():
        print(th, **print_args)
        traceback.print_stack(sys._current_frames()[th.ident])
    print(**print_args)
