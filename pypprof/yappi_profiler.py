import threading
import time
import yappi
from collections import defaultdict
from pypprof import builder

_lock = threading.Lock()

class YappiProfiler(object):
    def __init__(self, clock_type='cpu'):
        self._clock_type = clock_type

    def profile(self, duration_ns):
        """Collects a profile for the given duration (in nanoseconds)."""
        with _lock:
            yappi.stop()
            yappi.clear_stats()
            yappi.set_clock_type(self._clock_type)
            yappi.start()
            try:
                time.sleep(duration_ns / 1e9)
            finally:
                yappi.stop()
                stats = yappi.get_func_stats()
            
            return self._convert_to_pprof(stats, duration_ns)

    def _convert_to_pprof(self, stats, duration_ns):
        profile_builder = builder.Builder()
        samples = defaultdict(lambda: [0, 0]) # trace -> [count, value]
        
        # We want to track how much of a function's self-time is covered by its callers (edges).
        # Key: (name, filename, lineno), Value: covered_self_time_ns
        covered_self_time = defaultdict(int)
        
        # Helper to create frame tuple
        def make_frame(func_stat):
            # yappi uses 'lineno' which is start line.
            # pprof frame: (name, filename, start_line, line_number)
            # We don't have exact line number of call, just start line of function.
            return (func_stat.name, func_stat.module, func_stat.lineno, func_stat.lineno)

        # First pass: Process edges (Caller -> Callee)
        for func_stat in stats:
            caller_frame = make_frame(func_stat)
            
            if not hasattr(func_stat, 'children'):
                continue
                
            for child_stat in func_stat.children:
                callee_frame = make_frame(child_stat)
                
                # child_stat.ttot is time spent in callee when called by caller.
                # child_stat.tsub is time spent in callee's children when called by caller.
                # Self time in this context:
                self_time_ns = int((child_stat.ttot - child_stat.tsub) * 1e9)
                count = child_stat.ncall
                
                if self_time_ns < 0:
                    self_time_ns = 0
                
                # Trace: [Caller, Callee]
                # Note: pprof builder expects leaf at 0?
                # pypprof/thread_profiler.py: "The leaf frame is at position 0."
                # So trace should be (Callee, Caller)
                trace = (callee_frame, caller_frame)
                
                entry = samples[trace]
                entry[0] += count
                entry[1] += self_time_ns
                
                # Record covered self time for callee
                callee_key = (child_stat.name, child_stat.module, child_stat.lineno)
                covered_self_time[callee_key] += self_time_ns

        # Second pass: Process roots (Self time not covered by edges)
        for func_stat in stats:
            frame = make_frame(func_stat)
            key = (func_stat.name, func_stat.module, func_stat.lineno)
            
            total_self_time_ns = int((func_stat.ttot - func_stat.tsub) * 1e9)
            if total_self_time_ns < 0:
                total_self_time_ns = 0
                
            covered_ns = covered_self_time.get(key, 0)
            remainder_ns = total_self_time_ns - covered_ns
            
            # Allow for small floating point errors, but if significant, add as root sample
            # Also add if count > 0 and not fully covered (even if time is 0 due to precision)
            if remainder_ns > 0 or (covered_ns == 0 and func_stat.ncall > 0):
                # Trace: [Function] (Leaf)
                trace = (frame,)
                
                # For root samples, count is tricky. 
                # If we use ncall, we might double count executions if they were partially covered by edges.
                # But for self-time, we are adding the remainder.
                # We can set count to 0 or try to estimate? 
                # Let's set count to 0 for remainder samples to avoid inflating call counts,
                # unless it's a true root (not covered at all).
                
                count = 0
                if covered_ns == 0:
                    count = func_stat.ncall
                
                entry = samples[trace]
                entry[0] += count
                entry[1] += remainder_ns

        # Populate profile
        # samples map values are [count, value], need tuple
        final_samples = {k: tuple(v) for k, v in samples.items()}
        
        profile_type = 'CPU' if self._clock_type == 'cpu' else 'WALL'
        unit = 'nanoseconds'
        # Period: yappi is instrumenting, so period is effectively 1? 
        # Or we can leave it. 
        # builder expects period. For sampling it's sampling interval.
        # For instrumentation, maybe we can set it to 1 event?
        
        profile_builder.populate_profile(final_samples, profile_type, 'count', 1, duration_ns)
        return profile_builder.emit()
