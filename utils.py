import numpy as np

def compute_time_to_event(surv_funcs, threshold=0.8):
    time_to_event = []
    for surv_func in surv_funcs:
        # Find the time where survival probability <= threshold
        time_idx = np.where(surv_func.y <= threshold)[0]
        if len(time_idx) > 0:
            time_to_event.append(surv_func.x[time_idx[0]])  # First time below threshold
        else:
            time_to_event.append(np.inf)  # Event hasn't occurred
    return time_to_event