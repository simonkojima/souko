def fmt(v):
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def get_proc_name(resample, tmin, tmax, l_freq, h_freq):
    return f"resample-{fmt(resample)}_tmin-{fmt(tmin)}_tmax-{fmt(tmax)}_l_freq-{fmt(l_freq)}_h_freq-{fmt(h_freq)}"
