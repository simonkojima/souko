import numpy as np


def pick_ch(data, ch, ch_names, axis=0):
    for idx, c in enumerate(ch_names):
        if ch == c:
            return np.take(data, idx, axis=axis)
    raise ValueError()


def crop(data, tmin, tmax, times, include_tmax=True):
    diff = times - tmin
    Nmin = np.argmin(np.absolute(diff))

    diff = times - tmax
    Nmax = np.argmin(np.absolute(diff))

    if include_tmax:
        return data[Nmin : (Nmax + 1)], times[Nmin : (Nmax + 1)]
    else:
        return data[Nmin:Nmax], times[Nmin:Nmax]


def crop_multichannels(data, tmin, tmax, times, include_tmax=True):
    new_data = []
    for c in range(data.shape[0]):
        d, t = crop(
            data[c, :], tmin=tmin, tmax=tmax, times=times, include_tmax=include_tmax
        )
        new_data.append(d)
    new_data = np.stack(new_data, axis=0)

    return new_data, t


def drop_bad(data, min=-200, max=200, verbose=False):
    new_data = []
    for n in range(data.shape[0]):
        d = data[n, :]
        if (np.max(d) <= max) and (np.min(d) >= min):
            new_data.append(d)
    if verbose:
        if np.max(np.absolute(data)) >= max:
            print(np.max(np.absolute(data)))
    return np.stack(new_data, axis=0)
