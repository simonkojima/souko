import numpy as np


def pick_ch(data, ch, ch_names, axis=0):
    for idx, c in enumerate(ch_names):
        if ch == c:
            return np.take(data, idx, axis=axis)
    raise ValueError()


def crop(data, tmin, tmax, times, axis=0, include_tmax=True, return_times=True):
    diff = times - tmin
    Nmin = np.argmin(np.absolute(diff))

    diff = times - tmax
    Nmax = np.argmin(np.absolute(diff))

    if include_tmax:
        Nmax += 1

    times= times[Nmin:Nmax]

    if axis==0:
        data= data[Nmin:Nmax]
    elif axis==1:
        data=data[:, Nmin:Nmax]
    elif axis==2:
        data=data[:, :, Nmin:Nmax]
    else:
        raise NotImplementedError()

    if return_times:
        return data, times
    else:
        return data



def crop_multichannels(data, tmin, tmax, times, axis=1, include_tmax=True):
    """
    new_data = []
    for c in range(data.shape[0]):
        d, t = crop(
            data[c, :], tmin=tmin, tmax=tmax, times=times, include_tmax=include_tmax
        )
        new_data.append(d)
    new_data = np.stack(new_data, axis=0)
    """
    d, t = crop(
        data, tmin=tmin, tmax=tmax, times=times, include_tmax=include_tmax, axis=axis
    )

    return d, t


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
