from __future__ import absolute_import, division, print_function

import os
import pickle
import struct
import uuid

import numpy as np
import pandas as pd

try:
    from urllib.parse import quote, unquote
except ImportError:
    from urllib import quote, unquote

from .common import ZictBase


def _get_uniq_name():
    uniq_str = str(uuid.uuid4())
    uniq_str = 'f_' + uniq_str.replace('-', '')
    return uniq_str


def _safe_key(key):
    """
    Escape key so as to be usable on all filesystems.
    """
    # Even directory separators are unsafe.
    return quote(key, safe='')


def _unsafe_key(key):
    """
    Undo the escaping done by _safe_key().
    """
    return unquote(key)


class _WriteAheadLog:
    """
    Write ahead log implementation for storing key value pairs to the file
    Appends to the file key value and action as pickled tuples and reply them back when requested
    :param log_path: file path to the log file
    """

    def __init__(self, log_path):
        self.log_path = log_path

    def write_key_value_and_action(self, key, value, action):
        """
        Record key value pair and action to the log file
        :param key: key to record
        :param value: value to record
        :param action: action assigned to key value pair
        :return: None
        """
        with open(self.log_path, 'ab') as f:
            b = pickle.dumps((key, value, action))
            f.write(struct.pack('>H', len(b)))
            f.write(b)

    def get_all_pairs(self):
        """
        Get all tuples with key value and action from the log
        :return: list of tuples with key value and action
        """
        if not os.path.exists(self.log_path):
            return []
        result = []
        with open(self.log_path, 'rb') as f:
            while True:
                bytes_len_b = f.read(2)
                if not bytes_len_b:
                    break
                bytes_len = struct.unpack('>H', bytes_len_b)[0]
                payload = f.read(bytes_len)
                key_val_pair = pickle.loads(payload)
                result.append(key_val_pair)

        return result


_WAL_NAME = 'f_a3d4e639575448efa18ed45bdbf5882a.bin'


class SubstetutedValue:
    def __init__(self, val, base_path):
        self.has_substetuted_val = False
        self.base_path = base_path
        self.path = os.path.join(base_path, _get_uniq_name())
        if isinstance(val, pd.DataFrame):
            self.val_type = 'dataframe'
        elif isinstance(val, pd.Series):
            self.val_type = 'series'
        elif isinstance(val, np.ndarray):
            self.val_type = 'nparray'
            self.path += '.npy'
        else:
            self.val_type = 'generic'

        if self.val_type == 'dataframe':
            val.to_parquet(self.path, engine='pyarrow', compression='gzip')
        elif self.val_type == 'series':
            df = pd.DataFrame(val)
            df.to_parquet(self.path, engine='pyarrow', compression='gzip')
        elif self.val_type == 'nparray':
            np.save(self.path, val)
        else:
            val = self._substetute_vals(val)
            with open(self.path, 'wb') as f:
                pickle.dump(val, f)

    def _substetute_vals(self, vals):
        if not (isinstance(vals, list) or isinstance(vals, type)):
            return vals
        is_tuple = isinstance(vals, tuple)
        vals = list(vals)
        for i in range(len(vals)):
            if isinstance(vals[i], pd.DataFrame) or isinstance(vals[i], pd.Series) or isinstance(vals[i], np.ndarray):
                vals[i] = SubstetutedValue(vals[i], self.base_path)
                self.has_substetuted_val = True

        if is_tuple:
            vals = tuple(vals)

        return vals

    def _substetute_vals_back(self, vals):
        if not (isinstance(vals, list) or isinstance(vals, type)):
            return vals
        is_tuple = isinstance(vals, tuple)
        vals = list(vals)
        for i in range(len(vals)):
            if isinstance(vals[i], SubstetutedValue):
                vals[i] = vals[i].get_value()

        if is_tuple:
            vals = tuple(vals)

        return vals

    def get_value(self, substetute_vals=True):
        if self.val_type == 'dataframe':
            df = pd.read_parquet(self.path, engine='pyarrow')
            return df
        if self.val_type == 'series':
            df = pd.read_parquet(self.path, engine='pyarrow')
            cols = list(df.columns)
            return df[cols[0]]
        if self.val_type == 'nparray':
            val = np.load(self.path)
            return val
        result = None
        with open(self.path, 'rb') as f:
            result = pickle.load(f)
        if substetute_vals:
            result = self._substetute_vals_back(result)
        return result

    def get_file_path(self):
        return self.path

    def del_file(self):
        if self.has_substetuted_val:
            val = self.get_value(substetute_vals=False)
            for sub_val in val:
                if isinstance(sub_val, SubstetutedValue):
                    sub_val.del_file()

        os.remove(self.path)


#After that create a package and create a merge request
class File(ZictBase):
    """ Mutable Mapping interface to a directory

    Keys must be strings, values must be bytes

    Note this shouldn't be used for interprocess persistence, as keys
    are cached in memory.k

    Parameters
    ----------
    directory: string
    mode: string, ('r', 'w', 'a'), defaults to 'a'

    Examples
    --------
    >>> z = File('myfile')  # doctest: +SKIP
    >>> z['x'] = b'123'  # doctest: +SKIP
    >>> z['x']  # doctest: +SKIP
    b'123'

    Also supports writing lists of bytes objects

    >>> z['y'] = [b'123', b'4567']  # doctest: +SKIP
    >>> z['y']  # doctest: +SKIP
    b'1234567'

    Or anything that can be used with file.write, like a memoryview

    >>> z['data'] = np.ones(5).data  # doctest: +SKIP
    """
    def __init__(self, directory, mode='a'):
        self._directory = directory
        self._mode = mode
        self._keys = {}
        if not os.path.exists(self._directory):
            os.mkdir(self._directory)

        self._wal_path = os.path.join(directory, _WAL_NAME)
        self._wal = _WriteAheadLog(self._wal_path)
        for k, v, a in self._wal.get_all_pairs():
            if a == 'a':
                self._keys[k] = v
            else:
                del self._keys[k]

    def __str__(self):
        return '<File: %s, mode="%s", %d elements>' % (self._directory, self._mode, len(self))

    __repr__ = __str__

    def __getitem__(self, key):
        return self._keys[key].get_value()

    def __setitem__(self, key, value):
        substetuted_val = SubstetutedValue(value, self._directory)
        self._wal.write_key_value_and_action(key, substetuted_val, 'a')
        self._keys[key] = substetuted_val

    def __contains__(self, key):
        return key in self._keys

    def get_file_path(self, key):
        if key not in self._keys:
            raise KeyError(key)
        return self._keys[key].get_file_path()

    def keys(self):
        return iter(self._keys)

    __iter__ = keys

    def __delitem__(self, key):
        if key not in self._keys:
            raise KeyError(key)
        substetuted_val = self._keys[key]
        self._wal.write_key_value_and_action(key, substetuted_val, 'd')
        substetuted_val.del_file()
        del self._keys[key]

    def __len__(self):
        return len(self._keys)
