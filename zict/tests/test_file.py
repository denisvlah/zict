from __future__ import absolute_import, division, print_function
import pandas as pd
import numpy as np
import os
import shutil

import pytest

from zict.file import File, _WriteAheadLog, _WAL_NAME
from . import utils_test


@pytest.yield_fixture
def fn():
    filename = '.tmp'
    if os.path.exists(filename):
        if not os.path.isdir(filename):
            os.remove(filename)
        else:
            shutil.rmtree(filename)

    yield filename

    if os.path.exists(filename):
        if not os.path.isdir(filename):
            os.remove(filename)
        else:
            shutil.rmtree(filename)


def test_mapping(fn):
    """
    Test mapping interface for File().
    """
    z = File(fn)
    utils_test.check_mapping(z)


def test_str(fn):
    z = File(fn)
    assert fn in str(z)
    assert fn in repr(z)
    assert z._mode in str(z)
    assert z._mode in repr(z)


def test_delitem(fn):
    z = File(fn)

    z['x'] = b'123'
    path = z.get_file_path('x')
    assert os.path.exists(path)
    del z['x']
    assert not os.path.exists(path)


def test_missing_key(fn):
    z = File(fn)

    with pytest.raises(KeyError):
        z['x']


def test_arbitrary_chars(fn):
    z = File(fn)

    # Avoid hitting the Windows max filename length
    chunk = 16
    for i in range(1, 128, chunk):
        key = ''.join(['foo_'] + [chr(i) for i in range(i, min(128, i + chunk))])
        with pytest.raises(KeyError):
            z[key]
        z[key] = b'foo'
        assert z[key] == b'foo'
        assert list(z) == [key]
        assert list(z.keys()) == [key]
        assert list(z.items()) == [(key, b'foo')]
        assert list(z.values()) == [b'foo']

        zz = File(fn)
        assert zz[key] == b'foo'
        assert list(zz) == [key]
        assert list(zz.keys()) == [key]
        assert list(zz.items()) == [(key, b'foo')]
        assert list(zz.values()) == [b'foo']
        del zz

        del z[key]
        with pytest.raises(KeyError):
            z[key]


def test_item_with_very_long_name_can_be_read_and_deleted_and_restored(fn):
    z = File(fn)
    long_key1 = 'a' + 'a'.join(str(i) for i in range(500))
    long_key2 = 'b' + 'a'.join(str(i) for i in range(500))
    z[long_key1] = b'key1'
    z[long_key2] = b'key2'
    assert z[long_key1] == b'key1'
    assert z[long_key2] == b'key2'
    z2 = File(fn)
    assert z2[long_key1] == b'key1'
    assert z2[long_key2] == b'key2'
    del z2[long_key1]
    z3 = File(fn)
    assert long_key1 not in z3
    assert z3[long_key2] == b'key2'


def test_write_ahead_log_can_record_keys_and_replay_them_back(fn):
    file_path = fn
    wal = _WriteAheadLog(file_path)
    expected = [
        ('key1', 'val1', 'a'),
        ('key2', 'val2', 'a'),
        ('key3', 'val3', 'd')
    ]
    for key, val, action in expected:
        wal.write_key_value_and_action(key, val, action)

    vals = wal.get_all_pairs()
    assert expected == vals


def test_write_ahead_log_can_read_keys_from_file_writen_by_another_instance(fn):
    file_path = fn
    wal = _WriteAheadLog(file_path)
    expected = [
        ('key1', 'val1', 'a'),
        ('keyxxxx2', 'valllll2', 'a'),
        ('key3', 'val3', 'd')
    ]
    for key, val, action in expected:
        wal.write_key_value_and_action(key, val, action)

    wal2 = _WriteAheadLog(file_path)

    vals = wal2.get_all_pairs()
    assert expected == vals


def test_list_with_pandas_df_can_be_written(fn):
    z = File(fn)
    df1 = pd.DataFrame({'a': [1]})
    l = [df1, 'l']
    z['a'] = l
    val_back = z['a']
    assert isinstance(val_back, list)
    assert val_back[1] == 'l'
    assert val_back[0]['a'].iloc[0] == 1


def test_tuple_with_pandas_df_can_be_written(fn):
    z = File(fn)
    df1 = pd.DataFrame({'a': [1]})
    l = (df1, 'l')
    z['a'] = l
    val_back = z['a']
    assert isinstance(val_back, tuple)
    assert val_back[1] == 'l'
    assert val_back[0]['a'].iloc[0] == 1


def test_tuple_with_np_array_can_be_written(fn):
    z = File(fn)
    a = np.random.randint(100)
    l = (a, 'l')
    z['a'] = l
    val_back = z['a']
    assert isinstance(val_back, tuple)
    assert val_back[1] == 'l'
    assert val_back[0] == a


def test_list_with_np_array_can_be_written(fn):
    z = File(fn)
    a = np.random.randint(100)
    l = [a, 'l']
    z['a'] = l
    val_back = z['a']
    assert isinstance(val_back, list)
    assert val_back[1] == 'l'
    assert val_back[0] == a


def test_list_with_np_array_can_be_removed(fn):
    z = File(fn)
    a = np.random.randint(100)
    l = [a, 'l']
    z['a'] = l
    del z['a']
    file_names = os.listdir(z._directory)
    filtered_file_names = [f for f in file_names if f != _WAL_NAME]
    assert len(filtered_file_names) == 0


def test_pandas_obj_can_be_written(fn):
    z = File(fn)
    df = pd.DataFrame({'a': [1]})
    z['a'] = df
    assert z['a']['a'].iloc[0] == 1


def test_pandas_with_int_col_names_can_be_saved_and_retrieved(fn):
    z = File(fn)
    df1 = pd.DataFrame({1: [1, 2, 3], 'a': [1, 2, 3]})
    z['a'] = df1
    df2: pd.DataFrame = z['a']
    assert list(df1.columns) == list(df2.columns)
    assert list(df1[1]) == list(df2[1])
    assert list(df1['a']) == list(df2['a'])


def test_pandas_with_int_and_unicode_col_names_can_be_saved_and_retrieved(fn):
    z = File(fn)
    col_name = u'【明報專訊】'
    df1 = pd.DataFrame({1: [1, 2, 3], col_name: [1, 2, 3]})
    z['a'] = df1
    df2: pd.DataFrame = z['a']
    assert list(df1.columns) == list(df2.columns)
    assert list(df1[1]) == list(df2[1])
    assert list(df1[col_name]) == list(df2[col_name])


def test_pandas_series_with_int_col_names_can_be_saved_and_retrieved(fn):
    z = File(fn)
    s = pd.Series([1, 2, 3, 4], name=(1))
    z['a'] = s
    s2 = z['a']
    assert list(s) == list(s2)
    assert s.name == s2.name


def test_pandas_index_with_int_col_names_can_be_saved_and_retrieved(fn):
    z = File(fn)
    s = pd.Index([1, 2, 3, 4], name=(1))
    z['a'] = s
    s2 = z['a']
    assert list(s) == list(s2)
    assert s.name == s2.name


def test_pandas_index_with_empty_col_names_can_be_saved_and_retrieved(fn):
    z = File(fn)
    s = pd.Index([1, 2, 3, 4])
    z['a'] = s
    s2 = z['a']
    assert list(s) == list(s2)
    assert s.name == s2.name


def test_pandas_index_with_empty_str_col_names_can_be_saved_and_retrieved(fn):
    z = File(fn)
    s = pd.Index([1, 2, 3, 4], name='x')
    z['a'] = s
    s2 = z['a']
    assert list(s) == list(s2)
    assert s.name == s2.name


def test_empty_pandas_index_with_empty_str_col_names_can_be_saved_and_retrieved(fn):
    z = File(fn)
    s = pd.Index([], name='')
    z['a'] = s
    s2 = z['a']
    assert list(s) == list(s2)
    assert s.name == s2.name


def test_empty_pandas_df_with_empty_str_col_names_can_be_saved_and_retrieved(fn):
    z = File(fn)
    s = pd.DataFrame()
    z['a'] = s
    s2 = z['a']
    assert all(s == s2)
