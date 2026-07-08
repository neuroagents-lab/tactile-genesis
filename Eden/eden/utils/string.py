"""String and glob name-matching helpers."""

# Modified from isaaclab/utils/string.py

from collections.abc import Sequence
from typing import Any
import fnmatch
import re


_GLOB_WILDCARD_PATTERN = re.compile(r"(?<!\\)(\*|\?)")
_REGEX_INDICATOR_PATTERN = re.compile(r"(?<!\\)[\(\)\{\}\|\+\^\$]")
_DOT_STAR_PATTERN = re.compile(r"(?<!\\)\.\*")


def is_glob_pattern(pattern: str) -> bool:
    """Best-effort heuristic to determine whether the provided pattern looks like a glob."""
    if not isinstance(pattern, str):
        return False
    has_glob_wildcards = bool(_GLOB_WILDCARD_PATTERN.search(pattern))
    if not has_glob_wildcards:
        return False
    if _REGEX_INDICATOR_PATTERN.search(pattern):
        return False
    if _DOT_STAR_PATTERN.search(pattern):
        return False
    return True


def resolve_matching_names(
    keys: str | Sequence[str],
    list_of_strings: Sequence[str],
    preserve_order: bool = False,
    use_glob: bool | None = None,
) -> tuple[list[int], list[str]]:
    """Match a list of query patterns against a list of strings and return the matched indices and names.

    When a list of query regular expressions or glob patterns is provided, the function checks each target string
    against each query and returns the indices of the matched strings and the matched strings.

    If the :attr:`preserve_order` is True, the ordering of the matched indices and names is the same as the order
    of the provided list of strings. This means that the ordering is dictated by the order of the target strings
    and not the order of the query patterns.

    If the :attr:`preserve_order` is False, the ordering of the matched indices and names is the same as the order
    of the provided list of query patterns.

    For example, consider the list of strings is ['a', 'b', 'c', 'd', 'e'] and the patterns are ['a|c', 'b'].
    If :attr:`preserve_order` is False, then the function will return the indices of the matched strings and the
    strings as: ([0, 1, 2], ['a', 'b', 'c']). When :attr:`preserve_order` is True, it will return them as:
    ([0, 2, 1], ['a', 'c', 'b']).

    Note:
        The function does not sort the indices. It returns the indices in the order they are found.

    Parameters
    ----------
    keys: str | Sequence[str]
        A regular expression, glob pattern, or a list of either to match the strings in the list.
    list_of_strings: Sequence[str]
        A list of strings to match.
    preserve_order: bool
        Whether to preserve the order of the query keys in the returned values. Defaults to False.
    use_glob: bool | None
        Optional override for the matching strategy. When True, treat all keys as Unix shell-style glob
        patterns; when False, treat them as regular expressions. When None, automatically infer the mode per key
        using :func:`is_glob_pattern`. Defaults to None.

    Returns
    -------
    tuple[list[int], list[str]]
        A tuple of lists containing the matched indices and names.

    Raises
    ------
    ValueError
        When multiple matches are found for a string in the list.
    ValueError
        When not all query patterns are matched.
    """
    # resolve name keys
    if isinstance(keys, str):
        keys = [keys]
    # find matching patterns
    index_list = []
    names_list = []
    key_idx_list = []
    # book-keeping to check that we always have a one-to-one mapping
    # i.e. each target string should match only one pattern
    target_strings_match_found = [None for _ in range(len(list_of_strings))]
    keys_match_found = [[] for _ in range(len(keys))]
    # loop over all target strings
    for target_index, potential_match_string in enumerate(list_of_strings):
        for key_index, re_key in enumerate(keys):
            if use_glob is None:
                match_with_glob = is_glob_pattern(re_key)
            else:
                match_with_glob = use_glob
            matched = (
                fnmatch.fnmatchcase(potential_match_string, re_key)
                if match_with_glob
                else re.fullmatch(re_key, potential_match_string)
            )
            if matched:
                # check if match already found
                if target_strings_match_found[target_index]:
                    raise ValueError(
                        f"Multiple matches for '{potential_match_string}':"
                        f" '{target_strings_match_found[target_index]}' and '{re_key}'!"
                    )
                # add to list
                target_strings_match_found[target_index] = re_key
                index_list.append(target_index)
                names_list.append(potential_match_string)
                key_idx_list.append(key_index)
                # add for pattern key
                keys_match_found[key_index].append(potential_match_string)
    # reorder keys if they should be returned in order of the query keys
    if preserve_order:
        reordered_index_list = [None] * len(index_list)
        global_index = 0
        for key_index in range(len(keys)):
            for key_idx_position, key_idx_entry in enumerate(key_idx_list):
                if key_idx_entry == key_index:
                    reordered_index_list[key_idx_position] = global_index
                    global_index += 1
        # reorder index and names list
        index_list_reorder = [None] * len(index_list)
        names_list_reorder = [None] * len(index_list)
        for idx, reorder_idx in enumerate(reordered_index_list):
            index_list_reorder[reorder_idx] = index_list[idx]
            names_list_reorder[reorder_idx] = names_list[idx]
        # update
        index_list = index_list_reorder
        names_list = names_list_reorder
    # check that all query patterns are matched
    if not all(keys_match_found):
        # make this print nicely aligned for debugging
        msg = "\n"
        for key, value in zip(keys, keys_match_found):
            msg += f"\t{key}: {value}\n"
        msg += f"Available strings: {list_of_strings}\n"
        # raise error
        raise ValueError(
            f"Not all query patterns are matched! Please check that the provided patterns are correct: {msg}"
        )
    # return
    return index_list, names_list


def resolve_matching_names_values(
    data: dict[str, Any],
    list_of_strings: Sequence[str],
    preserve_order: bool = False,
    use_glob: bool | None = None,
) -> tuple[list[int], list[str], list[Any]]:
    """Match query patterns against a list of strings, returning matched indices, names, and values.

    If the :attr:`preserve_order` is True, the ordering of the matched indices and names is the same as the order
    of the provided list of strings. This means that the ordering is dictated by the order of the target strings
    and not the order of the query patterns.

    If the :attr:`preserve_order` is False, the ordering of the matched indices and names is the same as the order
    of the provided list of query patterns.

    For example, consider the dictionary is {"a|d|e": 1, "b|c": 2}, the list of strings is ['a', 'b', 'c', 'd', 'e'].
    If :attr:`preserve_order` is False, then the function will return the indices of the matched strings, the
    matched strings, and the values as: ([0, 1, 2, 3, 4], ['a', 'b', 'c', 'd', 'e'], [1, 2, 2, 1, 1]). When
    :attr:`preserve_order` is True, it will return them as: ([0, 3, 4, 1, 2], ['a', 'd', 'e', 'b', 'c'], [1, 1, 1, 2, 2]).

    Parameters
    ----------
    data: dict[str, Any]
        A dictionary mapping query patterns (regular expressions or globs) to values to match against the list.
    list_of_strings: Sequence[str]
        A list of strings to match.
    preserve_order: bool
        Whether to preserve the order of the query keys in the returned values. Defaults to False.
    use_glob:
        Optional override for the matching strategy. When True, treat all dictionary keys as glob patterns;
        when False, treat them as regular expressions. When None, automatically infer the mode per key using
        :func:`is_glob_pattern`. Defaults to None.

    Returns
    -------
    tuple[list[int], list[str], list[Any]]
        A tuple of lists containing the matched indices, names, and values.

    Raises
    ------
    TypeError
        When the input argument :attr:`data` is not a dictionary.
    ValueError
        When multiple matches are found for a string in the dictionary.
    ValueError
        When not all query patterns in the data keys are matched.
    """
    # check valid input
    if not isinstance(data, dict):
        raise TypeError(f"Input argument `data` should be a dictionary. Received: {data}")
    # find matching patterns
    index_list = []
    names_list = []
    values_list = []
    key_idx_list = []
    # book-keeping to check that we always have a one-to-one mapping
    # i.e. each target string should match only one pattern
    target_strings_match_found = [None for _ in range(len(list_of_strings))]
    keys_match_found = [[] for _ in range(len(data))]
    # loop over all target strings
    for target_index, potential_match_string in enumerate(list_of_strings):
        for key_index, (re_key, value) in enumerate(data.items()):
            if use_glob is None:
                match_with_glob = is_glob_pattern(re_key)
            else:
                match_with_glob = use_glob
            matched = (
                fnmatch.fnmatchcase(potential_match_string, re_key)
                if match_with_glob
                else re.fullmatch(re_key, potential_match_string)
            )
            if matched:
                # check if match already found
                if target_strings_match_found[target_index]:
                    raise ValueError(
                        f"Multiple matches for '{potential_match_string}':"
                        f" '{target_strings_match_found[target_index]}' and '{re_key}'!"
                    )
                # add to list
                target_strings_match_found[target_index] = re_key
                index_list.append(target_index)
                names_list.append(potential_match_string)
                values_list.append(value)
                key_idx_list.append(key_index)
                # add for pattern key
                keys_match_found[key_index].append(potential_match_string)
    # reorder keys if they should be returned in order of the query keys
    if preserve_order:
        reordered_index_list = [None] * len(index_list)
        global_index = 0
        for key_index in range(len(data)):
            for key_idx_position, key_idx_entry in enumerate(key_idx_list):
                if key_idx_entry == key_index:
                    reordered_index_list[key_idx_position] = global_index
                    global_index += 1
        # reorder index and names list
        index_list_reorder = [None] * len(index_list)
        names_list_reorder = [None] * len(index_list)
        values_list_reorder = [None] * len(index_list)
        for idx, reorder_idx in enumerate(reordered_index_list):
            index_list_reorder[reorder_idx] = index_list[idx]
            names_list_reorder[reorder_idx] = names_list[idx]
            values_list_reorder[reorder_idx] = values_list[idx]
        # update
        index_list = index_list_reorder
        names_list = names_list_reorder
        values_list = values_list_reorder
    # check that all query patterns are matched
    if not all(keys_match_found):
        # make this print nicely aligned for debugging
        msg = "\n"
        for key, value in zip(data.keys(), keys_match_found):
            msg += f"\t{key}: {value}\n"
        msg += f"Available strings: {list_of_strings}\n"
        # raise error
        raise ValueError(
            f"Not all query patterns are matched! Please check that the provided patterns are correct: {msg}"
        )
    # return
    return index_list, names_list, values_list
