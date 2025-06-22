import random
import os
import pickle
from typing import Any, Dict

class Cache:
    _store: Dict[int, Any]
    _cache_dir: str

    def __init__(self):
        self._store = {}
        self._cache_dir = ".cache"
        os.makedirs(self._cache_dir, exist_ok=True)

    def save(self, data: Any, key: int|None = None, to_disk: bool = False) -> int:
        """Save data in the cache and return a unique random integer ID (also prints it)."""
        if key is None:
            while True:
                key = random.randint(1, 2**31 - 1)
                if key not in self._store:
                    break
        
        self._store[key] = data
        print(f"[Cache] Saved data with ID: {key}")
        if to_disk:
            self.dump_to_disk(key)
        return key

    def load(self, key: int, from_disk: bool = False) -> Any:
        """
        Load data from the cache by ID. Raises KeyError if not found.
        If from_disk is set to True the data will be read from the disk.
        """
        if from_disk and not self.exists(key):
            return self.load_from_disk(key)
        return self._store[key]

    def exists(self, key: int) -> bool:
        """Check if the cached file for the given key exists in memory."""
        return self._store.get(key) != None

    def clear(self) -> None:
        """Clear the entire cache."""
        self._store.clear()

    def dump_to_disk(self, key: int) -> None:
            """Write the cached data for the given key to disk as a pickle file in .cache/."""
            if key not in self._store:
                raise KeyError(f"Key {key} not found in cache")

            filepath = os.path.join(self._cache_dir, f"{key}.pkl")
            with open(filepath, "wb") as f:
                pickle.dump(self._store[key], f)
            print(f"[Cache] Dumped key {key} to {filepath}")

    def exists_on_disk(self, key: int) -> bool:
        """Check if the cached file for the given key exists on disk."""
        filepath = os.path.join(self._cache_dir, f"{key}.pkl")
        return os.path.exists(filepath)

    def load_from_disk(self, key: int) -> Any:
        """
        Load cached data for the given key from disk, insert into memory cache,
        and return it. Raises FileNotFoundError if the file does not exist.
        """
        filepath = os.path.join(self._cache_dir, f"{key}.pkl")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No cached file found for key {key} at {filepath}")

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self._store[key] = data
        print(f"[Cache] Loaded key {key} from {filepath}")
        return data

