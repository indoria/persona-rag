# patch/sqlite3.py

import os
import sys
import sqlite3

MIN_SQLITE_VERSION_FOR_CHROMADB = (3, 35, 0)

PYSQLITE3_PATCH_ENABLED = False

try:
    # sqlite3.sqlite_version_info gives a tuple (major, minor, micro)
    system_sqlite_version_info = sqlite3.sqlite_version_info
    print(f"System SQLite version detected: {'.'.join(map(str, system_sqlite_version_info))}")

    if system_sqlite_version_info < MIN_SQLITE_VERSION_FOR_CHROMADB:
        print(f"System SQLite version ({'.'.join(map(str, system_sqlite_version_info))}) is less than required ({'.'.join(map(str, MIN_SQLITE_VERSION_FOR_CHROMADB))}). Attempting to enable pysqlite3 patch.")
        PYSQLITE3_PATCH_ENABLED = True
    else:
        print(f"System SQLite version ({'.'.join(map(str, system_sqlite_version_info))}) is sufficient. No pysqlite3 patch needed based on version check.")

except Exception as e:
    print(f"Could not determine system sqlite3 version or an error occurred: {e}. Falling back to default behavior (no auto-patch).")


if PYSQLITE3_PATCH_ENABLED or os.environ.get('FORCE_PYSQLITE3_PATCH') == '1':
    try:
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules['pysqlite3']
        print("Successfully swapped sqlite3 module with pysqlite3 for ChromaDB compatibility.")
    except ImportError:
        print("pysqlite3 not found. Falling back to default sqlite3. ChromaDB might still throw errors.")
else:
    print("PYSQLITE3_PATCH_ENABLED is false and FORCE_PYSQLITE3_PATCH not set. Using default sqlite3 module.")


import sqlite3
