#!/usr/bin/env python3
"""
Utility script to diagnose and fix sparse_embedding issues in Milvus collections.

This script helps with collections that have corrupted or invalid sparse_embedding data
by either dropping the problematic index or recreating the collection.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pymilvus import Collection, utility, connections
from app.core.config import get_settings


def check_sparse_embedding_issues(collection_name: str) -> bool:
    """Check if a collection has sparse_embedding field and potential issues."""
    try:
        if not utility.has_collection(collection_name):
            print(f"Collection '{collection_name}' does not exist")
            return False
        
        collection = Collection(collection_name)
        collection.load()
        
        # Check if sparse_embedding field exists
        has_sparse = any(field.name == "sparse_embedding" for field in collection.schema.fields)
        if not has_sparse:
            print(f"Collection '{collection_name}' does not have sparse_embedding field")
            return False
        
        print(f"✓ Collection '{collection_name}' has sparse_embedding field")
        
        # Try to query a small sample
        try:
            results = collection.query(
                expr='id != ""',
                output_fields=["id"],
                limit=1,
                offset=0
            )
            print(f"✓ Basic query works")
        except Exception as e:
            print(f"✗ Basic query failed: {e}")
            return True  # Has issues
        
        # Try to query with sparse_embedding field
        try:
            results = collection.query(
                expr='id != ""',
                output_fields=["id", "sparse_embedding"],
                limit=1,
                offset=0
            )
            print(f"✓ Query with sparse_embedding works")
            return False  # No issues
        except Exception as e:
            print(f"✗ Query with sparse_embedding failed: {e}")
            return True  # Has issues
            
    except Exception as e:
        print(f"Error checking collection: {e}")
        return False


def drop_sparse_embedding_index(collection_name: str) -> bool:
    """Drop the sparse_embedding index if it exists."""
    try:
        if not utility.has_collection(collection_name):
            print(f"Collection '{collection_name}' does not exist")
            return False
        
        collection = Collection(collection_name)
        
        # Check if sparse_embedding index exists
        try:
            collection.drop_index("sparse_embedding")
            print(f"✓ Successfully dropped sparse_embedding index from '{collection_name}'")
            return True
        except Exception as e:
            print(f"Note: Could not drop sparse_embedding index: {e}")
            return False
            
    except Exception as e:
        print(f"Error dropping index: {e}")
        return False


def main():
    settings = get_settings()
    
    # Connect to Milvus
    try:
        connections.connect(alias="default", uri=settings.milvus_uri)
        print(f"✓ Connected to Milvus at {settings.milvus_uri}")
    except Exception as e:
        print(f"✗ Failed to connect to Milvus: {e}")
        return 1
    
    # Get list of collections with potential issues
    if len(sys.argv) > 1:
        collections_to_check = sys.argv[1:]
    else:
        # Check all collections
        collections_to_check = utility.list_collections()
    
    if not collections_to_check:
        print("No collections found")
        return 0
    
    print(f"\nChecking {len(collections_to_check)} collection(s)...\n")
    
    problematic = []
    for col_name in collections_to_check:
        print(f"\nChecking '{col_name}':")
        if check_sparse_embedding_issues(col_name):
            problematic.append(col_name)
    
    if not problematic:
        print("\n✓ No problematic collections found")
        return 0
    
    print(f"\n⚠ Found {len(problematic)} collection(s) with sparse_embedding issues:")
    for col in problematic:
        print(f"  - {col}")
    
    # Ask for remediation
    if len(sys.argv) > 1 and "--fix" in sys.argv:
        print("\nAttempting to fix issues...")
        for col_name in problematic:
            print(f"\nFixing '{col_name}':")
            if drop_sparse_embedding_index(col_name):
                print(f"✓ Fixed '{col_name}'")
            else:
                print(f"✗ Could not fix '{col_name}'")
        
        print("\nNOTE: The sparse_embedding field still exists in the schema.")
        print("To fully remove it, you may need to drop and recreate the collection.")
    else:
        print("\nTo fix these issues, run:")
        print(f"  python scripts/fix_sparse_embedding.py {' '.join(problematic)} --fix")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
