import sys
from pathlib import Path
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.embed_index import load_vector_store
from src.generate import RAGChain

def test():
    print("Loading vector store...")
    vs = load_vector_store()
    if vs is None:
        print("Run python run.py --ingest first.")
        return
    chain = RAGChain(vs)
    
    print("First ask (main thread)...")
    try:
        res = chain.ask("Hello")
        print("First ask OK.")
    except Exception as e:
        print(f"First ask failed: {e}")
        return

    def second_ask():
        print("Second ask (new thread)...")
        try:
            res2 = chain.ask("Hello again")
            print("Second ask OK.")
        except Exception as e:
            print(f"Second ask failed: {e}")

    t = threading.Thread(target=second_ask)
    t.start()
    t.join()

if __name__ == "__main__":
    test()
