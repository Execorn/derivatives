import asyncio
import threading
import time
import pytest
from deepvol.api.websocket import ConflatedQueue

@pytest.mark.asyncio
async def test_conflated_queue_basic():
    queue = ConflatedQueue()
    queue.put("BTC", {"price": 100})
    
    batch = await queue.get()
    assert len(batch) == 1
    assert batch["BTC"] == {"price": 100}

@pytest.mark.asyncio
async def test_conflated_queue_conflation():
    queue = ConflatedQueue()
    queue.put("BTC", {"price": 100})
    queue.put("BTC", {"price": 101})
    queue.put("ETH", {"price": 50})
    queue.put("ETH", {"price": 51})
    
    batch = await queue.get()
    assert len(batch) == 2
    assert batch["BTC"] == {"price": 101}
    assert batch["ETH"] == {"price": 51}

@pytest.mark.asyncio
async def test_conflated_queue_thread_safe():
    queue = ConflatedQueue()
    
    def producer():
        for i in range(100):
            queue.put("BTC", {"price": i})
            time.sleep(0.001)
            
    thread = threading.Thread(target=producer)
    thread.start()
    
    vols = []
    t_end = time.time() + 0.5
    while time.time() < t_end:
        try:
            batch = await asyncio.wait_for(queue.get(), timeout=0.1)
            if "BTC" in batch:
                vols.append(batch["BTC"]["price"])
        except asyncio.TimeoutError:
            break
            
    thread.join()
    assert len(vols) > 0
    assert max(vols) <= 99
