import numpy as np
import torch
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from threading import Lock

from src.models.mark.dqn.utils.StateScaler import AdaptiveStateScaler
from src.config.config import (
    NUM_EPISODES,
    EVALUATE_INTERVAL,
    TRAIN_INTERVAL,
    BATCH_SIZE
)

class AsyncScalingManager:
    def __init__(self, scaler, max_workers=2, batch_size=32):
        self.scaler = scaler
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.batch_size = batch_size
        
        # Thread-safe queues for managing work
        self.pending_states = deque()
        self.pending_futures = deque()
        self.lock = Lock()
        
    def add_state(self, state_dict):
        """Add a state to be scaled asynchronously"""
        with self.lock:
            self.pending_states.append(state_dict)
    
    def scale_batch_worker(self, state_batch):
        """Worker function that runs in background thread"""
        scaled_batch = []
        for state_dict in state_batch:
            scaled_state = self.scaler.scale_state_vector(state_dict)
            scaled_batch.append(scaled_state)
        return np.array(scaled_batch, dtype=np.float32)
    
    def submit_batch_if_ready(self):
        """Submit a batch for processing if we have enough states"""
        with self.lock:
            if len(self.pending_states) >= self.batch_size:
                # Extract batch
                batch = []
                for _ in range(self.batch_size):
                    batch.append(self.pending_states.popleft())
                
                # Submit to thread pool
                future = self.executor.submit(self.scale_batch_worker, batch)
                self.pending_futures.append(future)
                return True
        return False
    
    def get_completed_batch(self, timeout=0.001):
        """Get a completed batch if available (non-blocking by default)"""
        with self.lock:
            if not self.pending_futures:
                return None
            
            # Check if the oldest future is done
            future = self.pending_futures[0]
            
        try:
            result = future.result(timeout=timeout)
            with self.lock:
                self.pending_futures.popleft()  # Remove completed future
            return result
        except:
            return None  # Not ready yet or timeout
    
    def force_process_remaining(self):
        """Process any remaining states (blocking)"""
        with self.lock:
            if self.pending_states:
                remaining = list(self.pending_states)
                self.pending_states.clear()
                
                future = self.executor.submit(self.scale_batch_worker, remaining)
                return future.result()  # Block until done
        return None
    
    def cleanup(self):
        """Clean up the thread pool"""
        self.executor.shutdown(wait=True)


# class AsyncScalingManager:
#     def __init__(self, scaler, max_workers=2):
#         self.scaler = scaler
#         self.executor = ThreadPoolExecutor(max_workers=max_workers)
#         # Note: self.batch_size in this manager refers to the batch size for internal async processing,
#         # not necessarily the training batch size of the agent.
#         # It should be large enough to amortize thread overhead, but small enough not to delay responses.
#         # A good default could be 1, or let the caller decide when to get results.
        
#         self.pending_states_for_scaling = deque() # Stores raw state_dicts
#         self.pending_futures = deque() # Stores futures for scaled (state, next_state) pairs
#         self.lock = Lock()
        
#     def add_states_for_replay(self, state_dict, next_state_dict):
#         """Add a (state, next_state) pair to be scaled asynchronously for replay buffer."""
#         with self.lock:
#             self.pending_states_for_scaling.append((state_dict, next_state_dict))
            
#     def _scale_worker(self, data_to_scale):
#         """Worker function for scaling (state, next_state) pairs."""
#         scaled_pairs = []
#         for state_dict, next_state_dict in data_to_scale:
#             scaled_state = self.scaler.scale_state_vector(state_dict)
#             scaled_next_state = self.scaler.scale_state_vector(next_state_dict)
#             scaled_pairs.append((scaled_state, scaled_next_state))
#         # Return as a tuple of arrays (scaled_states_batch, scaled_next_states_batch)
#         # to ensure consistent shapes for batching
#         return np.array([p[0] for p in scaled_pairs], dtype=np.float32), \
#                np.array([p[1] for p in scaled_pairs], dtype=np.float32)
    
#     def submit_scaling_work(self, num_to_submit):
#         """Submit a chunk of states for background processing."""
#         with self.lock:
#             if len(self.pending_states_for_scaling) >= num_to_submit:
#                 batch = []
#                 for _ in range(num_to_submit):
#                     batch.append(self.pending_states_for_scaling.popleft())
                
#                 future = self.executor.submit(self._scale_worker, batch)
#                 self.pending_futures.append(future)
#                 return True
#         return False
    
#     def get_completed_scaled_pairs(self, timeout=0): # Set timeout to 0 for non-blocking check
#         """Get a completed batch of (scaled_state, scaled_next_state) pairs if available."""
#         with self.lock:
#             if not self.pending_futures:
#                 return None
            
#             future = self.pending_futures[0] # Check the oldest one
            
#             if future.done(): # Check if the future is actually done
#                 self.pending_futures.popleft()
#                 try:
#                     return future.result() # This will get the (scaled_states_batch, scaled_next_states_batch)
#                 except Exception as e:
#                     print(f"Error in async scaling worker: {e}")
#                     return None
#             return None # Not ready yet

#     def cleanup(self):
#         """Clean up the thread pool"""
#         # Ensure all pending futures are processed or cancelled before shutdown
#         with self.lock:
#             for future in self.pending_futures:
#                 future.cancel() # Cancel any tasks still running
#             self.pending_futures.clear()
#         self.executor.shutdown(wait=True)


# Example usage in a training loop
def example_training_loop():
    """Example showing how to use async scaling in a training loop"""
    
    # Setup
    scaler = AdaptiveStateScaler(window_size=1000)
    async_manager = AsyncScalingManager(scaler, max_workers=2, batch_size=16)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Simulate some states
    def generate_dummy_state():
        return {
            'position': np.random.randn(),
            'velocity': np.random.randn(),
            'reward': np.random.randn(),
            'health': np.random.uniform(0, 100)
        }
    
    print("Starting async scaling example...")
    
    try:
        # Simulate environment steps
        for step in range(100):
            # 1. Generate new state (from your environment)
            state = generate_dummy_state()
            
            # 2. Add to async scaling queue
            async_manager.add_state(state)
            
            # 3. Try to submit batch for background processing
            async_manager.submit_batch_if_ready()
            
            # 4. Check for completed scaled batches (non-blocking)
            scaled_batch = async_manager.get_completed_batch()
            
            if scaled_batch is not None:
                # 5. Move to GPU and use for training
                gpu_batch = torch.tensor(scaled_batch, device=device)
                print(f"Step {step}: Got scaled batch of shape {gpu_batch.shape}")
                
                # Your training code here...
                # model.train_step(gpu_batch)
            
            # Simulate some processing time
            time.sleep(0.01)
        
        # Process any remaining states
        remaining = async_manager.force_process_remaining()
        if remaining is not None:
            gpu_batch = torch.tensor(remaining, device=device)
            print(f"Final batch shape: {gpu_batch.shape}")
    
    finally:
        async_manager.cleanup()


# Simpler example for basic async scaling
def simple_async_example():
    """Simpler example with just basic async scaling"""
    
    scaler = AdaptiveStateScaler()
    
    # Generate some dummy states
    states = []
    for i in range(50):
        state = {
            'x': np.random.randn(),
            'y': np.random.randn(),
            'z': np.random.randn()
        }
        states.append(state)
    
    print("Synchronous scaling:")
    start_time = time.time()
    
    # Synchronous version
    scaled_sync = []
    for state in states:
        scaled_state = scaler.scale_state_vector(state)
        scaled_sync.append(scaled_state)
    
    sync_time = time.time() - start_time
    print(f"Sync time: {sync_time:.4f}s")
    
    print("\nAsynchronous scaling:")
    start_time = time.time()
    
    # Asynchronous version
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all scaling jobs
        futures = []
        for state in states:
            future = executor.submit(scaler.scale_state_vector, state)
            futures.append(future)
        
        # Collect results as they complete
        scaled_async = []
        for future in as_completed(futures):
            scaled_state = future.result()
            scaled_async.append(scaled_state)
    
    async_time = time.time() - start_time
    print(f"Async time: {async_time:.4f}s")
    
    # Convert to GPU tensors
    scaled_batch = np.array(scaled_async, dtype=np.float32)
    gpu_tensor = torch.tensor(scaled_batch, device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Final GPU tensor shape: {gpu_tensor.shape}")


if __name__ == "__main__":
    print("=== Simple Async Example ===")
    simple_async_example()
    
    print("\n=== Training Loop Example ===")
    example_training_loop()