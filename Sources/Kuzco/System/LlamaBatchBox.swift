//
//  LlamaBatchBox.swift
//  Kuzco
//
//  Memory-safe wrapper for llama_batch to prevent double-free issues
//

import Foundation
import llama

/// Thread-safe wrapper for llama_batch that ensures single ownership and proper cleanup
final class LlamaBatchBox {
    let pointer: UnsafeMutablePointer<llama_batch>
    private let lock = NSLock()
    
    init(maxTokens: Int32, embeddingSize: Int32 = 0, numSequences: Int32 = 1) {
        pointer = .allocate(capacity: 1)
        pointer.initialize(to: llama_batch_init(maxTokens, embeddingSize, numSequences))
    }
    
    deinit {
        // Free the internal buffers allocated by llama_batch_init
        llama_batch_free(pointer.pointee)
        // Clean up our pointer
        pointer.deinitialize(count: 1)
        pointer.deallocate()
    }
    
    /// Thread-safe access to the batch pointer
    func withBatch<T>(_ body: (UnsafeMutablePointer<llama_batch>) throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try body(pointer)
    }
    
    /// Clear the batch (reset token count)
    func clear() {
        lock.lock()
        defer { lock.unlock() }
        pointer.pointee.n_tokens = 0
    }
}