//
//  LlamaKitBridge.swift
//  Kuzco
//
//  Created by Jared Cassoutt on 5/29/25.
//

import Foundation
import llama

typealias CLlamaModel = OpaquePointer
typealias CLlamaContext = OpaquePointer
typealias CLlamaToken = llama_token
typealias CLlamaBatch = llama_batch


enum LlamaKitBridge {
    static func initializeLlamaBackend() {
        // Initialize the backend - this is critical for proper C++ initialization
        llama_backend_init()
        print("🦙 Kuzco: llama_backend_init called 🦙")
    }

    static func releaseLlamaBackend() {
        llama_backend_free()
    }

    /// Validates a model file before attempting to load it
    static func validateModelFile(path: String) throws {
        // Check if file exists
        guard FileManager.default.fileExists(atPath: path) else {
            throw KuzcoError.modelFileNotAccessible(path: path)
        }
        
        // Check file size (must be at least 1MB for a valid model)
        do {
            let attributes = try FileManager.default.attributesOfItem(atPath: path)
            if let fileSize = attributes[.size] as? UInt64, fileSize < 1_048_576 {
                throw KuzcoError.modelInitializationFailed(details: "Model file is too small (\(fileSize) bytes). Minimum expected size is 1MB.")
            }
        } catch {
            throw KuzcoError.modelFileNotAccessible(path: path)
        }
        
        // Validate GGUF magic bytes
        guard let fileHandle = FileHandle(forReadingAtPath: path) else {
            throw KuzcoError.modelFileNotAccessible(path: path)
        }
        
        defer { fileHandle.closeFile() }
        
        do {
            let magicBytes = fileHandle.readData(ofLength: 4)
            let expectedMagic = "GGUF".data(using: .ascii)!
            
            if magicBytes.count < 4 || magicBytes != expectedMagic {
                throw KuzcoError.configurationInvalid(reason: "Model file does not appear to be a valid GGUF format. Expected magic bytes 'GGUF', but found different format.")
            }
        } catch {
            throw KuzcoError.modelInitializationFailed(details: "Failed to read model file header: \(error.localizedDescription)")
        }
    }

    static func loadModel(from path: String, settings: InstanceSettings) throws -> CLlamaModel {
        // Pre-validate the model file
        try validateModelFile(path: path)
        
        // Wrap the actual loading in error handling
        do {
            var mparams = llama_model_default_params()
            mparams.n_gpu_layers = settings.offloadedGpuLayers
            mparams.use_mmap = settings.enableMemoryMapping
            mparams.use_mlock = settings.enableMemoryLocking

            print("🦙 Kuzco - Attempting to load model from: \(path) 🦙")
            print("🦙 GPU layers: \(settings.offloadedGpuLayers), mmap: \(settings.enableMemoryMapping), mlock: \(settings.enableMemoryLocking) 🦙")
            
            // Log llama.cpp library version info for debugging integration issues
            print("🦙 llama.cpp library loaded - checking version... 🦙")
            
            guard let modelPtr = llama_model_load_from_file(path, mparams) else {
                // Enhanced error handling with architecture detection
                let fileName = (path as NSString).lastPathComponent.lowercased()
                
                if fileName.contains("qwen3") || fileName.contains("qwen2") {
                    let architecture = fileName.contains("qwen3") ? "qwen3" : "qwen2"
                    throw KuzcoError.unsupportedModelArchitecture(
                        architecture: architecture,
                        suggestedAction: "Your version of llama.cpp doesn't support \(architecture) architecture. The model will still work but may use fallback formatting. Consider updating llama.cpp or using a compatible model format. Kuzco will attempt to handle this model with ChatML formatting."
                    )
                } else if fileName.contains("deepseek") {
                    throw KuzcoError.unsupportedModelArchitecture(
                        architecture: "deepseek",
                        suggestedAction: "DeepSeek models require specific llama.cpp support. Try using a different model or updating to a more recent version of llama.cpp."
                    )
                } else if fileName.contains("claude") || fileName.contains("gpt") {
                    throw KuzcoError.configurationInvalid(reason: "This appears to be a commercial model format that cannot be loaded with llama.cpp. Please use a GGUF-format open source model.")
                } else {
                    throw KuzcoError.modelInitializationFailed(details: "llama_load_model_from_file returned null for path \(path). This could be due to: 1) Unsupported model architecture, 2) Corrupted model file, 3) Insufficient memory, 4) Invalid GGUF format. Try using a different model or check the file integrity.")
                }
            }
            
            print("🦙 Kuzco - Model loaded successfully 🦙")
            
            // Print full diagnostics including llama.cpp version
            let diagnostics = getModelDiagnostics(model: modelPtr)
            print(diagnostics)
            
            // Extra validation for Gemma models
            if let arch = getModelArchitecture(model: modelPtr), 
               arch.lowercased().contains("gemma") {
                print("🦙 Kuzco: Gemma model detected, performing extra validation... 🦙")
                
                // Get the vocab object
                guard let vocab = llama_model_get_vocab(modelPtr) else {
                    print("🦙 Kuzco ERROR: Failed to get vocab object for Gemma model! 🦙")
                    throw KuzcoError.modelInitializationFailed(details: "Gemma model loaded but vocab object is null")
                }
                
                // Check vocab type
                let vocabType = llama_vocab_type(vocab)
                print("🦙 Kuzco: Vocab type raw value: \(vocabType.rawValue) 🦙")
                
                // Get token count
                let tokenCount = llama_vocab_n_tokens(vocab)
                print("🦙 Kuzco: Token count: \(tokenCount) 🦙")
                
                // Try to get a specific token to verify vocab is working
                let bosToken = llama_vocab_bos(vocab)
                let eosToken = llama_vocab_eos(vocab)
                print("🦙 Kuzco: BOS token: \(bosToken), EOS token: \(eosToken) 🦙")
                
                // Test tokenization directly here to catch the issue early
                print("🦙 Kuzco: Testing direct tokenization... 🦙")
                
                // WORKAROUND: Skip tokenization test for Gemma models
                // The tokenizer crashes when called with nil buffer (two-pass pattern)
                // But might work with a pre-allocated buffer
                print("🦙 Kuzco WARNING: Skipping tokenization test for Gemma 🦙")
                print("🦙 Kuzco: Known issue - tokenizer crashes with nil buffer 🦙")
                print("🦙 Kuzco: Will attempt single-pass tokenization with pre-allocated buffer 🦙")
                
                // Try single-pass with generous buffer instead
                let testText = "test"
                var tokens = [llama_token](repeating: 0, count: 100)
                print("🦙 Kuzco: Attempting single-pass tokenization with buffer... 🦙")
                
                let result = tokens.withUnsafeMutableBufferPointer { buffer in
                    testText.withCString { cstr in
                        llama_tokenize(modelPtr, cstr, Int32(strlen(cstr)), buffer.baseAddress, Int32(buffer.count), false, false)
                    }
                }
                
                if result > 0 {
                    print("🦙 Kuzco SUCCESS: Single-pass tokenization worked! Got \(result) tokens 🦙")
                    print("🦙 Kuzco: Gemma model CAN tokenize with pre-allocated buffer 🦙")
                } else if result < 0 {
                    print("🦙 Kuzco INFO: Tokenization returned \(result) - buffer too small, need \(-result) tokens 🦙")
                    // This is actually OK - it means tokenization works but needs bigger buffer
                } else {
                    print("🦙 Kuzco ERROR: Tokenization returned 0 - complete failure 🦙")
                    throw KuzcoError.modelInitializationFailed(details: "Gemma tokenizer completely non-functional")
                }
            }
            
            return modelPtr
            
        } catch let error as KuzcoError {
            // Re-throw our custom errors
            throw error
        } catch {
            // Catch any unexpected errors from llama.cpp
            throw KuzcoError.modelInitializationFailed(details: "Unexpected error during model loading: \(error.localizedDescription)")
        }
    }

    /// Attempts to load a model with fallback approaches for unsupported architectures
    static func loadModelWithFallback(from path: String, settings: InstanceSettings, fallbackArchitecture: ModelArchitecture? = nil) throws -> CLlamaModel {
        do {
            return try loadModel(from: path, settings: settings)
        } catch let error as KuzcoError {
            print("🦙 Kuzco - Primary model load failed: \(error.localizedDescription) 🦙")
            
            // Try with reduced GPU layers as fallback
            if case .unsupportedModelArchitecture = error, settings.offloadedGpuLayers > 0 {
                print("🦙 Kuzco - Attempting fallback with CPU-only processing 🦙")
                var fallbackSettings = settings
                fallbackSettings.offloadedGpuLayers = 0
                
                do {
                    return try loadModel(from: path, settings: fallbackSettings)
                } catch {
                    print("🦙 Kuzco - Fallback also failed 🦙")
                    throw error
                }
            }
            
            // For unsupported architecture errors, we'll let the higher level handle fallback
            throw error
        }
    }

    static func freeModel(_ model: CLlamaModel) {
        do {
            llama_model_free(model)
            print("🦙 Kuzco - Model freed successfully 🦙")
        } catch {
            print("🦙 Kuzco - Warning: Error freeing model: \(error.localizedDescription) 🦙")
        }
    }

    static func createContext(for model: CLlamaModel, settings: InstanceSettings) throws -> CLlamaContext {
        do {
            var cparams = llama_context_default_params()
            cparams.n_ctx = settings.contextLength
            cparams.n_batch = settings.processingBatchSize
            cparams.n_ubatch = settings.processingBatchSize
            cparams.flash_attn = settings.useFlashAttention
            cparams.n_threads = settings.cpuThreadCount
            cparams.n_threads_batch = settings.cpuThreadCount

            print("🦙 Kuzco - Creating context with: ctx=\(settings.contextLength), batch=\(settings.processingBatchSize), threads=\(settings.cpuThreadCount) 🦙")

            guard let contextPtr = llama_init_from_model(model, cparams) else {
                throw KuzcoError.contextCreationFailed(details: "llama_init_from_model returned null. This may be due to insufficient memory or invalid context parameters.")
            }
            
            print("🦙 Kuzco - Context created successfully 🦙")
            return contextPtr
            
        } catch let error as KuzcoError {
            throw error
        } catch {
            throw KuzcoError.contextCreationFailed(details: "Unexpected error during context creation: \(error.localizedDescription)")
        }
    }

    static func freeContext(_ context: CLlamaContext) {
        do {
            llama_free(context)
            print("🦙 Kuzco - Context freed successfully 🦙")
        } catch {
            print("🦙 Kuzco - Warning: Error freeing context: \(error.localizedDescription) 🦙")
        }
    }

    static func getModelMaxContextLength(context: CLlamaContext) -> UInt32 {
        return llama_n_ctx(context)
    }

    static func getModelVocabularySize(model: CLlamaModel) -> Int32 {
        // Get vocab from model first, then get token count
        guard let vocab = llama_model_get_vocab(model) else { return 0 }
        return llama_vocab_n_tokens(vocab)
    }

    static func tokenize(text: String, model: CLlamaModel, addBos: Bool, parseSpecial: Bool) throws -> [CLlamaToken] {
        // This should be checked before calling tokenize, but double-check here
        guard hasUsableTokenizer(model: model) else {
            print("🦙 KuzcoBridge Error: Attempted to tokenize with model lacking tokenizer 🦙")
            throw KuzcoError.tokenizationFailed(details: "Model lacks usable tokenizer. Please use a GGUF with embedded tokenizer.")
        }
        
        // Log tokenization attempt for debugging
        print("🦙 Tokenizing text of length \(text.count), addBos=\(addBos), parseSpecial=\(parseSpecial) 🦙")
        
        // WORKAROUND: Check if this is a Gemma model
        let isGemma = getModelArchitecture(model: model)?.lowercased().contains("gemma") ?? false
        
        if isGemma {
            // WORKAROUND: For Gemma, use single-pass with generous buffer
            // The two-pass pattern crashes when passing nil buffer
            print("🦙 Using single-pass tokenization for Gemma model 🦙")
            
            // Estimate tokens generously (roughly 1.5 tokens per character for safety)
            let estimatedTokens = max(100, text.count * 2)
            var tokens = Array<CLlamaToken>(repeating: 0, count: estimatedTokens)
            
            let actualCount = tokens.withUnsafeMutableBufferPointer { buffer in
                text.withCString { cstr in
                    llama_tokenize(model, cstr, Int32(strlen(cstr)), buffer.baseAddress, Int32(buffer.count), addBos, parseSpecial)
                }
            }
            
            if actualCount > 0 {
                // Success - return only the used tokens
                return Array(tokens.prefix(Int(actualCount)))
            } else if actualCount < 0 {
                // Buffer too small - try again with exact size
                let neededSize = -actualCount
                print("🦙 Buffer too small, retrying with size \(neededSize) 🦙")
                
                var biggerTokens = Array<CLlamaToken>(repeating: 0, count: Int(neededSize))
                let retryCount = biggerTokens.withUnsafeMutableBufferPointer { buffer in
                    text.withCString { cstr in
                        llama_tokenize(model, cstr, Int32(strlen(cstr)), buffer.baseAddress, Int32(buffer.count), addBos, parseSpecial)
                    }
                }
                
                if retryCount > 0 {
                    return Array(biggerTokens.prefix(Int(retryCount)))
                } else {
                    throw KuzcoError.tokenizationFailed(details: "Gemma tokenization failed even with correctly sized buffer")
                }
            } else {
                throw KuzcoError.tokenizationFailed(details: "Gemma tokenization returned 0 tokens")
            }
        }
        
        // Standard two-pass tokenization for non-Gemma models
        // First pass: Get the required token count
        let requiredCount = text.withCString { cstr in
            llama_tokenize(model, cstr, Int32(strlen(cstr)), nil, 0, addBos, parseSpecial)
        }
        
        // Handle negative return (indicates required size)
        let tokenCount = requiredCount < 0 ? -requiredCount : requiredCount
        
        guard tokenCount > 0 else {
            print("🦙 KuzcoBridge: Tokenization resulted in 0 tokens for text: '\(text)' 🦙")
            return []
        }
        
        // Second pass: Allocate exact buffer and tokenize
        var tokens = Array<CLlamaToken>(repeating: 0, count: Int(tokenCount))
        let actualCount = tokens.withUnsafeMutableBufferPointer { buffer in
            text.withCString { cstr in
                llama_tokenize(model, cstr, Int32(strlen(cstr)), buffer.baseAddress, Int32(buffer.count), addBos, parseSpecial)
            }
        }
        
        if actualCount <= 0 {
            throw KuzcoError.tokenizationFailed(details: "Tokenization failed: returned \(actualCount) tokens")
        }
        
        // Defensive: trim if actual count differs from required
        if actualCount != tokenCount {
            print("🦙 KuzcoBridge Warning: Token count mismatch. Required: \(tokenCount), Actual: \(actualCount) 🦙")
            return Array(tokens.prefix(Int(actualCount)))
        }
        
        return tokens
    }

    static func detokenize(token: CLlamaToken, model: CLlamaModel) -> String {
        let bufferSize = 128
        var buffer = [CChar](repeating: 0, count: bufferSize)

        let nChars = llama_token_to_piece(model, token, &buffer, Int32(bufferSize), 0, false)

        if nChars <= 0 {
            if nChars < 0 && -Int(nChars) > bufferSize {
                print("🦙 KuzcoBridge Error: Buffer too small for detokenizing token \(token). Required: \(-Int(nChars)), available: \(bufferSize) 🦙")
                return "<\(token_id_error_buffer_small)>"
            }

            return ""
        }

        let pieceBytes = buffer.prefix(Int(nChars)).map { UInt8(bitPattern: $0) }
        return String(decoding: pieceBytes, as: UTF8.self)
    }

    private static let token_id_error_buffer_small = -3

    private static let token_id_error = -1
    private static let token_unknown = -2


    // MARK: Batch Processing & KV Cache
    static func createBatch(maxTokens: UInt32, embeddingSize: Int32 = 0, numSequences: Int32 = 1) throws -> CLlamaBatch {
        let batch = llama_batch_init(Int32(maxTokens), embeddingSize, numSequences)
        return batch
    }

    static func freeBatch(_ batch: CLlamaBatch) {
        llama_batch_free(batch)
    }

    static func clearBatch(_ batch: UnsafeMutablePointer<CLlamaBatch>) {
        batch.pointee.n_tokens = 0
    }

    static func addTokenToBatch(batch: UnsafeMutablePointer<CLlamaBatch>, token: CLlamaToken, position: Int32, sequenceId: Int32, enableLogits: Bool) {
        let currentTokenIndex = batch.pointee.n_tokens

        batch.pointee.token[Int(currentTokenIndex)] = token
        batch.pointee.pos[Int(currentTokenIndex)] = llama_pos(position)
        batch.pointee.n_seq_id[Int(currentTokenIndex)] = 1

        batch.pointee.seq_id[Int(currentTokenIndex)]!.pointee = llama_seq_id(sequenceId)

        batch.pointee.logits[Int(currentTokenIndex)] = enableLogits ? 1 : 0
        batch.pointee.n_tokens += 1
    }

    static func setThreads(for context: CLlamaContext, mainThreads: Int32, batchThreads: Int32) {
        llama_set_n_threads(context, mainThreads, batchThreads)
    }

    static func processBatch(context: CLlamaContext, batch: UnsafeMutablePointer<CLlamaBatch>) throws {
        do {
            let result = llama_decode(context, batch.pointee)
            if result != 0 {
                let errorMsg = "llama_decode returned \(result). This may indicate insufficient memory, invalid batch, or model corruption."
                print("🦙 KuzcoBridge Batch Processing Error: \(errorMsg) 🦙")
                throw KuzcoError.predictionFailed(details: errorMsg)
            }
        } catch let error as KuzcoError {
            throw error
        } catch {
            throw KuzcoError.predictionFailed(details: "Unexpected error during batch processing: \(error.localizedDescription)")
        }
    }

    static func getLogitsOutput(context: CLlamaContext, fromBatchTokenIndex index: Int32) -> UnsafeMutablePointer<Float>? {
        do {
            return llama_get_logits_ith(context, index)
        } catch {
            print("🦙 KuzcoBridge Warning: Error getting logits: \(error.localizedDescription) 🦙")
            return nil
        }
    }
    
    static func clearKeyValueCache(context: CLlamaContext) {
        do {
            let memory = llama_get_memory(context)
            llama_memory_clear(memory, false)
            print("🦙 Kuzco - KV cache cleared successfully 🦙")
        } catch {
            print("🦙 Kuzco - Warning: Error clearing KV cache: \(error.localizedDescription) 🦙")
        }
    }

    static func removeTokensFromKeyValueCache(context: CLlamaContext, sequenceId: Int32, fromPosition start: Int32, toPosition end: Int32) {
        do {
            let memory = llama_get_memory(context)
            _ = llama_memory_seq_rm(memory, llama_seq_id(sequenceId), llama_pos(start), llama_pos(end))
        } catch {
            print("🦙 Kuzco - Warning: Error removing tokens from KV cache: \(error.localizedDescription) 🦙")
        }
    }

    static func sampleTokenGreedy(model: CLlamaModel, context: CLlamaContext, logits: UnsafeMutablePointer<Float>) -> CLlamaToken {
        do {
            let vocab = llama_model_get_vocab(model)
            let vocabSize = llama_vocab_n_tokens(vocab)
            
            guard vocabSize > 0 else {
                print("🦙 KuzcoBridge Error: Invalid vocabulary size: \(vocabSize) 🦙")
                return 0
            }
            
            var maxLogit: Float = -Float.infinity
            var bestToken: CLlamaToken = 0
            
            for i in 0..<Int(vocabSize) {
                if logits[i] > maxLogit {
                    maxLogit = logits[i]
                    bestToken = CLlamaToken(i)
                }
            }
            return bestToken
        } catch {
            print("🦙 KuzcoBridge Error: Exception during token sampling: \(error.localizedDescription) 🦙")
            return 0
        }
    }

    static func getBosToken(model: CLlamaModel) -> CLlamaToken {
        do {
            let vocab = llama_model_get_vocab(model)
            return llama_vocab_bos(vocab)
        } catch {
            print("🦙 KuzcoBridge Warning: Error getting BOS token: \(error.localizedDescription) 🦙")
            return 1 // Common fallback BOS token ID
        }
    }

    static func getEosToken(model: CLlamaModel) -> CLlamaToken {
        do {
            let vocab = llama_model_get_vocab(model)
            return llama_vocab_eos(vocab)
        } catch {
            print("🦙 KuzcoBridge Warning: Error getting EOS token: \(error.localizedDescription) 🦙")
            return 2 // Common fallback EOS token ID
        }
    }

    static func isEndOfGenerationToken(model: CLlamaModel, token: CLlamaToken) -> Bool {
        do {
            let vocab = llama_model_get_vocab(model)
            return token == llama_vocab_eos(vocab) || llama_vocab_is_eog(vocab, token)
        } catch {
            print("🦙 KuzcoBridge Warning: Error checking EOG token: \(error.localizedDescription) 🦙")
            return token == 2 // Fallback check for common EOS token
        }
    }
    
    // MARK: Model Configuration Queries
    
    /// Check if model has a usable tokenizer (not just vocab, but actual tokenizer implementation)
    static func hasUsableTokenizer(model: CLlamaModel) -> Bool {
        guard let vocab = llama_model_get_vocab(model) else { 
            print("🦙 KuzcoBridge: No vocab found in model 🦙")
            return false 
        }
        
        let tokenCount = llama_vocab_n_tokens(vocab)
        if tokenCount <= 0 { 
            print("🦙 KuzcoBridge: Vocab has no tokens (count: \(tokenCount)) 🦙")
            return false 
        }
        
        // Check vocab type - NONE means no tokenizer
        let vocabType = llama_vocab_type(vocab)
        if vocabType == llama_vocab_type(rawValue: 0) { // LLAMA_VOCAB_TYPE_NONE
            print("🦙 KuzcoBridge: Vocab type is NONE - no tokenizer implementation 🦙")
            return false
        }
        
        // Log the tokenizer type for debugging
        let typeStr: String
        switch vocabType.rawValue {
        case 1: typeStr = "SPM (SentencePiece)"  // LLAMA_VOCAB_TYPE_SPM
        case 2: typeStr = "BPE (Byte-Pair Encoding)"  // LLAMA_VOCAB_TYPE_BPE
        case 3: typeStr = "WPM (WordPiece)"  // LLAMA_VOCAB_TYPE_WPM
        case 4: typeStr = "UGM (Unigram)"  // LLAMA_VOCAB_TYPE_UGM
        case 5: typeStr = "RWKV"  // LLAMA_VOCAB_TYPE_RWKV
        case 6: typeStr = "PLaMo-2"  // LLAMA_VOCAB_TYPE_PLAMO2
        default: typeStr = "Unknown(\(vocabType.rawValue))"
        }
        print("🦙 KuzcoBridge: Tokenizer type: \(typeStr), token count: \(tokenCount) 🦙")
        
        return true
    }
    
    static func shouldAddBOSToken(model: CLlamaModel) -> Bool {
        // Get the vocab from the model and check its BOS preference
        let vocab = llama_model_get_vocab(model)
        guard vocab != nil else { return false }
        return llama_vocab_get_add_bos(vocab)
    }
    
    static func shouldAddEOSToken(model: CLlamaModel) -> Bool {
        // Get the vocab from the model and check its EOS preference
        let vocab = llama_model_get_vocab(model)
        guard vocab != nil else { return false }
        return llama_vocab_get_add_eos(vocab)
    }
    
    static func modelHasTokenizer(model: CLlamaModel) -> Bool {
        // Check if model has valid vocabulary using the model-specific function
        let vocabSize = getModelVocabularySize(model: model)
        if vocabSize <= 0 { return false }
        
        // Try to read a tokenizer metadata key
        var buf = [CChar](repeating: 0, count: 64)
        let got = buf.withUnsafeMutableBufferPointer { ptr in
            llama_model_meta_val_str(model, "tokenizer.ggml.type", ptr.baseAddress, ptr.count)
        }
        
        // If we have vocab size but no tokenizer metadata, it might still work
        // Return true if we found metadata OR if vocab is reasonably sized
        return got > 0 || vocabSize > 100
    }
    
    static func getModelArchitecture(model: CLlamaModel) -> String? {
        var buf = [CChar](repeating: 0, count: 64)
        let got = buf.withUnsafeMutableBufferPointer { ptr in
            llama_model_meta_val_str(model, "general.architecture", ptr.baseAddress, ptr.count)
        }
        return (got > 0) ? String(cString: buf) : nil
    }
    
    static func supportsSpecialTokens(model: CLlamaModel) -> Bool {
        // Check if model architecture is Gemma (which doesn't use special tokens the same way)
        if let arch = getModelArchitecture(model: model), arch.lowercased().contains("gemma") {
            return false
        }
        
        // Check if model has special tokens defined
        var buf = [CChar](repeating: 0, count: 64)
        let got = buf.withUnsafeMutableBufferPointer { ptr in
            llama_model_meta_val_str(model, "tokenizer.ggml.add_bos_token", ptr.baseAddress, ptr.count)
        }
        
        // Default to supporting special tokens if not explicitly disabled
        return got <= 0 || String(cString: buf) != "false"
    }
    
    /// Get diagnostic information about the model and llama.cpp version
    static func getModelDiagnostics(model: CLlamaModel) -> String {
        var diagnostics = "🦙 Model Diagnostics:\n"
        
        // Get llama.cpp commit/version if available
        var buf = [CChar](repeating: 0, count: 128)
        _ = buf.withUnsafeMutableBufferPointer { ptr in
            llama_model_meta_val_str(model, "llama.cpp.commit", ptr.baseAddress, ptr.count)
        }
        if buf[0] != 0 {
            diagnostics += "  llama.cpp commit: \(String(cString: buf))\n"
        }
        
        // Architecture
        if let arch = getModelArchitecture(model: model) {
            diagnostics += "  Architecture: \(arch)\n"
        }
        
        // Check for tokenizer model blob (the actual data, not just metadata)
        buf = [CChar](repeating: 0, count: 128)
        let hasTokenizerModel = buf.withUnsafeMutableBufferPointer { ptr in
            llama_model_meta_val_str(model, "tokenizer.ggml.model", ptr.baseAddress, ptr.count)
        }
        diagnostics += "  Tokenizer model blob: \(hasTokenizerModel > 0 ? "present" : "MISSING")\n"
        
        // Tokenizer type
        buf = [CChar](repeating: 0, count: 128)
        _ = buf.withUnsafeMutableBufferPointer { ptr in
            llama_model_meta_val_str(model, "tokenizer.ggml.type", ptr.baseAddress, ptr.count)
        }
        if buf[0] != 0 {
            diagnostics += "  Tokenizer type: \(String(cString: buf))\n"
        }
        
        // Tokenizer info
        diagnostics += "  Has usable tokenizer: \(hasUsableTokenizer(model: model))\n"
        diagnostics += "  Vocab size: \(getModelVocabularySize(model: model))\n"
        
        // BOS/EOS preferences
        diagnostics += "  Add BOS: \(shouldAddBOSToken(model: model))\n"
        diagnostics += "  Add EOS: \(shouldAddEOSToken(model: model))\n"
        
        return diagnostics
    }
}
