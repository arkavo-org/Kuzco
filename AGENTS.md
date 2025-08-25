# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kuzco is a Swift package that provides local Large Language Model (LLM) inference for iOS and macOS applications, built on top of llama.cpp. It enables on-device AI without network dependencies.

## Build and Development Commands

```bash
# Build the package
swift build

# Run tests
swift test

# Build for specific platform (iOS/macOS)
xcodebuild -scheme Kuzco -destination 'platform=iOS Simulator,name=iPhone 15'
xcodebuild -scheme Kuzco -destination 'platform=macOS'

# Clean build
swift package clean
```

## Architecture and Key Components

### Core Architecture

The package follows a layered architecture with clear separation of concerns:

1. **Public Interface Layer** (`KuzcoSwift.swift`)
   - Main entry point via `Kuzco` actor (singleton pattern)
   - Manages instance lifecycle and caching
   - Provides safe loading methods with error handling
   - Auto-detects model architectures from filenames

2. **Instance Management** (`LlamaInstance.swift`)
   - Encapsulates llama.cpp context and model
   - Handles prediction streaming with async/await
   - Manages conversation context and memory
   - Thread-safe operations via actor model

3. **Bridge Layer** (`System/LlamaKitBridge.swift`)
   - Direct interface to llama.cpp C++ library via Swift-C interop
   - Handles model loading, tokenization, and inference
   - GGUF file validation and error handling
   - Memory management for C++ resources

4. **Configuration System** (`Configuration/`)
   - `ModelProfile`: Model identification and architecture specification
   - `InstanceSettings`: Hardware optimization (GPU layers, threads, context)
   - `PredictionConfig`: Generation parameters (temperature, sampling)
   - Auto-detection of architectures with fallback support

5. **Dialog Management** (`Dialog/`)
   - `InteractionFormatter`: Architecture-specific prompt formatting
   - `Turn`: Conversation history management
   - `EosHandler`: End-of-sequence token handling per architecture
   - Support for 13+ model architectures (LLaMA, Qwen, Mistral, etc.)

### Key Design Patterns

- **Actor-based concurrency**: Thread-safe instance management
- **Streaming responses**: AsyncStream for real-time generation
- **Caching strategy**: In-memory and persistent model caching
- **Error recovery**: Comprehensive error types with recovery suggestions
- **Architecture detection**: Automatic model type detection from filenames

### Critical Implementation Details

1. **Vendored XCFramework**: The package uses a vendored `llama.xcframework` in `Vendors/` directory, built with Metal support and Gemma-3 architecture compatibility.

2. **Model Loading Flow**:
   - File validation (GGUF magic bytes check)
   - Architecture auto-detection or fallback
   - Context creation with hardware-optimized settings
   - Instance caching for reuse

3. **Memory Management**:
   - Automatic cleanup of C++ resources via deinit
   - Context size management for iOS memory constraints
   - GPU layer offloading for Apple Silicon optimization

4. **Error Handling**:
   - `KuzcoError` enum with detailed error cases
   - Recovery suggestions for common issues
   - Safe loading methods that return Result types

## Model Architecture Support

The package supports multiple LLM architectures with automatic detection:
- LLaMA (2, 3, 3.1, 3.2)
- Qwen (2, 3)
- Mistral/Mixtral
- Phi (3, 3.5)
- Gemma
- DeepSeek
- Command-R
- Yi
- OpenChat

Each architecture has specific:
- Prompt formatting rules
- EOS (end-of-sequence) tokens
- Context handling requirements

## Testing Considerations

- Test with various GGUF model files
- Verify architecture detection logic
- Check memory management on iOS devices
- Validate streaming response handling
- Test error recovery paths

## Performance Optimization

- Adjust `gpuOffloadLayers` based on device capabilities
- Use appropriate `contextLength` for memory constraints
- Configure `cpuThreadCount` for optimal parallelism
- Use quantized models (Q4_0, Q4_1) for mobile devices