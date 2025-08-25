//
//  KuzcoBridge.c
//  Kuzco
//
//  C bridge functions for llama.cpp integration
//

#include <llama.h>
#include <string.h>

// Check if model should add BOS token based on its configuration
int kuzco_should_add_bos_token(const struct llama_model * model) {
    if (model == NULL) return 0;
    
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    if (vocab == NULL) return 0;
    
    return llama_vocab_get_add_bos(vocab) ? 1 : 0;
}

// Check if model should add EOS token based on its configuration
int kuzco_should_add_eos_token(const struct llama_model * model) {
    if (model == NULL) return 0;
    
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    if (vocab == NULL) return 0;
    
    return llama_vocab_get_add_eos(vocab) ? 1 : 0;
}

// Check if model has a valid tokenizer
int kuzco_model_has_tokenizer(const struct llama_model * model) {
    if (model == NULL) return 0;
    
    // Check vocab size
    const int32_t n_vocab = llama_n_vocab(model);
    if (n_vocab <= 0) return 0;
    
    // Check for tokenizer metadata
    char buf[64] = {0};
    if (llama_model_meta_val_str(model, "tokenizer.ggml.model", buf, sizeof(buf)) > 0) {
        // Found tokenizer model type (e.g., "llama", "gpt2", "bert")
        return 1;
    }
    
    // Also check for tokenizer type
    memset(buf, 0, sizeof(buf));
    if (llama_model_meta_val_str(model, "tokenizer.ggml.type", buf, sizeof(buf)) > 0) {
        // Found tokenizer type (e.g., "spm", "bpe", "wpm")
        return 1;
    }
    
    // Check for any tokenizer-related metadata
    memset(buf, 0, sizeof(buf));
    if (llama_model_meta_val_str(model, "tokenizer.ggml.tokens", buf, sizeof(buf)) > 0) {
        return 1;
    }
    
    // If we have vocab size but no tokenizer metadata, it might still work
    // but we should warn about it
    return n_vocab > 100 ? 1 : 0;  // Assume valid if vocab is reasonably sized
}

// Get model architecture
int kuzco_get_model_architecture(const struct llama_model * model, char * buf, size_t buf_size) {
    if (model == NULL || buf == NULL || buf_size == 0) return -1;
    
    return llama_model_meta_val_str(model, "general.architecture", buf, buf_size);
}

// Check if model supports special token parsing
int kuzco_supports_special_tokens(const struct llama_model * model) {
    if (model == NULL) return 0;
    
    // Check if model has special tokens defined
    char buf[64] = {0};
    if (llama_model_meta_val_str(model, "tokenizer.ggml.add_bos_token", buf, sizeof(buf)) > 0) {
        return 1;
    }
    
    // Most models support special tokens, but Gemma might not use them the same way
    memset(buf, 0, sizeof(buf));
    if (llama_model_meta_val_str(model, "general.architecture", buf, sizeof(buf)) > 0) {
        if (strstr(buf, "gemma") != NULL) {
            return 0;  // Gemma models typically don't use special tokens the same way
        }
    }
    
    return 1;  // Default to supporting special tokens
}