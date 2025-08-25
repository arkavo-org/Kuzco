//
//  KuzcoBridge.h
//  Kuzco
//
//  C bridge functions for llama.cpp integration
//

#ifndef KUZCO_BRIDGE_H
#define KUZCO_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct llama_model;

// Check if model should add BOS token based on its configuration
int kuzco_should_add_bos_token(const struct llama_model * model);

// Check if model should add EOS token based on its configuration
int kuzco_should_add_eos_token(const struct llama_model * model);

// Check if model has a valid tokenizer
int kuzco_model_has_tokenizer(const struct llama_model * model);

// Get model architecture (returns bytes written, -1 on error)
int kuzco_get_model_architecture(const struct llama_model * model, char * buf, size_t buf_size);

// Check if model supports special token parsing
int kuzco_supports_special_tokens(const struct llama_model * model);

#ifdef __cplusplus
}
#endif

#endif // KUZCO_BRIDGE_H