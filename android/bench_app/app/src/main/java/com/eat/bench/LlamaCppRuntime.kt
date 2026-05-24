package com.eat.bench

class LlamaCppRuntime : BenchRuntime {
    override fun open(modelPath: String) {
        throw NotImplementedError("wire llama.cpp Android lib later")
    }

    override fun generate(prompt: String, maxTokens: Int, onFirstToken: () -> Unit): GenerationResult {
        throw NotImplementedError("wire llama.cpp Android lib later")
    }

    override fun close() {
        // no-op
    }
}
