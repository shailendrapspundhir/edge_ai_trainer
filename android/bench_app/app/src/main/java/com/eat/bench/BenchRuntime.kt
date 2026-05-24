package com.eat.bench

interface BenchRuntime : AutoCloseable {
    fun open(modelPath: String)
    fun generate(prompt: String, maxTokens: Int, onFirstToken: () -> Unit): GenerationResult
    override fun close()
}

data class GenerationResult(
    val text: String,
    val tokenCount: Int
)
