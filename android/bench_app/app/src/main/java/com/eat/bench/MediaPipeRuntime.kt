package com.eat.bench

import android.content.Context
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInference.LlmInferenceOptions
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

class MediaPipeRuntime(private val context: Context) : BenchRuntime {

    private var engine: LlmInference? = null
    private var configuredMaxTokens: Int = 512

    override fun open(modelPath: String) {
        val opts = LlmInferenceOptions.builder()
            .setModelPath(modelPath)
            .setMaxTokens(configuredMaxTokens)
            .build()
        engine = LlmInference.createFromOptions(context, opts)
    }

    fun setMaxTokens(value: Int) {
        configuredMaxTokens = value
    }

    override fun generate(prompt: String, maxTokens: Int, onFirstToken: () -> Unit): GenerationResult {
        val e = engine ?: error("MediaPipeRuntime not opened")
        val firstFired = AtomicBoolean(false)
        val tokenCount = AtomicInteger(0)
        val sb = StringBuilder()
        val done = CountDownLatch(1)
        val errorRef = AtomicReference<Throwable?>(null)

        e.generateResponseAsync(prompt) { partial, finished ->
            try {
                if (!partial.isNullOrEmpty()) {
                    if (firstFired.compareAndSet(false, true)) {
                        onFirstToken()
                    }
                    sb.append(partial)
                    tokenCount.incrementAndGet()
                }
                if (finished) done.countDown()
            } catch (t: Throwable) {
                errorRef.set(t)
                done.countDown()
            }
        }
        done.await()
        errorRef.get()?.let { throw it }
        return GenerationResult(text = sb.toString(), tokenCount = tokenCount.get())
    }

    override fun close() {
        engine?.close()
        engine = null
    }
}
