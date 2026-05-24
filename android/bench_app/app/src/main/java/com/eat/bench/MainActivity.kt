package com.eat.bench

import android.content.Intent
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import kotlin.system.measureNanoTime

class MainActivity : AppCompatActivity() {

    private val tag = "EATBench"
    private val json = Json { prettyPrint = true; ignoreUnknownKeys = true }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val action = intent?.action
        if (action == ACTION_RUN_BENCH) {
            CoroutineScope(Dispatchers.Default).launch {
                runBench(intent)
                finish()
            }
        } else {
            finish()
        }
    }

    private fun runBench(intent: Intent) {
        val modelPath = intent.getStringExtra("model_path") ?: error("missing model_path")
        val runtimeName = intent.getStringExtra("runtime") ?: "mediapipe"
        val promptsPath = intent.getStringExtra("prompts_path") ?: error("missing prompts_path")
        val outputPath = intent.getStringExtra("output_path") ?: error("missing output_path")
        val warmup = intent.getIntExtra("warmup", 1)

        val promptSet = json.decodeFromString<PromptSet>(File(promptsPath).readText())

        val runtime: BenchRuntime = when (runtimeName) {
            "mediapipe" -> MediaPipeRuntime(applicationContext).also { it.setMaxTokens(promptSet.max_tokens) }
            "llamacpp" -> LlamaCppRuntime()
            else -> error("unknown runtime: $runtimeName")
        }

        val results = mutableListOf<PromptResult>()
        var peakRssMb = 0.0

        try {
            runtime.open(modelPath)

            if (promptSet.prompts.isNotEmpty()) {
                val warmupPrompt = promptSet.prompts.first().text
                repeat(warmup) {
                    runtime.generate(warmupPrompt, maxTokens = promptSet.max_tokens, onFirstToken = {})
                }
            }

            for (spec in promptSet.prompts) {
                System.gc()
                var ttftNs = 0L
                var totalTokens = 0
                val totalNs = measureNanoTime {
                    val startNs = System.nanoTime()
                    val gen = runtime.generate(spec.text, maxTokens = promptSet.max_tokens) {
                        ttftNs = System.nanoTime() - startNs
                    }
                    totalTokens = gen.tokenCount
                }
                val ttftMs = ttftNs / 1_000_000.0
                val totalMs = totalNs / 1_000_000.0
                val decodeMs = (totalMs - ttftMs).coerceAtLeast(1.0)
                val decodeTps = if (totalTokens > 1) (totalTokens - 1) * 1000.0 / decodeMs else 0.0
                val rss = Metrics.processRssMb()
                if (rss > peakRssMb) peakRssMb = rss

                results.add(
                    PromptResult(
                        id = spec.id,
                        ttft_ms = ttftMs,
                        decode_tokens_per_s = decodeTps,
                        total_tokens = totalTokens,
                        total_ms = totalMs,
                        peak_rss_mb = rss
                    )
                )
            }
        } catch (t: Throwable) {
            Log.e(tag, "bench failed", t)
        } finally {
            try { runtime.close() } catch (_: Throwable) {}
        }

        val summary = RunSummary(
            median_decode_tokens_per_s = Metrics.median(results.map { it.decode_tokens_per_s }),
            p95_ttft_ms = Metrics.percentile(results.map { it.ttft_ms }, 0.95),
            peak_rss_mb = peakRssMb
        )

        val report = BenchReport(
            device = Metrics.deviceInfo(),
            runtime = runtimeName,
            model_path = modelPath,
            results = results,
            summary = summary
        )

        val outFile = File(outputPath)
        outFile.parentFile?.mkdirs()
        outFile.writeText(json.encodeToString(report))

        val done = Intent(ACTION_RUN_DONE).apply {
            putExtra("result_path", outputPath)
            setPackage(packageName)
        }
        sendBroadcast(done)
    }

    companion object {
        const val ACTION_RUN_BENCH = "com.eat.bench.RUN_BENCH"
        const val ACTION_RUN_DONE = "com.eat.bench.RUN_DONE"
    }
}
