package com.eat.bench

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import kotlinx.serialization.Serializable

@Serializable
data class DeviceInfo(
    val manufacturer: String,
    val model: String,
    val android_sdk: Int
)

@Serializable
data class PromptResult(
    val id: String,
    val ttft_ms: Double,
    val decode_tokens_per_s: Double,
    val total_tokens: Int,
    val total_ms: Double,
    val peak_rss_mb: Double
)

@Serializable
data class RunSummary(
    val median_decode_tokens_per_s: Double,
    val p95_ttft_ms: Double,
    val peak_rss_mb: Double
)

@Serializable
data class BenchReport(
    val device: DeviceInfo,
    val runtime: String,
    val model_path: String,
    val results: List<PromptResult>,
    val summary: RunSummary
)

@Serializable
data class PromptSpec(val id: String, val text: String)

@Serializable
data class PromptSet(val prompts: List<PromptSpec>, val max_tokens: Int = 128)

object Metrics {
    fun deviceInfo(): DeviceInfo = DeviceInfo(
        manufacturer = Build.MANUFACTURER ?: "unknown",
        model = Build.MODEL ?: "unknown",
        android_sdk = Build.VERSION.SDK_INT
    )

    fun processRssMb(): Double {
        val rt = Runtime.getRuntime()
        val used = rt.totalMemory() - rt.freeMemory()
        return used / (1024.0 * 1024.0)
    }

    fun systemUsedMb(context: Context): Double {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo()
        am.getMemoryInfo(mi)
        val used = mi.totalMem - mi.availMem
        return used / (1024.0 * 1024.0)
    }

    fun median(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.sorted()
        val n = sorted.size
        return if (n % 2 == 1) sorted[n / 2] else (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }

    fun percentile(values: List<Double>, p: Double): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.sorted()
        val idx = ((sorted.size - 1) * p).toInt().coerceIn(0, sorted.size - 1)
        return sorted[idx]
    }
}
