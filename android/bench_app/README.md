# EAT Bench App

Minimal Android benchmark harness for on-device LLM runtimes (MediaPipe Tasks GenAI; llama.cpp JNI stub). Designed to be driven by the desktop orchestrator over `adb`.

## Layout

```
android/bench_app/
  settings.gradle.kts
  build.gradle.kts
  gradle.properties
  app/
    build.gradle.kts
    proguard-rules.pro
    src/main/
      AndroidManifest.xml
      java/com/eat/bench/{MainActivity,BenchRuntime,MediaPipeRuntime,LlamaCppRuntime,Metrics}.kt
      res/values/{strings,themes}.xml
```

## Build

A Gradle wrapper is intentionally not committed. Generate it once with a local Gradle (8.7+):

```
cd android/bench_app
gradle wrapper --gradle-version 8.7
./gradlew :app:assembleDebug
```

The APK lands at `app/build/outputs/apk/debug/app-debug.apk`.

Install:

```
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

## Prompt input JSON

```
{
  "prompts": [
    {"id": "p1", "text": "Write a haiku about edge inference."},
    {"id": "p2", "text": "Explain MoE in 2 sentences."}
  ],
  "max_tokens": 128
}
```

Push it (and the model) to the device:

```
adb push prompts.json /data/local/tmp/eat/<run_id>/prompts.json
adb push model.task    /data/local/tmp/eat/<run_id>/model.task
```

`/data/local/tmp` is readable by the app process via the absolute path; no SAF or storage permission needed for files there.

## Run

```
adb shell am start --wait \
  -n com.eat.bench/.MainActivity \
  -a com.eat.bench.RUN_BENCH \
  --es model_path    /data/local/tmp/eat/<run_id>/model.task \
  --es runtime       mediapipe \
  --es prompts_path  /data/local/tmp/eat/<run_id>/prompts.json \
  --es output_path   /data/local/tmp/eat/<run_id>/results.json \
  --ei warmup 1
```

The activity finishes itself after writing the JSON and also broadcasts `com.eat.bench.RUN_DONE` with the `result_path` extra. Pull results:

```
adb pull /data/local/tmp/eat/<run_id>/results.json ./results.json
```

## Output JSON

```
{
  "device": { "manufacturer": "...", "model": "...", "android_sdk": 34 },
  "runtime": "mediapipe",
  "model_path": "...",
  "results": [
    { "id": "p1", "ttft_ms": 142.5, "decode_tokens_per_s": 38.4,
      "total_tokens": 96, "total_ms": 2641.0, "peak_rss_mb": 1820.3 }
  ],
  "summary": {
    "median_decode_tokens_per_s": 37.9,
    "p95_ttft_ms": 220.1,
    "peak_rss_mb": 1850.0
  }
}
```

## Runtimes

- `mediapipe` (default): uses `com.google.mediapipe:tasks-genai`. Expects a `.task` bundle on disk.
- `llamacpp`: stub. Hook in a JNI build that links libllama.so for the device ABI later.

Both implement `BenchRuntime` (`open`, `generate`, `close`) so `MainActivity` is runtime-agnostic.
