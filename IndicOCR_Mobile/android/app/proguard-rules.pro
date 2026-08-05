# proguard-rules.pro
# ─────────────────────────────────────────────────────────────────────────────
# ProGuard / R8 rules for ONNX Runtime Mobile + Indic OCR engine
# ─────────────────────────────────────────────────────────────────────────────
 
# Keep ONNX Runtime classes (required – ORT uses reflection internally)
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**
 
# Keep our OCR engine
-keep class com.indicocr.IndicOCREngine { *; }
-keep class com.indicocr.IndicOCREngine$* { *; }
 
# Keep JSON parsing classes (used for label_map.json)
-keep class org.json.** { *; }

# Keep Apache POI and StAX classes (required for Excel export)
-keep class org.apache.poi.** { *; }
-keep class org.apache.xmlbeans.** { *; }
-keep class javax.xml.stream.** { *; }
-keep class com.fasterxml.aalto.** { *; }
-keep class org.codehaus.stax2.** { *; }

# Suppress warnings for POI and its dependencies
-dontwarn org.apache.poi.**
-dontwarn org.apache.xmlbeans.**
-dontwarn javax.xml.stream.**
-dontwarn com.fasterxml.aalto.**
-dontwarn java.awt.**
-dontwarn org.etsi.**
-dontwarn org.openxmlformats.**

# Suppress warnings for OkHttp and Okio
-dontwarn okhttp3.**
-dontwarn okio.**

# Suppress warnings for optional native libs
-dontwarn com.microsoft.onnxruntime.**
 
 