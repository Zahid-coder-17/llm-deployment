package com.embeddra.edge

import android.Manifest
import android.app.Activity
import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.hardware.camera2.CameraManager
import android.media.AudioManager
import android.net.Uri
import android.os.BatteryManager
import android.os.Bundle
import android.os.SystemClock
import android.provider.AlarmClock
import android.provider.ContactsContract
import android.provider.Settings
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.view.Gravity
import android.view.KeyEvent
import android.view.View
import android.view.inputmethod.EditorInfo
import android.widget.*
import java.io.File
import java.util.Locale
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.regex.Pattern
import kotlin.concurrent.thread

data class ActionStep(
    val type: String,
    val targetId: Int,
    val targetName: String,
    val thought: String,
    val action: String,
    val realScreenDumping: String
)

class MainActivity : Activity() {

    companion object {
        var instance: MainActivity? = null
        private const val REQUEST_CODE_ATTACH_DOC = 2001
    }

    private lateinit var runtime: EmbeddraRuntime
    private lateinit var vocabDecoder: VocabularyDecoder
    private var speechRecognizer: SpeechRecognizer? = null
    private var isListening = false
    @Volatile private var isModelReady = false
    @Volatile private var isGenerating = false

    private lateinit var chatMessagesContainer: LinearLayout
    private lateinit var chatScrollView: ScrollView
    private lateinit var promptInput: EditText
    private lateinit var btnSend: ImageButton
    private lateinit var btnMic: ImageButton
    private lateinit var btnAttachDoc: ImageButton
    private lateinit var btnNewChat: ImageButton
    private lateinit var quickSuggestions: LinearLayout
    private lateinit var headerSubtitle: TextView

    // Document Attachment Controls
    private lateinit var docAttachmentBar: LinearLayout
    private lateinit var tvAttachedDocName: TextView
    private lateinit var btnRemoveDoc: TextView
    private var attachedDocContent: String? = null
    private var attachedDocName: String? = null

    private var pendingQuery: String? = null

    // Multi-turn conversational history buffer for pure LLM reasoning
    private val conversationHistory = mutableListOf<Pair<String, String>>()
    private val isExecuting = AtomicBoolean(false)

    override fun onNewIntent(intent: Intent?) {
        super.onNewIntent(intent)
        handleIntentQuery(intent)
    }

    private fun handleIntentQuery(intent: Intent?) {
        val q = intent?.getStringExtra("query")?.trim() ?: return
        if (q.isNotEmpty()) {
            if (isModelReady) {
                runOnUiThread { submitUserQuery(q) }
            } else {
                pendingQuery = q
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        instance = this
        setContentView(R.layout.activity_main)

        chatMessagesContainer = findViewById(R.id.chatMessagesContainer)
        chatScrollView = findViewById(R.id.chatScrollView)
        promptInput = findViewById(R.id.promptInput)
        btnSend = findViewById(R.id.btnSend)
        btnMic = findViewById(R.id.btnMic)
        btnAttachDoc = findViewById(R.id.btnAttachDoc)
        btnNewChat = findViewById(R.id.btnNewChat)
        quickSuggestions = findViewById(R.id.quickSuggestions)
        headerSubtitle = findViewById(R.id.headerSubtitle)

        docAttachmentBar = findViewById(R.id.docAttachmentBar)
        tvAttachedDocName = findViewById(R.id.tvAttachedDocName)
        btnRemoveDoc = findViewById(R.id.btnRemoveDoc)

        runtime = EmbeddraRuntime.getInstance()
        runtime.initialize()

        vocabDecoder = VocabularyDecoder(this)
        vocabDecoder.loadVocabulary()

        initSpeech()

        // ChatGPT-Style Morphing Send / Stop Button Click Listener
        btnSend.setOnClickListener {
            if (isGenerating) {
                stopCurrentGeneration()
            } else {
                val query = promptInput.text.toString().trim()
                if (query.isNotEmpty() || attachedDocContent != null) {
                    val finalQuery = if (query.isNotEmpty()) query else "Please analyze and summarize the attached document."
                    submitUserQuery(finalQuery)
                }
            }
        }

        promptInput.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_SEND) {
                if (!isGenerating) {
                    val query = promptInput.text.toString().trim()
                    if (query.isNotEmpty() || attachedDocContent != null) {
                        val finalQuery = if (query.isNotEmpty()) query else "Please analyze and summarize the attached document."
                        submitUserQuery(finalQuery)
                    }
                }
                true
            } else {
                false
            }
        }

        btnMic.setOnClickListener {
            toggleSpeechInput()
        }

        btnAttachDoc.setOnClickListener {
            openDocumentPicker()
        }

        btnRemoveDoc.setOnClickListener {
            clearAttachedDocument()
        }

        btnNewChat.setOnClickListener {
            resetConversation()
        }

        setupChips()
        handleIntentQuery(intent)
        loadLocalWeights()
    }

    private fun openDocumentPicker() {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "*/*"
            putExtra(Intent.EXTRA_MIME_TYPES, arrayOf("text/*", "application/json", "application/pdf", "text/plain", "application/javascript"))
        }
        try {
            startActivityForResult(intent, REQUEST_CODE_ATTACH_DOC)
        } catch (e: Exception) {
            Toast.makeText(this, "Document picker unavailable: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_CODE_ATTACH_DOC && resultCode == RESULT_OK) {
            val uri = data?.data ?: return
            handleSelectedDocument(uri)
        }
    }

    private fun handleSelectedDocument(uri: Uri) {
        try {
            var name = "document.txt"
            contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                val nameIndex = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                if (nameIndex != -1 && cursor.moveToFirst()) {
                    name = cursor.getString(nameIndex) ?: "document.txt"
                }
            }
            val text = contentResolver.openInputStream(uri)?.bufferedReader()?.use { it.readText() } ?: ""
            if (text.isNotBlank()) {
                attachedDocContent = text
                attachedDocName = name
                val sizeKb = text.toByteArray().size / 1024.0
                val sizeStr = if (sizeKb >= 1.0) "%.1f KB".format(sizeKb) else "${text.length} chars"
                tvAttachedDocName.text = "📄 $name ($sizeStr)"
                docAttachmentBar.visibility = View.VISIBLE
                docAttachmentBar.alpha = 0f
                docAttachmentBar.animate().alpha(1f).setDuration(220).start()
                Toast.makeText(this, "Attached: $name", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Document is empty", Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            android.util.Log.e("MainActivity", "Doc read error", e)
            Toast.makeText(this, "Failed to read doc: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun clearAttachedDocument() {
        attachedDocContent = null
        attachedDocName = null
        if (docAttachmentBar.visibility == View.VISIBLE) {
            docAttachmentBar.animate().alpha(0f).setDuration(160).withEndAction {
                docAttachmentBar.visibility = View.GONE
            }.start()
        }
    }

    fun setGeneratingState(generating: Boolean) {
        isGenerating = generating
        runOnUiThread {
            if (generating) {
                btnSend.setImageResource(R.drawable.ic_stop_square)
                btnSend.setBackgroundResource(R.drawable.bg_chatgpt_stop)
                btnSend.contentDescription = "Stop Generation"
            } else {
                btnSend.setImageResource(R.drawable.ic_arrow_up)
                btnSend.setBackgroundResource(R.drawable.bg_chatgpt_send)
                btnSend.contentDescription = "Send"
            }
        }
    }

    fun stopCurrentGeneration() {
        try {
            runtime.stopGeneration()
            setGeneratingState(false)
            YoriAccessibilityService.instance?.updateFloatingHud("⚡ Generation Stopped")
            YoriAccessibilityService.instance?.hideFloatingHud(1200)
        } catch (_: Exception) {}
    }

    override fun onResume() {
        super.onResume()
        updateHeaderStatus()
        YoriAccessibilityService.instance?.hideFloatingHud(0)
    }

    override fun onPause() {
        super.onPause()
        if (isGenerating) {
            YoriAccessibilityService.instance?.updateFloatingHud("⚡ Yori: Generating in background... (Tap to open)")
        }
    }

    private fun updateHeaderStatus() {
        val modelTag = if (isModelReady) "LLM Ready ✓" else "Loading LLM..."
        if (YoriAccessibilityService.isServiceActive) {
            headerSubtitle.text = "$modelTag • Inspector: Active ✓"
            headerSubtitle.setTextColor(Color.parseColor("#10B981"))
            headerSubtitle.setOnClickListener(null)
        } else {
            headerSubtitle.text = "$modelTag • Tap to enable Inspector"
            headerSubtitle.setTextColor(Color.parseColor("#CC785C"))
            headerSubtitle.setOnClickListener {
                startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
            }
        }
    }

    private fun setupChips() {
        findViewById<TextView>(R.id.chipAlarm).setOnClickListener {
            submitUserQuery("Check my alarms on this phone")
        }
        findViewById<TextView>(R.id.chipFlight).setOnClickListener {
            submitUserQuery("Turn on flashlight")
        }
        findViewById<TextView>(R.id.chipWhatsApp).setOnClickListener {
            submitUserQuery("Summarize message from Zahid on WhatsApp and reply")
        }
        findViewById<TextView>(R.id.chipWifi).setOnClickListener {
            submitUserQuery("Turn off Wi-Fi in settings")
        }
    }

    private fun initSpeech() {
        val perms = mutableListOf<String>()
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            perms.add(Manifest.permission.RECORD_AUDIO)
        }
        if (checkSelfPermission(Manifest.permission.READ_CONTACTS) != PackageManager.PERMISSION_GRANTED) {
            perms.add(Manifest.permission.READ_CONTACTS)
        }
        if (perms.isNotEmpty()) {
            requestPermissions(perms.toTypedArray(), 101)
        }

        if (SpeechRecognizer.isRecognitionAvailable(this)) {
            speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
            speechRecognizer?.setRecognitionListener(object : RecognitionListener {
                override fun onReadyForSpeech(params: Bundle?) {}
                override fun onBeginningOfSpeech() {}
                override fun onRmsChanged(rmsdB: Float) {}
                override fun onBufferReceived(buffer: ByteArray?) {}
                override fun onEndOfSpeech() {
                    isListening = false
                    btnMic.setBackgroundResource(R.drawable.bg_thought_box)
                }
                override fun onError(error: Int) {
                    isListening = false
                    btnMic.setBackgroundResource(R.drawable.bg_thought_box)
                }
                override fun onResults(results: Bundle?) {
                    val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    if (!matches.isNullOrEmpty()) {
                        promptInput.setText(matches[0])
                        submitUserQuery(matches[0])
                    }
                    isListening = false
                    btnMic.setBackgroundResource(R.drawable.bg_thought_box)
                }
                override fun onPartialResults(partialResults: Bundle?) {
                    val matches = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    if (!matches.isNullOrEmpty()) {
                        promptInput.setText(matches[0])
                    }
                }
                override fun onEvent(eventType: Int, params: Bundle?) {}
            })
        }
    }

    private fun toggleSpeechInput() {
        if (isListening) {
            speechRecognizer?.stopListening()
            isListening = false
            btnMic.setBackgroundResource(R.drawable.bg_thought_box)
        } else {
            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
                putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
                putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            }
            try {
                speechRecognizer?.startListening(intent)
                isListening = true
                btnMic.setBackgroundResource(R.drawable.bg_button_terracotta)
                Toast.makeText(this, "Listening offline...", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Toast.makeText(this, "Speech recognition error: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private class StreamingBubbleViews(
        val container: LinearLayout,
        val thoughtLabel: TextView,
        val thoughtContent: TextView,
        val responseView: TextView
    )

    private fun createStreamingBubble(): StreamingBubbleViews {
        val agentLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.START
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { setMargins(0, 10, 0, 10) }
            alpha = 0f
            translationY = 30f
            animate().alpha(1f).translationY(0f).setDuration(220).start()
        }

        val card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundResource(R.drawable.bg_bubble_agent)
            setPadding(26, 22, 26, 22)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }

        val thoughtHeader = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setBackgroundResource(R.drawable.bg_thought_box)
            setPadding(20, 14, 20, 14)
            isClickable = true
            isFocusable = true
        }
        val thoughtLabel = TextView(this).apply {
            text = "⚡ Yori Thinking Live...  ▾"
            setTextColor(Color.parseColor("#CC785C"))
            textSize = 11f
        }
        thoughtHeader.addView(thoughtLabel)

        val thoughtContent = TextView(this).apply {
            text = ""
            setTextColor(Color.parseColor("#9E9AA6"))
            textSize = 10f
            setPadding(16, 14, 16, 14)
            visibility = View.GONE
        }

        thoughtHeader.setOnClickListener {
            if (thoughtContent.visibility == View.VISIBLE) {
                thoughtContent.visibility = View.GONE
                thoughtLabel.text = thoughtLabel.text.toString().replace("▴", "▾")
            } else {
                thoughtContent.visibility = View.VISIBLE
                thoughtLabel.text = thoughtLabel.text.toString().replace("▾", "▴")
            }
        }

        val responseView = TextView(this).apply {
            setTextColor(Color.WHITE)
            textSize = 13f
            setLineSpacing(4f, 1.15f)
            setPadding(0, 16, 0, 0)
            text = "▍"
        }

        card.addView(thoughtHeader)
        card.addView(thoughtContent)
        card.addView(responseView)
        agentLayout.addView(card)
        chatMessagesContainer.addView(agentLayout)
        chatScrollView.post { chatScrollView.fullScroll(View.FOCUS_DOWN) }

        return StreamingBubbleViews(agentLayout, thoughtLabel, thoughtContent, responseView)
    }

    private fun submitUserQuery(rawQuery: String) {
        val query = rawQuery.trim()
        if (query.isEmpty() && attachedDocContent == null) return
        if (!isExecuting.compareAndSet(false, true)) {
            runOnUiThread {
                Toast.makeText(this, "Yori is already running a task. Please wait...", Toast.LENGTH_SHORT).show()
            }
            return
        }
        android.util.Log.i("MainActivity", ">> RECEIVED QUERY: '$query' (docAttached=${attachedDocName != null})")

        promptInput.setText("")
        quickSuggestions.visibility = View.GONE

        val docNameSnapshot = attachedDocName
        val docContentSnapshot = attachedDocContent
        clearAttachedDocument()

        addUserBubble(query, docNameSnapshot)
        val liveBubble = createStreamingBubble()

        thread {
            var step: ActionStep? = null
            try {
                step = executeRealDeviceAction(query, docNameSnapshot, docContentSnapshot, liveBubble)
            } catch (e: Throwable) {
                android.util.Log.e("MainActivity", "Execution error", e)
                step = ActionStep(
                    type = "ERROR",
                    targetId = 0,
                    targetName = "Yori error: ${e.message ?: "Unknown"}",
                    thought = "Exception: ${e.message}",
                    action = "DONE(\"Error\")",
                    realScreenDumping = "[Error: ${e.message}]"
                )
            } finally {
                isExecuting.set(false)
                setGeneratingState(false)
                val finalStep = step
                YoriAccessibilityService.instance?.hideFloatingHud(3500)
                runOnUiThread {
                    if (finalStep != null) {
                        liveBubble.thoughtLabel.text = "Thought (Real Trace)  ▾"
                        liveBubble.thoughtContent.text = "${finalStep.thought}\n\n[REAL DEVICE SCREEN CAPTURE]:\n${finalStep.realScreenDumping}"
                        liveBubble.responseView.text = finalStep.targetName
                    }
                    chatScrollView.post { chatScrollView.fullScroll(View.FOCUS_DOWN) }
                }
            }
        }
    }

    private fun lookupContactPhone(nameQuery: String): String? {
        if (checkSelfPermission(Manifest.permission.READ_CONTACTS) != PackageManager.PERMISSION_GRANTED) {
            return null
        }
        val cleanName = nameQuery.trim()
        if (cleanName.isEmpty()) return null
        val cursor = contentResolver.query(
            ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
            arrayOf(
                ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME,
                ContactsContract.CommonDataKinds.Phone.NUMBER
            ),
            "${ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME} LIKE ?",
            arrayOf("%$cleanName%"),
            null
        )
        cursor?.use {
            if (it.moveToFirst()) {
                return it.getString(1)?.replace(Regex("[\\s\\-()]"), "")
            }
        }
        return null
    }

    private fun setFlashlight(turnOn: Boolean): Boolean {
        return try {
            val camMgr = getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val cameraId = camMgr.cameraIdList.firstOrNull() ?: return false
            camMgr.setTorchMode(cameraId, turnOn)
            true
        } catch (e: Exception) {
            false
        }
    }

    private fun dispatchMediaKey(keyCode: Int) {
        val audioMgr = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val now = SystemClock.uptimeMillis()
        audioMgr.dispatchMediaKeyEvent(KeyEvent(now, now, KeyEvent.ACTION_DOWN, keyCode, 0))
        audioMgr.dispatchMediaKeyEvent(KeyEvent(now, now, KeyEvent.ACTION_UP, keyCode, 0))
    }

    private fun runSingleLlmTurn(
        prompt: String,
        maxTok: Int,
        isUiTurn: Boolean,
        isExternalApp: Boolean,
        liveBubble: StreamingBubbleViews
    ): String {
        setGeneratingState(true)
        android.util.Log.i("MainActivity", ">> PROMPT TO LLM (len=${prompt.length}):\n$prompt")
        val tokens = vocabDecoder.encode(prompt)
        android.util.Log.i("MainActivity", ">> TOKENS ENCODED: count=${tokens.size} first10=${tokens.take(10)} last5=${tokens.takeLast(5)}")
        val latch = CountDownLatch(1)
        val sb = StringBuilder()

        val samplingTemp = 0.0f
        val genThread = thread {
            try {
                runtime.generateTokens(tokens, maxTokens = maxTok, temperature = samplingTemp, listener = object : EmbeddraRuntime.TokenListener {
                    override fun onToken(token: String, tokenId: Int, isFinal: Boolean) {
                        if (tokenId != 128001 && tokenId != 128009) {
                            val decoded = vocabDecoder.decodeToken(tokenId)
                            sb.append(decoded)
                        }
                        val liveText = sb.toString()

                        runOnUiThread {
                            val displayPreview = liveText
                                .substringBefore("<action>")
                                .replace("<thought>", "")
                                .replace("</thought>", "")
                                .trim()
                            liveBubble.responseView.text = if (displayPreview.isNotEmpty()) "$displayPreview ▍" else "▍"
                            liveBubble.thoughtContent.text = "Streaming live from ARM NEON...\n$liveText"
                            chatScrollView.post { chatScrollView.fullScroll(View.FOCUS_DOWN) }

                            val isGoalDriven = liveText.contains("🎯 Goal:") || liveText.contains("<action")
                            if (isExternalApp || isGoalDriven) {
                                val hudPreview = if (displayPreview.isNotEmpty()) {
                                    val lastThoughtLine = displayPreview.lines().lastOrNull { it.isNotBlank() } ?: displayPreview
                                    val cleanLine = if (lastThoughtLine.length > 70) "..." + lastThoughtLine.takeLast(67) else lastThoughtLine
                                    "⚡ Yori: $cleanLine ▍"
                                } else {
                                    "⚡ Yori: Thinking... ▍"
                                }
                                YoriAccessibilityService.instance?.updateFloatingHud(hudPreview)
                            }
                        }

                        val hasThoughtEnd = liveText.contains("</thought>")
                        val hasActionStart = liveText.contains("<action>")
                        val hasActionEnd = liveText.contains("</action>")
                        val isGoalDrivenAction = liveText.contains("🎯 Goal:") || hasActionStart

                        // Pure Conversational answers end cleanly at </thought> with no action
                        val finishedChat = hasThoughtEnd && !isGoalDrivenAction

                        if (isFinal || hasActionEnd || finishedChat) {
                            if (hasActionEnd || finishedChat) {
                                runtime.stopGeneration()
                            }
                            latch.countDown()
                        }
                    }
                })
            } catch (e: Exception) {
                android.util.Log.e("MainActivity", "Gen thread error", e)
            } finally {
                latch.countDown()
            }
        }

        val completed = latch.await(90000, TimeUnit.MILLISECONDS)
        if (!completed) {
            android.util.Log.w("MainActivity", "LLM turn timed out after 90s, forcing stop.")
            runtime.stopGeneration()
        }
        try { genThread.join(2000) } catch (_: Exception) {}
        setGeneratingState(false)
        return sb.toString().trim()
    }

    /**
     * Pure Generalist Autonomous Execution Engine:
     * - Grounds the LLM with the universal system capability contract.
     * - Operates in closed loop: perceives live UI hierarchy on every turn.
     * - Maintains conversation history across turns.
     * - Ingests document content cleanly when provided.
     */
    private fun executeRealDeviceAction(
        query: String,
        docName: String?,
        docContent: String?,
        liveBubble: StreamingBubbleViews
    ): ActionStep {
        val service = YoriAccessibilityService.instance
        val maxTurns = 6
        var lastStep: ActionStep? = null
        var lastActionStr = ""
        var prevScreenDump = ""

        val systemContract = "<system>You are YORI, an on-device executive agent built on the Maria-1B architecture by Embeddra AI Research Lab.\n" +
            "- For conversation, questions, advice, identity, or documents: answer directly inside <thought> with NO <action>.\n" +
            "- For device operations and app workflows: inspect the visible <screen> and execution <history>, reason step-by-step in <thought>:\n" +
            "  🎯 Goal: [Clarified user objective]\n" +
            "  📋 Plan: [Numbered steps to achieve goal]\n" +
            "  ⚡ Immediate Action: [Specific rationale grounded in visible screen and action history]\n" +
            "- Then emit ONE action from:\n" +
            "  OPEN_APP(appName)\n" +
            "  OPEN_SETTINGS(settingName)\n" +
            "  CLICK(elementLabel | [index])\n" +
            "  TYPE(elementLabel | [index], text)\n" +
            "  SCROLL(UP | DOWN | LEFT | RIGHT)\n" +
            "  PRESS_KEY(BACK | HOME | ENTER)\n" +
            "  TORCH(1 | 0)\n" +
            "  VOLUME(UP | DOWN | MUTE)\n" +
            "  SET_ALARM(HH:MM)\n" +
            "  DONE(summary)\n" +
            "</system>\n"

        // Format document text into prompt if attached
        val queryWithDoc = if (!docContent.isNullOrBlank()) {
            val snippet = docContent.take(3500)
            "<document name=\"${docName ?: "document"}\">\n$snippet\n</document>\n\n$query"
        } else {
            query
        }

        for (turn in 1..maxTurns) {
            val rawDump = service?.dumpRealScreen() ?: "[SCREEN: active]"
            val lines = rawDump.lines().map { it.trim() }.filter { it.isNotEmpty() }
            val screenHeader = lines.firstOrNull() ?: "[SCREEN: active]"
            val rawScreenLines = if (lines.size > 1) lines.subList(1, lines.size) else emptyList<String>()
            val chromeNoise = setOf(
                "clear", "more options", "more actions", "action menu", "home", "shorts", "create", "you", "subscriptions",
                "visit site", "subscribe", "community", "videos", "playlists", "posts", "live"
            )
            val seenLabels = HashSet<String>()
            val filteredLines = rawScreenLines.filter { line ->
                val label = line.substringAfter("\"", "").substringBeforeLast("\"", "").trim()
                val labelLower = label.lowercase()
                val isNoise = labelLower.startsWith("@") || label.length <= 1 ||
                    label == "•" || label == "·" || label == "|" ||
                    labelLower.contains(" subscribers") || labelLower.contains(" videos") ||
                    labelLower in chromeNoise
                !isNoise && seenLabels.add(labelLower)
            }
            val screenLines = if (filteredLines.isNotEmpty()) filteredLines else rawScreenLines
            var compactScreen = when {
                screenLines.size > 8 -> screenHeader + "\n" + screenLines.take(8).joinToString("\n")
                screenLines.isNotEmpty() -> screenHeader + "\n" + screenLines.joinToString("\n")
                else -> screenHeader
            }

            // Closed-loop negative environment feedback: if screen did not change after previous action, inform the LLM
            if (turn > 1 && compactScreen == prevScreenDump && lastActionStr.isNotEmpty()) {
                compactScreen += "\n[ENVIRONMENT FEEDBACK: Previous action '$lastActionStr' had no effect. Visible screen is unchanged. Choose a different element from visible list or alternate action.]"
            }
            prevScreenDump = compactScreen

            val isOnYoriSelf = screenHeader.contains(packageName) || screenHeader == "[SCREEN: active]"
            val isExternalApp = !isOnYoriSelf
            val isUiTurn = (turn > 1 && screenLines.isNotEmpty()) || (isExternalApp && screenLines.isNotEmpty())

            val prompt = if (isUiTurn) {
                "${systemContract}<user>$queryWithDoc</user>\n<screen>\n$compactScreen\n</screen>\n<thought>\n"
            } else {
                val historyBlock = StringBuilder()
                if (conversationHistory.isNotEmpty()) {
                    for ((u, a) in conversationHistory.takeLast(3)) {
                        historyBlock.append("<user>$u</user>\n<thought>\n$a\n</thought>\n")
                    }
                }
                "${systemContract}${historyBlock}<user>$queryWithDoc</user>\n<thought>\n"
            }

            val maxTokens = if (isUiTurn) 160 else 320
            val modelOutput = runSingleLlmTurn(prompt, maxTokens, isUiTurn, isExternalApp, liveBubble)
            android.util.Log.i("MainActivity", "Turn=$turn output=$modelOutput")
            val stats = try { runtime.getPerformanceStats() } catch (_: Exception) { null }
            val speedInfo = if (stats != null && stats.tokensGenerated > 0) {
                " | ${stats.timeToFirstTokenMs.toInt()}ms TTFT, ${"%.1f".format(stats.tokensPerSecond)} tok/s"
            } else ""
            val thoughtHeader = "🧠 Maria-1B Core (Step $turn)$speedInfo"

            // Pure Conversational LLM Response (no <action> emitted)
            if (!modelOutput.contains("<action>")) {
                val chatReply = modelOutput.substringBefore("</thought>").replace("<thought>", "").trim()
                if (isExternalApp && turn < maxTurns && chatReply.isEmpty()) {
                    android.util.Log.w("MainActivity", "Turn $turn empty in external app. Retrying next turn...")
                    Thread.sleep(1000)
                    continue
                }
                if (chatReply.isNotEmpty()) {
                    conversationHistory.add(query to chatReply)
                }
                return ActionStep(
                    type = "CHAT",
                    targetId = 0,
                    targetName = if (chatReply.isNotEmpty()) chatReply else "Completed.",
                    thought = "$thoughtHeader\nMode: Local Conversational LLM",
                    action = "CHAT_RESPONSE",
                    realScreenDumping = compactScreen
                )
            }

            // Real-Time Action Step emitted by model
            val llmThought = modelOutput.substringBefore("<action>").replace("</thought>", "").replace("<thought>", "").trim()
            val actionStr = modelOutput.substringAfter("<action>").substringBefore("</action>").trim()
            lastActionStr = actionStr
            val fullThought = if (llmThought.isNotEmpty()) "$thoughtHeader\n$llmThought" else thoughtHeader

            val toolName = actionStr.substringBefore("(").trim().uppercase()
            val actionArgs = extractArgs(actionStr)
            val actionReadable = when (toolName) {
                "OPEN_APP" -> "Opening ${actionArgs.firstOrNull() ?: "app"}"
                "OPEN_SETTINGS" -> "Opening ${actionArgs.firstOrNull() ?: "system"} settings"
                "CLICK" -> "Tapping \"${actionArgs.firstOrNull() ?: "element"}\""
                "TYPE" -> "Typing \"${actionArgs.getOrNull(1) ?: actionArgs.firstOrNull() ?: ""}\""
                else -> actionStr
            }
            YoriAccessibilityService.instance?.updateFloatingHud("⚡ Yori: $actionReadable...")

            // Pure Execution of Model's Chosen Tool
            val step = executeLlmAction(actionStr, fullThought, compactScreen, service)
            lastStep = step

            YoriAccessibilityService.instance?.updateFloatingHud("✓ ${step.targetName}")

            val isNavOrUiStep = toolName in setOf("OPEN_APP", "OPEN_SETTINGS", "CLICK", "TYPE", "SCROLL", "SCROLL_DOWN", "SCROLL_UP", "PRESS_KEY", "WAIT")
            if (!isNavOrUiStep || service == null || toolName == "DONE") {
                return step
            }

            val stepDelay = when (toolName) {
                "OPEN_APP", "OPEN_SETTINGS" -> 2500L
                "CLICK" -> 1500L
                "TYPE" -> 1200L
                "SCROLL", "SCROLL_DOWN", "SCROLL_UP" -> 1000L
                else -> 800L
            }
            Thread.sleep(stepDelay)
        }

        return lastStep ?: ActionStep("DONE", 0, "Autonomous task cycle complete.", "Exhausted max turns", "DONE", "")
    }

    private data class ParsedAction(
        val type: String,
        val targetIndex: Int,
        val targetLabel: String,
        val text: String,
        val rawAction: String
    )

    private fun extractArgs(actionStr: String): List<String> {
        val argsStr = actionStr.substringAfter("(").substringBeforeLast(")")
        if (argsStr.isBlank()) return emptyList()
        val matcher = Pattern.compile("\"([^\"]*)\"|'([^']*)'|([^,]+)").matcher(argsStr)
        val args = mutableListOf<String>()
        while (matcher.find()) {
            val v = (matcher.group(1) ?: matcher.group(2) ?: matcher.group(3))?.trim() ?: ""
            if (v.isNotEmpty()) args.add(v)
        }
        return args
    }

    private fun parseActionString(actionStr: String): ParsedAction {
        val clean = actionStr.trim()
        val type = clean.substringBefore("(").trim().uppercase()
        val args = extractArgs(clean)

        var targetIndex = -1
        var targetLabel = ""
        var text = ""

        when (type) {
            "CLICK", "TAP" -> {
                val arg0 = args.firstOrNull() ?: ""
                val indexMatch = Pattern.compile("\\[(\\d+)\\]").matcher(arg0)
                if (indexMatch.find()) {
                    targetIndex = indexMatch.group(1)?.toIntOrNull() ?: -1
                } else {
                    targetIndex = arg0.toIntOrNull() ?: -1
                }
                targetLabel = arg0.trim('\"', '\'')
            }
            "TYPE", "INPUT_TEXT" -> {
                val arg0 = args.firstOrNull() ?: ""
                val indexMatch = Pattern.compile("\\[(\\d+)\\]").matcher(arg0)
                if (indexMatch.find()) {
                    targetIndex = indexMatch.group(1)?.toIntOrNull() ?: -1
                } else {
                    targetIndex = arg0.toIntOrNull() ?: -1
                }
                targetLabel = arg0.trim('\"', '\'')
                text = args.getOrNull(1)?.trim('\"', '\'') ?: ""
            }
            "OPEN_APP", "OPEN_SETTINGS" -> {
                text = args.firstOrNull()?.trim('\"', '\'') ?: ""
            }
            "SET_ALARM" -> {
                text = args.firstOrNull()?.trim('\"', '\'') ?: "07:00"
            }
            "TORCH" -> {
                text = args.firstOrNull()?.trim('\"', '\'') ?: "1"
            }
            "VOLUME" -> {
                text = args.firstOrNull()?.trim('\"', '\'') ?: "UP"
            }
            "DONE" -> {
                text = args.firstOrNull()?.trim('\"', '\'') ?: "Task complete."
            }
        }

        return ParsedAction(type, targetIndex, targetLabel, text, clean)
    }

    private fun executeLlmAction(
        actionStr: String,
        thoughtStr: String,
        compactScreen: String,
        service: YoriAccessibilityService?
    ): ActionStep {
        val parsed = parseActionString(actionStr)
        val args = extractArgs(actionStr)

        when (parsed.type) {
            "OPEN_SETTINGS" -> {
                val target = (args.firstOrNull() ?: "").uppercase().trim()
                val action = when {
                    target.contains("WIFI") || target.contains("WI-FI") -> Settings.ACTION_WIFI_SETTINGS
                    target.contains("BLUETOOTH") -> Settings.ACTION_BLUETOOTH_SETTINGS
                    target.contains("DISPLAY") || target.contains("BRIGHTNESS") -> Settings.ACTION_DISPLAY_SETTINGS
                    target.contains("SECURITY") || target.contains("LOCK") -> Settings.ACTION_SECURITY_SETTINGS
                    target.contains("BATTERY") -> Intent.ACTION_POWER_USAGE_SUMMARY
                    target.contains("SOUND") || target.contains("VOLUME") -> Settings.ACTION_SOUND_SETTINGS
                    target.contains("APP") -> Settings.ACTION_APPLICATION_SETTINGS
                    target.contains("DATE") || target.contains("TIME") -> Settings.ACTION_DATE_SETTINGS
                    else -> Settings.ACTION_SETTINGS
                }
                try { startActivity(Intent(action).apply { flags = Intent.FLAG_ACTIVITY_NEW_TASK }) } catch (_: Exception) {}
                return ActionStep("OPEN_SETTINGS", 0, "⚙️ Opened $target settings", thoughtStr, actionStr, "[SETTINGS]")
            }
            "OPEN_APP" -> {
                val appTarget = (args.firstOrNull() ?: "").lowercase().trim()
                if (appTarget.contains("setting") || appTarget.contains("wifi") || appTarget.contains("bluetooth")) {
                    return executeLlmAction("OPEN_SETTINGS(\"$appTarget\")", thoughtStr, compactScreen, service)
                }
                val mainIntent = Intent(Intent.ACTION_MAIN, null).apply { addCategory(Intent.CATEGORY_LAUNCHER) }
                val launchables = packageManager.queryIntentActivities(mainIntent, 0)
                val matchedPkg = launchables.firstOrNull { resolveInfo ->
                    val label = resolveInfo.loadLabel(packageManager).toString().lowercase()
                    val pkg = resolveInfo.activityInfo.packageName.lowercase()
                    label == appTarget || label.contains(appTarget) || pkg.contains(appTarget)
                }?.activityInfo?.packageName ?: run {
                    when {
                        appTarget.contains("whatsapp") -> "com.whatsapp"
                        appTarget.contains("chrome") -> "com.android.chrome"
                        appTarget.contains("youtube") -> "com.google.android.youtube"
                        appTarget.contains("clock") || appTarget.contains("alarm") -> "com.sec.android.app.clockpackage"
                        appTarget.contains("setting") -> "com.android.settings"
                        appTarget.contains("camera") -> "com.sec.android.app.camera"
                        appTarget.contains("gmail") || appTarget.contains("email") -> "com.google.android.gm"
                        else -> null
                    }
                }
                if (matchedPkg != null) {
                    val launchIntent = packageManager.getLaunchIntentForPackage(matchedPkg)
                    if (launchIntent != null) {
                        launchIntent.flags = Intent.FLAG_ACTIVITY_NEW_TASK
                        startActivity(launchIntent)
                        return ActionStep("OPEN_APP", 0, "🚀 Opened $appTarget", thoughtStr, actionStr, "[APP_SCREEN]")
                    }
                }
                return ActionStep("OPEN_APP", 0, "Could not find installed app '$appTarget'", thoughtStr, actionStr, compactScreen)
            }
            "SET_ALARM" -> {
                val timeStr = parsed.text.trim()
                val parts = timeStr.split(":")
                val hour = parts.getOrNull(0)?.toIntOrNull() ?: 7
                val minute = parts.getOrNull(1)?.toIntOrNull() ?: 0
                val intent = Intent(AlarmClock.ACTION_SET_ALARM).apply {
                    putExtra(AlarmClock.EXTRA_HOUR, hour)
                    putExtra(AlarmClock.EXTRA_MINUTES, minute)
                    putExtra(AlarmClock.EXTRA_MESSAGE, "Yori Alarm")
                    putExtra(AlarmClock.EXTRA_SKIP_UI, false)
                    flags = Intent.FLAG_ACTIVITY_NEW_TASK
                }
                try {
                    startActivity(intent)
                    return ActionStep("SET_ALARM", 0, "⏰ Set alarm for %02d:%02d".format(hour, minute), thoughtStr, actionStr, "[ALARM_SET]")
                } catch (e: Exception) {
                    return ActionStep("SET_ALARM", 0, "Failed to set alarm: ${e.message}", thoughtStr, actionStr, compactScreen)
                }
            }
            "TORCH" -> {
                val on = parsed.text == "1" || parsed.text.equals("ON", ignoreCase = true)
                val ok = setFlashlight(on)
                return ActionStep("TORCH", 0, if (ok) (if (on) "🔦 Flashlight turned ON" else "🔦 Flashlight turned OFF") else "Flashlight toggle failed", thoughtStr, actionStr, compactScreen)
            }
            "VOLUME" -> {
                val direction = parsed.text.uppercase()
                when {
                    direction.contains("UP") -> dispatchMediaKey(KeyEvent.KEYCODE_VOLUME_UP)
                    direction.contains("DOWN") -> dispatchMediaKey(KeyEvent.KEYCODE_VOLUME_DOWN)
                    direction.contains("MUTE") -> dispatchMediaKey(KeyEvent.KEYCODE_VOLUME_MUTE)
                }
                return ActionStep("VOLUME", 0, "🔊 Adjusted volume ($direction)", thoughtStr, actionStr, compactScreen)
            }
            "SCROLL", "SCROLL_DOWN" -> {
                val ok = service?.scrollDown() ?: false
                return ActionStep("SCROLL", 0, if (ok) "⬇️ Scrolled down" else "Reached bottom", thoughtStr, actionStr, compactScreen)
            }
            "SCROLL_UP" -> {
                val ok = service?.scrollUp() ?: false
                return ActionStep("SCROLL_UP", 0, if (ok) "⬆️ Scrolled up" else "Reached top", thoughtStr, actionStr, compactScreen)
            }
            "PRESS_KEY" -> {
                val key = parsed.text.uppercase()
                val ok = when {
                    key.contains("BACK") -> service?.goBack() ?: false
                    key.contains("HOME") -> service?.goHome() ?: false
                    else -> false
                }
                return ActionStep("PRESS_KEY", 0, if (ok) "Navigated $key" else "Key $key failed", thoughtStr, actionStr, compactScreen)
            }
        }

        val actionSummary = when (parsed.type) {
            "CLICK", "TAP" -> {
                if (parsed.targetIndex == -1 && parsed.targetLabel.isNotEmpty()) {
                    val success = service?.clickElementByText(parsed.targetLabel) ?: false
                    if (success) "Tapped '${parsed.targetLabel}'" else "Searched for '${parsed.targetLabel}'"
                } else {
                    val success = service?.clickElement(parsed.targetIndex) ?: false
                    if (success) "Tapped element [${parsed.targetIndex}]" else "Targeted element [${parsed.targetIndex}]"
                }
            }
            "TYPE", "INPUT_TEXT" -> {
                if (parsed.targetIndex == -1 && parsed.targetLabel.isNotEmpty()) {
                    val success = service?.typeIntoFocusedOrTarget(parsed.text, parsed.targetLabel) ?: false
                    if (success) "Entered '${parsed.text}' into '${parsed.targetLabel}'" else "Typed '${parsed.text}'"
                } else {
                    val success = service?.typeText(parsed.targetIndex, parsed.text) ?: false
                    if (success) "Entered '${parsed.text}' into [${parsed.targetIndex}]" else "Inspected field [${parsed.targetIndex}]"
                }
            }
            "DONE" -> parsed.text.ifEmpty { "Completed!" }
            else -> parsed.rawAction
        }

        return ActionStep(
            type = parsed.type,
            targetId = parsed.targetIndex,
            targetName = actionSummary,
            thought = thoughtStr,
            action = actionStr,
            realScreenDumping = compactScreen
        )
    }

    private fun addUserBubble(text: String, docName: String? = null) {
        val userLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.END
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { setMargins(0, 10, 0, 10) }
            alpha = 0f
            translationY = 30f
            animate().alpha(1f).translationY(0f).setDuration(220).start()
        }

        if (!docName.isNullOrEmpty()) {
            val docBadge = TextView(this).apply {
                this.text = "📎 Attached: $docName"
                setTextColor(Color.parseColor("#E0E0FF"))
                textSize = 10f
                setBackgroundResource(R.drawable.bg_doc_badge)
                setPadding(18, 8, 18, 8)
                val lp = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { setMargins(0, 0, 4, 6) }
                layoutParams = lp
            }
            userLayout.addView(docBadge)
        }

        val bubble = TextView(this).apply {
            this.text = text
            setTextColor(Color.WHITE)
            textSize = 13f
            setBackgroundResource(R.drawable.bg_bubble_user)
            setPadding(30, 20, 30, 20)
            maxWidth = 720
        }
        userLayout.addView(bubble)
        chatMessagesContainer.addView(userLayout)
        chatScrollView.post { chatScrollView.fullScroll(View.FOCUS_DOWN) }
    }

    private fun resetConversation() {
        conversationHistory.clear()
        clearAttachedDocument()
        runtime.stopGeneration()

        val count = chatMessagesContainer.childCount
        if (count > 1) {
            chatMessagesContainer.animate().alpha(0f).setDuration(160).withEndAction {
                chatMessagesContainer.removeViews(1, count - 1)
                quickSuggestions.visibility = View.VISIBLE
                chatMessagesContainer.alpha = 1f
                headerSubtitle.text = "Fresh conversation started ✓"
                headerSubtitle.setTextColor(Color.parseColor("#10B981"))
                headerSubtitle.postDelayed({
                    updateHeaderStatus()
                }, 2200)
            }.start()
        } else {
            Toast.makeText(this, "New conversation started", Toast.LENGTH_SHORT).show()
        }
    }

    private fun loadLocalWeights() {
        thread {
            val dir = getExternalFilesDir(null)
            val eerFile = File(dir, "maria_1b_android_executive_merged.eer")
            val targetFile = if (eerFile.exists()) eerFile else File(dir, "maria_1b.eer")
            android.util.Log.i("MainActivity", "Loading local model: ${targetFile.absolutePath} (exists=${targetFile.exists()}, size=${targetFile.length()})")
            val ok = runtime.loadModel(targetFile.absolutePath)
            isModelReady = ok
            runOnUiThread {
                updateHeaderStatus()
                pendingQuery?.let { q ->
                    pendingQuery = null
                    submitUserQuery(q)
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (instance == this) instance = null
        speechRecognizer?.destroy()
    }
}
