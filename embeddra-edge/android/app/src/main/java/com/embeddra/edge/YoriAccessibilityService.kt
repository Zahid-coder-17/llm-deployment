package com.embeddra.edge

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.graphics.Path
import android.graphics.PixelFormat
import android.graphics.Rect
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.TextUtils
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import android.widget.LinearLayout
import android.widget.TextView

class YoriAccessibilityService : AccessibilityService() {

    companion object {
        private const val TAG = "YoriAccessibility"
        var instance: YoriAccessibilityService? = null
            private set

        val isServiceActive: Boolean
            get() = instance != null
    }

    private val currentElements = mutableListOf<AccessibilityNodeInfo>()
    var lastPackageName: String = "unknown"

    private var windowManager: WindowManager? = null
    private var hudView: View? = null
    private var hudTextView: TextView? = null
    private val mainHandler = Handler(Looper.getMainLooper())
    private var hideRunnable: Runnable? = null

    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this
        windowManager = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        Log.i(TAG, "YORI Accessibility Service Connected & Active!")
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        event?.packageName?.let {
            lastPackageName = it.toString()
        }
    }

    override fun onInterrupt() {
        Log.w(TAG, "YORI Accessibility Service Interrupted.")
    }

    override fun onDestroy() {
        super.onDestroy()
        hideRunnable?.let { mainHandler.removeCallbacks(it) }
        removeFloatingHud()
        if (instance == this) {
            instance = null
        }
        Log.i(TAG, "YORI Accessibility Service Destroyed.")
    }

    fun showFloatingHud(text: String) {
        updateFloatingHud(text)
    }

    fun updateFloatingHud(text: String) {
        mainHandler.post {
            hideRunnable?.let { mainHandler.removeCallbacks(it) }
            hideRunnable = null

            if (hudView == null) {
                try {
                    val wm = windowManager ?: (getSystemService(Context.WINDOW_SERVICE) as WindowManager).also { windowManager = it }
                    val view = createHudView()
                    hudView = view
                    val params = WindowManager.LayoutParams(
                        WindowManager.LayoutParams.WRAP_CONTENT,
                        WindowManager.LayoutParams.WRAP_CONTENT,
                        WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY,
                        WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                            WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL or
                            WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
                        PixelFormat.TRANSLUCENT
                    ).apply {
                        gravity = Gravity.TOP or Gravity.CENTER_HORIZONTAL
                        y = 110
                    }
                    val displayMetrics = resources.displayMetrics
                    hudTextView?.maxWidth = (displayMetrics.widthPixels * 0.88).toInt()
                    hudTextView?.text = text
                    wm.addView(view, params)
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to add floating HUD: ${e.message}")
                }
            } else {
                try {
                    hudTextView?.text = text
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to update floating HUD: ${e.message}")
                }
            }
        }
    }

    fun hideFloatingHud(delayMs: Long = 0) {
        mainHandler.post {
            hideRunnable?.let { mainHandler.removeCallbacks(it) }
            val r = Runnable {
                removeFloatingHud()
            }
            hideRunnable = r
            if (delayMs > 0) {
                mainHandler.postDelayed(r, delayMs)
            } else {
                r.run()
            }
        }
    }

    private fun removeFloatingHud() {
        val view = hudView
        hudView = null
        hudTextView = null
        if (view != null) {
            try {
                if (view.isAttachedToWindow) {
                    windowManager?.removeViewImmediate(view)
                } else {
                    windowManager?.removeView(view)
                }
            } catch (e: Exception) {
                Log.w(TAG, "Notice removing floating HUD: ${e.message}")
            }
        }
    }

    private fun createHudView(): View {
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            val bg = GradientDrawable().apply {
                shape = GradientDrawable.RECTANGLE
                cornerRadius = 48f
                setColor(Color.parseColor("#E014131D"))
                setStroke(2, Color.parseColor("#50FFFFFF"))
            }
            background = bg
            setPadding(32, 14, 20, 14)
            elevation = 16f
            isClickable = true
            isFocusable = true
            setOnClickListener {
                try {
                    val launchIntent = packageManager.getLaunchIntentForPackage(packageName)?.apply {
                        flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_REORDER_TO_FRONT
                    }
                    if (launchIntent != null) startActivity(launchIntent)
                } catch (_: Exception) {}
            }
        }

        val textView = TextView(this).apply {
            setTextColor(Color.WHITE)
            textSize = 12f
            maxLines = 2
            ellipsize = TextUtils.TruncateAt.END
            setLineSpacing(4f, 1.1f)
        }
        hudTextView = textView
        container.addView(textView)

        val stopButton = TextView(this).apply {
            text = " ■ Stop"
            setTextColor(Color.parseColor("#FF6B6B"))
            textSize = 11f
            setTypeface(null, android.graphics.Typeface.BOLD)
            setPadding(18, 6, 10, 6)
            isClickable = true
            isFocusable = true
            setOnClickListener {
                MainActivity.instance?.stopCurrentGeneration()
                hideFloatingHud(100)
            }
        }
        container.addView(stopButton)

        return container
    }

    /**
     * Dumps the REAL on-device view hierarchy of whatever app is currently on the phone screen.
     * Returns: (Formatted Screen Hierarchy String, List of Actionable Nodes)
     */
    fun dumpRealScreen(): String {
        currentElements.clear()
        val root = rootInActiveWindow ?: return "[SCREEN: empty]\n(No active window detected)"

        val pkgName = root.packageName?.toString() ?: lastPackageName
        val sb = StringBuilder()
        sb.append("[SCREEN: $pkgName]\n")

        val bounds = Rect()
        collectNodes(root, currentElements, sb, bounds)
        return sb.toString().trimEnd()
    }

    private fun collectNodes(
        node: AccessibilityNodeInfo?,
        elements: MutableList<AccessibilityNodeInfo>,
        sb: StringBuilder,
        rect: Rect
    ) {
        if (node == null) return

        val text = node.text?.toString()?.trim() ?: ""
        val desc = node.contentDescription?.toString()?.trim() ?: ""
        val displayLabel = if (text.isNotEmpty()) text else desc
        if (displayLabel.startsWith("⚡ Yori") || displayLabel.contains("Yori:")) return
        val className = node.className?.toString()?.substringAfterLast(".") ?: "View"
        val isClickable = node.isClickable
        val isCheckable = node.isCheckable
        val isChecked = node.isChecked
        val isEditable = node.isEditable

        node.getBoundsInScreen(rect)

        // Only include meaningful interactive or informative elements
        val isMeaningful = displayLabel.isNotEmpty() || isClickable || isCheckable || isEditable

        if (isMeaningful && rect.width() > 0 && rect.height() > 0) {
            elements.add(node)
            val index = elements.size
            val stateStr = if (isCheckable) (if (isChecked) " (ON)" else " (OFF)") else ""
            val typeStr = when {
                isCheckable -> "Switch"
                isEditable -> "EditText"
                isClickable -> "Button"
                else -> className
            }
            sb.append("[$index] $typeStr \"$displayLabel\"$stateStr\n")
        }

        for (i in 0 until node.childCount) {
            collectNodes(node.getChild(i), elements, sb, rect)
        }
    }

    /**
     * Executes a REAL click on the physical screen node.
     */
    fun clickElement(index: Int): Boolean {
        if (index < 1 || index > currentElements.size) {
            Log.e(TAG, "Cannot click: index $index out of bounds (1..${currentElements.size})")
            return false
        }
        val node = currentElements[index - 1]

        // 1. Try standard accessibility click
        var curr: AccessibilityNodeInfo? = node
        while (curr != null) {
            if (curr.isClickable) {
                val success = curr.performAction(AccessibilityNodeInfo.ACTION_CLICK)
                if (success) {
                    Log.i(TAG, "Clicked element [$index] via ACTION_CLICK")
                    return true
                }
            }
            curr = curr.parent
        }

        // 2. Fallback: Dispatch real hardware touch gesture at bounding box center
        val rect = Rect()
        node.getBoundsInScreen(rect)
        if (rect.width() > 0 && rect.height() > 0) {
            val cx = rect.centerX().toFloat()
            val cy = rect.centerY().toFloat()
            return dispatchTapGesture(cx, cy)
        }

        return false
    }

    fun goHome(): Boolean = performGlobalAction(GLOBAL_ACTION_HOME)
    fun goBack(): Boolean = performGlobalAction(GLOBAL_ACTION_BACK)
    fun openRecents(): Boolean = performGlobalAction(GLOBAL_ACTION_RECENTS)
    fun openNotifications(): Boolean = performGlobalAction(GLOBAL_ACTION_NOTIFICATIONS)
    fun openQuickSettings(): Boolean = performGlobalAction(GLOBAL_ACTION_QUICK_SETTINGS)
    fun openPowerDialog(): Boolean = performGlobalAction(GLOBAL_ACTION_POWER_DIALOG)

    fun takeScreenshot(): Boolean {
        return if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.P) {
            performGlobalAction(GLOBAL_ACTION_TAKE_SCREENSHOT)
        } else false
    }

    fun lockScreen(): Boolean {
        return if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.P) {
            performGlobalAction(GLOBAL_ACTION_LOCK_SCREEN)
        } else false
    }

    fun scrollDown(): Boolean = dispatchSwipe(540f, 1600f, 540f, 500f, 300)
    fun scrollUp(): Boolean = dispatchSwipe(540f, 600f, 540f, 1700f, 300)
    fun swipeLeft(): Boolean = dispatchSwipe(900f, 1100f, 180f, 1100f, 250)
    fun swipeRight(): Boolean = dispatchSwipe(180f, 1100f, 900f, 1100f, 250)

    private fun dispatchSwipe(x1: Float, y1: Float, x2: Float, y2: Float, durationMs: Long): Boolean {
        val path = Path().apply {
            moveTo(x1, y1)
            lineTo(x2, y2)
        }
        val stroke = GestureDescription.StrokeDescription(path, 0, durationMs)
        val gesture = GestureDescription.Builder().addStroke(stroke).build()
        return dispatchGesture(gesture, null, null)
    }

    fun clickElementByText(textQuery: String): Boolean {
        val root = rootInActiveWindow ?: return false
        val qLower = textQuery.lowercase().trim()
        val queue = java.util.ArrayDeque<AccessibilityNodeInfo>()
        queue.add(root)

        val candidateNodes = mutableListOf<AccessibilityNodeInfo>()
        val nodeRect = Rect()
        while (queue.isNotEmpty()) {
            val node = queue.poll() ?: continue
            val text = (node.text?.toString() ?: "").lowercase()
            val desc = (node.contentDescription?.toString() ?: "").lowercase()

            node.getBoundsInScreen(nodeRect)
            val isTopSearchInput = nodeRect.bottom <= 260 && (node.isEditable || (text == qLower && qLower.contains(" ")))

            if (!isTopSearchInput && (text.contains(qLower) || desc.contains(qLower))) {
                if (clickNode(node)) return true
            }

            if (!isTopSearchInput && (text.isNotEmpty() || desc.isNotEmpty() || node.isClickable)) {
                candidateNodes.add(node)
            }

            for (i in 0 until node.childCount) {
                node.getChild(i)?.let { queue.add(it) }
            }
        }

        // Fuzzy fallback: match by token / word overlap
        val qWords = qLower.split(Regex("[^a-zA-Z0-9]+")).filter { it.length >= 2 }
        var bestNode: AccessibilityNodeInfo? = null
        var bestScore = 0

        for (node in candidateNodes) {
            val combined = ((node.text?.toString() ?: "") + " " + (node.contentDescription?.toString() ?: "")).lowercase()
            var score = 0
            for (w in qWords) {
                if (combined.contains(w)) {
                    score += 3
                } else if (combined.split(Regex("[^a-zA-Z0-9]+")).any { it.startsWith(w.take(3)) || w.startsWith(it.take(3)) }) {
                    score += 1
                }
            }
            if (score > bestScore) {
                bestScore = score
                bestNode = node
            }
        }

        if (bestNode != null && bestScore > 0) {
            Log.i(TAG, "clickElementByText: fuzzy matched query '$textQuery' to '${bestNode.text ?: bestNode.contentDescription}' with score $bestScore")
            return clickNode(bestNode)
        }

        return false
    }

    fun clickNode(node: AccessibilityNodeInfo): Boolean {
        val rect = Rect()
        node.getBoundsInScreen(rect)
        if (rect.width() > 0 && rect.height() > 0 && rect.centerY() > 100 && rect.centerY() < 2320) {
            val cx = rect.centerX().toFloat()
            val cy = rect.centerY().toFloat()
            if (dispatchTapGesture(cx, cy)) return true
        }
        var curr: AccessibilityNodeInfo? = node
        while (curr != null) {
            if (curr.isClickable) {
                if (curr.performAction(AccessibilityNodeInfo.ACTION_CLICK)) return true
            }
            curr = curr.parent
        }
        if (rect.width() > 0 && rect.height() > 0) {
            return dispatchTapGesture(rect.centerX().toFloat(), rect.centerY().toFloat())
        }
        return false
    }

    fun typeIntoFocusedOrTarget(text: String, targetQuery: String? = null): Boolean {
        val root = rootInActiveWindow ?: return false
        val queue = java.util.ArrayDeque<AccessibilityNodeInfo>()
        queue.add(root)

        var targetEditText: AccessibilityNodeInfo? = null

        while (queue.isNotEmpty()) {
            val node = queue.poll() ?: continue
            if (node.isEditable) {
                if (targetQuery == null) {
                    targetEditText = node
                    break
                } else {
                    val hint = (node.hintText?.toString() ?: "").lowercase()
                    val desc = (node.contentDescription?.toString() ?: "").lowercase()
                    if (hint.contains(targetQuery.lowercase()) || desc.contains(targetQuery.lowercase())) {
                        targetEditText = node
                        break
                    }
                    if (targetEditText == null) targetEditText = node
                }
            }
            for (i in 0 until node.childCount) {
                node.getChild(i)?.let { queue.add(it) }
            }
        }

        if (targetEditText == null && !targetQuery.isNullOrEmpty()) {
            var clicked = clickElementByText(targetQuery)
            if (!clicked) {
                // Compose/placeholder fallback: match common chat/search input bar labels
                val inputSynonyms = listOf("Ask ChatGPT", "Message", "Type a message", "Search", "Write", "Chat")
                for (syn in inputSynonyms) {
                    if (clickElementByText(syn)) {
                        clicked = true
                        break
                    }
                }
            }
            if (clicked) {
                Thread.sleep(600)
                val newRoot = rootInActiveWindow
                if (newRoot != null) {
                    val q2 = java.util.ArrayDeque<AccessibilityNodeInfo>()
                    q2.add(newRoot)
                    while (q2.isNotEmpty()) {
                        val n = q2.poll() ?: continue
                        if (n.isEditable || n.isFocused) {
                            targetEditText = n
                            break
                        }
                        for (i in 0 until n.childCount) {
                            n.getChild(i)?.let { q2.add(it) }
                        }
                    }
                }
            }
        }

        if (targetEditText != null) {
            targetEditText.performAction(AccessibilityNodeInfo.ACTION_FOCUS)
            val args = Bundle().apply {
                putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text)
            }
            val ok = targetEditText.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)
            if (ok && android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
                val q = (targetQuery ?: "").lowercase()
                if (q.contains("search") || q.contains("query") || q.contains("url") || q.contains("address")) {
                    targetEditText.performAction(AccessibilityNodeInfo.AccessibilityAction.ACTION_IME_ENTER.id)
                }
            }
            return ok
        }
        return false
    }

    fun readVisibleChatMessages(): List<String> {
        val root = rootInActiveWindow ?: return emptyList()
        val messages = mutableListOf<String>()
        val queue = java.util.ArrayDeque<AccessibilityNodeInfo>()
        queue.add(root)

        val ignoreList = setOf("chats", "calls", "updates", "tools", "search…", "camera", "more options", "new chat", "archived")

        while (queue.isNotEmpty()) {
            val node = queue.poll() ?: continue
            val text = node.text?.toString()?.trim() ?: ""
            val desc = node.contentDescription?.toString()?.trim() ?: ""
            val label = if (text.isNotEmpty()) text else desc

            if (label.isNotEmpty() && !label.startsWith("http") && label.length > 1) {
                if (label.lowercase() !in ignoreList && label !in messages) {
                    messages.add(label)
                }
            }

            for (i in 0 until node.childCount) {
                node.getChild(i)?.let { queue.add(it) }
            }
        }
        return messages
    }

    /**
     * Types REAL text into the physical screen's input field.
     */
    fun typeText(index: Int, text: String): Boolean {
        if (index < 1 || index > currentElements.size) return false
        val node = currentElements[index - 1]

        node.performAction(AccessibilityNodeInfo.ACTION_FOCUS)
        val args = Bundle().apply {
            putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text)
        }
        return node.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)
    }

    private fun dispatchTapGesture(x: Float, y: Float): Boolean {
        val path = Path().apply { moveTo(x, y) }
        val stroke = GestureDescription.StrokeDescription(path, 0, 50)
        val gesture = GestureDescription.Builder().addStroke(stroke).build()
        return dispatchGesture(gesture, null, null)
    }
}

