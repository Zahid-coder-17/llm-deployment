with open(r'embeddra-edge\android\app\src\main\java\com\embeddra\edge\MainActivity.kt', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Fix dumpRealScreen parsing
t1 = """        for (turn in 1..maxTurns) {
            val (screenHeader, rawScreenLines) = service?.dumpHierarchy() ?: ("[SCREEN: active]" to emptyList())"""

r1 = """        for (turn in 1..maxTurns) {
            val rawDump = service?.dumpRealScreen() ?: "[SCREEN: active]"
            val lines = rawDump.lines().map { it.trim() }.filter { it.isNotEmpty() }
            val screenHeader = lines.firstOrNull() ?: "[SCREEN: active]"
            val rawScreenLines = if (lines.size > 1) lines.subList(1, lines.size) else emptyList<String>()"""

# 2. Fix actions calling service methods
t2 = """            "SCROLL", "SCROLL_DOWN" -> {
                val ok = service?.scrollForward() ?: false
                return ActionStep("SCROLL", 0, if (ok) "⬇️ Scrolled down" else "Reached bottom", thoughtStr, actionStr, compactScreen)
            }
            "SCROLL_UP" -> {
                val ok = service?.scrollBackward() ?: false
                return ActionStep("SCROLL_UP", 0, if (ok) "⬆️ Scrolled up" else "Reached top", thoughtStr, actionStr, compactScreen)
            }
            "PRESS_KEY" -> {
                val key = parsed.text.uppercase()
                val ok = when {
                    key.contains("BACK") -> service?.pressBack() ?: false
                    key.contains("HOME") -> service?.pressHome() ?: false
                    else -> false
                }
                return ActionStep("PRESS_KEY", 0, if (ok) "Navigated $key" else "Key $key failed", thoughtStr, actionStr, compactScreen)
            }
        }

        val actionSummary = when (parsed.type) {
            "CLICK", "TAP" -> {
                if (parsed.targetIndex == -1 && parsed.targetLabel.isNotEmpty()) {
                    val success = service?.clickByLabel(parsed.targetLabel) ?: false
                    if (success) "Tapped '${parsed.targetLabel}'" else "Searched for '${parsed.targetLabel}'"
                } else {
                    val success = service?.clickNode(parsed.targetIndex) ?: false
                    if (success) "Tapped element [${parsed.targetIndex}]" else "Targeted element [${parsed.targetIndex}]"
                }
            }
            "TYPE", "INPUT_TEXT" -> {
                if (parsed.targetIndex == -1 && parsed.targetLabel.isNotEmpty()) {
                    val success = service?.typeByLabel(parsed.targetLabel, parsed.text) ?: false
                    if (success) "Entered '${parsed.text}' into '${parsed.targetLabel}'" else "Typed '${parsed.text}'"
                } else {
                    val success = service?.typeText(parsed.targetIndex, parsed.text) ?: false
                    if (success) "Entered '${parsed.text}' into [${parsed.targetIndex}]" else "Inspected field [${parsed.targetIndex}]"
                }
            }"""

r2 = """            "SCROLL", "SCROLL_DOWN" -> {
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
            }"""

# 3. Fix runtime.resetKVCache() call
t3 = """    private fun resetConversation() {
        conversationHistory.clear()
        clearAttachedDocument()
        runtime.resetKVCache()"""

r3 = """    private fun resetConversation() {
        conversationHistory.clear()
        clearAttachedDocument()
        runtime.stopGeneration()"""

assert t1 in content, "t1 not found"
assert t2 in content, "t2 not found"
assert t3 in content, "t3 not found"

content = content.replace(t1, r1).replace(t2, r2).replace(t3, r3)

with open(r'embeddra-edge\android\app\src\main\java\com\embeddra\edge\MainActivity.kt', 'w', encoding='utf-8') as f:
    f.write(content)
print("PATCH COMPLETED!")
