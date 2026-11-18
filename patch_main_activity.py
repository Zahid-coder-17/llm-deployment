with open(r'embeddra-edge\android\app\src\main\java\com\embeddra\edge\MainActivity.kt', 'r', encoding='utf-8') as f:
    content = f.read()

target = """                            if (isExternalApp) {
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

                        val afterThought = if (liveText.contains("</thought>")) liveText.substringAfter("</thought>").trimStart() else ""
                        val finishedChat = !isUiTurn && liveText.contains("</thought>") && afterThought.length >= 4 && !afterThought.startsWith("<")

                        if (isFinal || liveText.contains("</action>") || finishedChat) {
                            if (liveText.contains("</action>") || finishedChat) {
                                runtime.stopGeneration()
                            }
                            latch.countDown()
                        }"""

replacement = """                            val isGoalDriven = liveText.contains("🎯 Goal:") || liveText.contains("<action")
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
                        }"""

assert target in content, "Target still not matched!"
new_content = content.replace(target, replacement)
with open(r'embeddra-edge\android\app\src\main\java\com\embeddra\edge\MainActivity.kt', 'w', encoding='utf-8') as f:
    f.write(new_content)
print("SUCCESSFULLY PATCHED MainActivity.kt!")
