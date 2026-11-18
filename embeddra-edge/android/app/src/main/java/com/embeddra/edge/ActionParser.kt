package com.embeddra.edge

import java.util.regex.Pattern

data class ParsedAction(
    val type: String,        // CLICK, TYPE, SCROLL, PRESS_KEY, WAIT, DONE, INVALID
    val targetIndex: Int = 0,
    val targetLabel: String = "",
    val text: String = "",
    val direction: String = "",
    val key: String = "",
    val rawAction: String = ""
)

object ActionParser {

    private val CLICK_PATTERN = Pattern.compile("^CLICK\\(\\[(\\d+)\\]\\)$")
    private val CLICK_LABEL_PATTERN = Pattern.compile("^CLICK\\(\"(.*)\"\\)$", Pattern.DOTALL)
    private val TYPE_PATTERN = Pattern.compile("^TYPE\\(\\[(\\d+)\\],\\s*\"(.*)\"\\)$", Pattern.DOTALL)
    private val TYPE_LABEL_PATTERN = Pattern.compile("^TYPE\\(\"(.*?)\",\\s*\"(.*)\"\\)$", Pattern.DOTALL)
    private val SCROLL_PATTERN = Pattern.compile("^SCROLL\\((UP|DOWN|LEFT|RIGHT)\\)$", Pattern.CASE_INSENSITIVE)
    private val KEY_PATTERN = Pattern.compile("^PRESS_KEY\\((BACK|HOME|ENTER)\\)$", Pattern.CASE_INSENSITIVE)
    private val WAIT_PATTERN = Pattern.compile("^WAIT\\((\\d+)\\)$")
    private val DONE_PATTERN = Pattern.compile("^DONE\\(\"(.*)\"\\)$", Pattern.DOTALL)

    fun parse(actionStr: String): ParsedAction {
        val clean = actionStr.trim()

        val mClick = CLICK_PATTERN.matcher(clean)
        if (mClick.matches()) {
            return ParsedAction("CLICK", targetIndex = mClick.group(1).toInt(), rawAction = clean)
        }

        val mClickLabel = CLICK_LABEL_PATTERN.matcher(clean)
        if (mClickLabel.matches()) {
            return ParsedAction("CLICK", targetLabel = mClickLabel.group(1) ?: "", rawAction = clean)
        }

        val mType = TYPE_PATTERN.matcher(clean)
        if (mType.matches()) {
            return ParsedAction("TYPE", targetIndex = mType.group(1).toInt(), text = mType.group(2) ?: "", rawAction = clean)
        }

        val mTypeLabel = TYPE_LABEL_PATTERN.matcher(clean)
        if (mTypeLabel.matches()) {
            return ParsedAction("TYPE", targetLabel = mTypeLabel.group(1) ?: "", text = mTypeLabel.group(2) ?: "", rawAction = clean)
        }

        val mScroll = SCROLL_PATTERN.matcher(clean)
        if (mScroll.matches()) {
            return ParsedAction("SCROLL", direction = mScroll.group(1).uppercase(), rawAction = clean)
        }

        val mKey = KEY_PATTERN.matcher(clean)
        if (mKey.matches()) {
            return ParsedAction("PRESS_KEY", key = mKey.group(1).uppercase(), rawAction = clean)
        }

        val mDone = DONE_PATTERN.matcher(clean)
        if (mDone.matches()) {
            return ParsedAction("DONE", text = mDone.group(1) ?: "", rawAction = clean)
        }

        // Robust fallback parser for LLM variations (single quotes, trailing semicolons/commas, loose spacing)
        val tool = clean.substringBefore("(").trim().uppercase()
        val inside = clean.substringAfter("(", "").substringBeforeLast(")", "").trim()
        val args = mutableListOf<String>()
        val argMatcher = Pattern.compile("[\"']([^\"']*)[\"']|([^,\\s]+)").matcher(inside)
        while (argMatcher.find()) {
            args.add(argMatcher.group(1) ?: argMatcher.group(2) ?: "")
        }

        when (tool) {
            "CLICK" -> {
                val target = args.firstOrNull() ?: ""
                val idx = target.removeSurrounding("[", "]").toIntOrNull()
                return if (idx != null) {
                    ParsedAction("CLICK", targetIndex = idx, rawAction = clean)
                } else {
                    ParsedAction("CLICK", targetLabel = target, rawAction = clean)
                }
            }
            "TYPE" -> {
                val target = args.firstOrNull() ?: ""
                val text = if (args.size > 1) args.subList(1, args.size).joinToString(", ") else ""
                val idx = target.removeSurrounding("[", "]").toIntOrNull()
                return if (idx != null) {
                    ParsedAction("TYPE", targetIndex = idx, text = text, rawAction = clean)
                } else {
                    ParsedAction("TYPE", targetLabel = target, text = text, rawAction = clean)
                }
            }
            "SCROLL", "SCROLL_DOWN", "SCROLL_UP" -> {
                val dir = args.firstOrNull()?.uppercase() ?: if (tool.contains("UP")) "UP" else "DOWN"
                return ParsedAction("SCROLL", direction = dir, rawAction = clean)
            }
            "PRESS_KEY" -> {
                val key = args.firstOrNull()?.uppercase() ?: "BACK"
                return ParsedAction("PRESS_KEY", key = key, rawAction = clean)
            }
            "DONE" -> {
                return ParsedAction("DONE", text = args.firstOrNull() ?: "Completed", rawAction = clean)
            }
            "WAIT" -> {
                return ParsedAction("WAIT", rawAction = clean)
            }
        }

        return ParsedAction("INVALID", rawAction = clean)
    }
}

